"""Energy for the delay-free reference network (Study 1 accounting, TASK 4).

The delay-free model has the same topology as the per-synapse delay model but
recurrent delays fixed at 0 and BatchNorm removed:

    layers: Linear(140->256) -> LIF -> Linear(256->256) -> LIF(rec) -> Linear(256->20)
    recurrent_delays: 65536 values, all zero, frozen

So the delay table costs NOTHING (a constant zero needs no storage), and the
spike window only needs L=3 (t-1 and t-2), not L=55.

Run:
    python measure_nodelay.py --checkpoint ../checkpoints/nodelay_cos_10ep --samples 2264
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "net"))

from study1_measure import (E_AC, E_MAC, E_ACC, P_LEAK, E_HOP, F,
                            MESHES, load_test, bin_one, make_hop,
                            Placement, layers_for, fanout_targets)

NODELAY = dict(name="nodelay", params=172564, neurons=256, d_state=1,
               n_delay=0, n_buffer=256, width=256,
               fanout=dict(input=256, hidden1=256, hidden2=276))

# L = 3 is what delta-kernel recurrence needs: y[t-1] and y[t-2].
SLOTS_NODELAY = 3


def load_model(ckpt: pathlib.Path, T: int, ts_ms: float):
    """Load the delay-free checkpoint.

    The local DelRec checkout has no SNN_vanilla_synaptic_recurrent (that class
    exists only in the training bundle), so load the bundle's networks_shd as a
    standalone module instead of mutating the checkout.
    """
    import importlib.util

    bundle = ROOT.parent / "training_bundle" / "delrec" / "networks_shd.py"
    spec = importlib.util.spec_from_file_location("networks_shd_nodelay", bundle)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import configs.perf_SHD as cfg_mod

    cfg = cfg_mod.Config()
    saved = ckpt / "config.json"
    if saved.exists():
        for k, v in json.loads(saved.read_text()).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    cfg.hidden_layers = [256, 256]

    net = mod.SNN_vanilla_synaptic_recurrent(cfg)
    sd = torch.load(ckpt / "best.pth", map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"] if "model" in sd else sd)
    net.eval()
    return net, cfg


def measure(ckpt, T, times, units, n_samples, ts_ms=14.0):
    from delrec.delay_layers import synaptic_recdel

    net, cfg = load_model(ckpt, T, ts_ms)
    rec = next(m for m in net.modules() if isinstance(m, synaptic_recdel))

    n = min(n_samples, len(times))
    sop_total = 0
    spike_by_layer = {"hidden1": 0, "hidden2": 0}

    for i in range(n):
        x = bin_one(times[i], units[i], ts_ms, T)          # (T, 140)
        x = torch.from_numpy(x).unsqueeze(1)               # (T, 1, 140)
        with torch.no_grad():
            net(x)
        # layers.2 / layers.6 are the spike_registrator wrappers that actually
        # record spikes; LIFNode itself has no .spikes attribute.
        mods = dict(net.named_modules())
        h1 = mods["layers.2"].spikes
        h2 = mods["layers.6"].spikes
        s1 = int(h1.float().sum().item())
        s2 = int(h2.float().sum().item())
        spike_by_layer["hidden1"] += s1
        spike_by_layer["hidden2"] += s2
        # fanout: hidden1 spikes drive 256 MACs; hidden2 spikes drive 20 (out) + 256 (rec)
        sop_total += s1 * 256 + s2 * (20 + 256)

    return sop_total / n, spike_by_layer["hidden1"] / n, spike_by_layer["hidden2"] / n, rec


def energy(T, sop, fh, ncores, slots=None):
    slots = SLOTS_NODELAY if slots is None else slots
    a = NODELAY
    N, ds = a["neurons"], a["d_state"]
    total_mac = sop + N * T * ds
    t_inf = total_mac / (ncores * F)

    syn = sop * E_AC
    neu = N * T * ds * E_MAC
    state = N * T * ds * 2 * E_ACC
    buffer = a["n_buffer"] * T * 2 * E_ACC
    w_read = sop * E_ACC

    # bit accounting: no delay table at all
    sb_bits = N * ds * 32 + a["n_delay"] * 32 + a["n_buffer"] * max(slots - 1, 0)
    w_bits = (a["params"] - a["n_delay"]) * 32
    leak_sb = sb_bits * P_LEAK * t_inf * 1e12
    leak_w = w_bits * P_LEAK * t_inf * 1e12
    comm = fh * E_HOP

    s1 = syn + neu + state + buffer + leak_sb + comm
    s1_sram = s1 + w_read
    ts = state + buffer + leak_sb
    return dict(t_inf=t_inf * 1e6, sop=sop, fh=fh,
                syn=syn, neu=neu, state=state, buffer=buffer,
                leak_sb=leak_sb, comm=comm, leak_w=leak_w, w_read=w_read,
                study1=s1, study1_sram=s1_sram, ts=ts,
                sb_bits=sb_bits, w_bits=w_bits,
                ts_share=ts / s1, ts_share_sram=ts / s1_sram)


def fanout_tables(ncores, mn, hidden=256, out=20):
    """Build exactly the same fanout tables study1_measure.build_tabs builds.

    Returns {(src, dst): per_core_array} plus the placement, so the hop cost of a
    spike here is computed identically to the delay and PMSN models.
    """
    hop = make_hop(ncores, mn)
    pl = Placement(layers_for(hidden=hidden, out=out), n_cores=ncores)
    tbl = {}
    for src, dss in (("hidden1", ["hidden2"]), ("hidden2", ["out", "hidden2"])):
        for dst in dss:
            d = [0] * ncores
            Ls = pl.by_name(src)
            for k in range(Ls.width):
                sc = pl.core_of(Ls.lo + k)
                tg = fanout_targets(pl, src, [dst], sc)
                d[sc] = sum(hop[sc][x] for x in tg)
            tbl[(src, dst)] = d
    return hop, pl, tbl


def hop_cost(tbl, src, spikes_per_core):
    """Spikes emitted from each core -> total hops, summed over that layer's edges."""
    per = [a for (s2, _), a in tbl.items() if s2 == src]
    return sum(spikes_per_core[sc] * sum(a[sc] for a in per) for sc in range(len(spikes_per_core)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="../checkpoints/nodelay_cos_10ep")
    ap.add_argument("--samples", type=int, default=2264)
    ap.add_argument("--out", default="../results/nodelay_measure.json")
    args = ap.parse_args()

    ckpt = (HERE / args.checkpoint).resolve()
    print(f"checkpoint: {ckpt}")

    times, units, labels = load_test()
    print(f"test samples: {len(labels)}, using {args.samples}")

    out = {"checkpoint": str(ckpt), "samples": args.samples, "rows": []}
    for T in (50, 100, 140):
        sop, s1, s2, rec = measure(ckpt, T, times, units, args.samples)
        sigma = float(rec.sigma)
        L = int(np.ceil(1 + 0.0 + (1 + sigma))) + 1
        print(f"  T={T:>3}: SOP {sop:>12,.0f}  h1 spikes {s1:>7,.0f}  "
              f"h2 spikes {s2:>7,.0f}  sigma {sigma:.3f}  L {L}")

        for cn, ncores, mn in MESHES:
            hop, pl, tbl = fanout_tables(ncores, mn)
            # spread each layer's spikes uniformly over the cores its neurons occupy
            fh = 0.0
            for src, spk in (("hidden1", s1), ("hidden2", s2)):
                Ls = pl.by_name(src)
                cores = sorted({pl.core_of(Ls.lo + k) for k in range(Ls.width)})
                per_core = np.zeros(ncores)
                for c in cores:
                    per_core[c] = spk * (sum(1 for k in range(Ls.width)
                                             if pl.core_of(Ls.lo + k) == c) / Ls.width)
                fh += hop_cost(tbl, src, per_core)
            e = energy(T, sop, fh, ncores)
            e["mesh"] = cn
            e["T"] = T
            out["rows"].append(e)
            print(f"      {cn:<4} fh {fh:>12,.0f}  study1 {e['study1']/1e6:>8.2f} uJ  "
                  f"sram {e['study1_sram']/1e6:>8.2f}  ts% {100*e['ts_share']:>5.1f}  "
                  f"bits {e['sb_bits']:>9,}")

    p = (HERE / args.out).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
