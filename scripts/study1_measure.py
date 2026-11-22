"""Study 1 energy model — 2,264-sample SHD test split, per-synapse delays (h=256).

All checkpoint paths are **relative to this file**, so the whole
Study1_Deliverable_v2 directory can be moved or copied anywhere:

    Study1_Deliverable_v2/
    ├── scripts/study1_measure.py      <- this file
    ├── checkpoints/                   <- 3 delay + 6 PMSN
    └── results/

Two external paths are configurable by environment variable, because they are
large and belong to the upstream repositories rather than this deliverable:

    DELREC_ROOT    default  D:\\DelRec        (DelRec checkout, for the model classes)
    SHD_TEST_H5    default  <DELREC_ROOT>/Datasets/SHD/extract/shd_test.h5

Five Study 1 categories (white paper p.11-12):
    1. synaptic operations   -> syn + neu
    2. communication         -> comm
    3. delay storage         -> buffer
    4. state memory          -> state
    5. leakage               -> leak (state + buffer)

Weight leakage is reported separately and excluded from the temporal-state share.

Time model: t_inf = (SOP + N*T*d_state) / (ncores * F),  par = 1, F = 2 GHz
Unit prices: E_AC=0.9, E_MAC=4.6, E_ACC=10 pJ, P_leak=11.9 nW/bit, E_hop=3.5 pJ
"""
import os
import pathlib
import sys

import h5py
import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                       # Study1_Deliverable_v2/
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"

sys.path.insert(0, str(HERE))

DELREC = pathlib.Path(os.environ.get("DELREC_ROOT", r"D:\DelRec"))
DATA = pathlib.Path(os.environ.get(
    "SHD_TEST_H5", str(DELREC / "Datasets" / "SHD" / "extract" / "shd_test.h5")))

if not DELREC.exists():
    raise SystemExit(
        f"DelRec checkout not found: {DELREC}\n"
        f"Set DELREC_ROOT to your clone, e.g.  export DELREC_ROOT=/content/DelRec")
if not DATA.exists():
    raise SystemExit(
        f"SHD test split not found: {DATA}\n"
        f"Set SHD_TEST_H5 to shd_test.h5")

sys.path.insert(0, str(DELREC))
sys.path.insert(0, str(DELREC / "src"))

from net.topology import Mesh                    # noqa: E402
from net.placement import Placement, layers_for, fanout_targets   # noqa: E402

N_TEST = -1   # -1 = all test samples

E_AC, E_MAC, E_ACC, P_LEAK, E_HOP, F = 0.9, 4.6, 10.0, 11.9e-9, 3.5, 2e9
MESHES = [("2x2", 4, 2), ("4x4", 16, 4), ("8x8", 64, 8)]

# Delay checkpoints: per-synapse (N,N) delays, hidden = [256, 256].
DELAY_RUNS = {
    50:  str(CKPT / "delay_T50"),
    100: str(CKPT / "delay_T100"),
    140: str(CKPT / "delay_T140"),
}
PMSN_RUNS = {
    ("lif", 50):  str(CKPT / "pmsn_lif_T50" / "checkpoint.pth"),
    ("mc", 50):   str(CKPT / "pmsn_mc_T50" / "checkpoint.pth"),
    ("lif", 100): str(CKPT / "pmsn_lif_T100" / "checkpoint.pth"),
    ("mc", 100):  str(CKPT / "pmsn_mc_T100" / "checkpoint.pth"),
    ("lif", 140): str(CKPT / "pmsn_lif_T140" / "checkpoint.pth"),
    ("mc", 140):  str(CKPT / "pmsn_mc_T140" / "checkpoint.pth"),
}
# delay: per-synapse recurrent delays, hidden = [256, 256].
#   params  = 140*256 + 256 + 256*2 (BN) + 256*256 + 256 + 256*2 (BN)
#             + 256*256 (recurrent w) + 256*256 (delays) + 256*20 + 20 = 239,124
#   neurons = 256 + 256 + 20 = 532
#   n_delay = 256*256 = 65,536   (synaptic_recdel.recurrent_delays shape)
#   n_buffer = 256               (y_window is (B, N, L-1); N = hidden2 width)
ARCH = {
    "delay": dict(params=239124, neurons=532, d_state=1,
                  n_delay=256 * 256, n_buffer=256),
    "lif":   dict(params=187988, neurons=704, d_state=1, n_delay=0),
    "mc":    dict(params=200660, neurons=704, d_state=8, n_delay=0),
}

# test accuracy on the full 2,264-sample SHD test split (from each final_test.json)
ACC = {("delay", 50): 84.54, ("delay", 100): 84.30, ("delay", 140): 83.81,
       ("lif", 50): 88.96, ("lif", 100): 86.40, ("lif", 140): 87.10,
       ("mc", 50): 91.21, ("mc", 100): 88.30, ("mc", 140): 88.96}

# Buffer depth for the h=256 delay model: ceil(max_d) + 2, read off
# final_delays.npz of each checkpoint (max_d 35.1 / 41.6 / 37.2).
SLOTS = {50: 38, 100: 44, 140: 40}


def load_test():
    f = h5py.File(DATA, "r")
    return f["spikes"]["times"], f["spikes"]["units"], np.asarray(f["labels"], dtype=np.int64)


def bin_one(times_s, units, ts_ms, T, W=700, nb=5):
    t = 1000.0 * np.asarray(times_s, dtype=np.float64)
    if t.size == 0:
        return np.zeros((T, W // nb), dtype=np.float32)
    t = t - t[0]
    idx = (t // ts_ms).astype(np.int64)
    keep = idx < T
    idx, u = idx[keep], np.asarray(units, dtype=np.int64)[keep]
    fr = np.zeros((T, W), dtype=np.float32)
    np.add.at(fr, (idx, u), 1.0)
    return fr.reshape(T, W // nb, nb).sum(axis=2).astype(np.float32)


def make_hop(nc, mn=None):
    if mn is None:
        return [[abs(i - j) for j in range(nc)] for i in range(nc)]
    tab = Mesh(mn).path_table()
    h = [[0] * nc for _ in range(nc)]
    for u in range(nc):
        for v in range(nc):
            if u != v:
                p = tab[u][v]
                h[u][v] = len(p) - 1 if p else 0
    return h


def build_tabs():
    tabs = {}
    for cn, nc, mn in MESHES:
        hop = make_hop(nc, mn)
        pl = Placement(layers_for(hidden=352, out=20), n_cores=nc)
        tbl = {}
        for src, dss in (("hidden1", ["hidden2"]), ("hidden2", ["out", "hidden2"])):
            for dst in dss:
                d = [0] * nc
                L = pl.by_name(src)
                for k in range(L.width):
                    sc = pl.core_of(L.lo + k)
                    tg = fanout_targets(pl, src, [dst], sc)
                    d[sc] = sum(hop[sc][x] for x in tg)
                tbl[(src, dst)] = d
        tabs[cn] = (pl, tbl)
    return tabs


def hops_of(rows, pl, tbl):
    tot = 0
    for _t, at in rows.items():
        for sl, idxs in at.items():
            base = pl.by_name(sl).lo
            per = [a for (s2, _), a in tbl.items() if s2 == sl]
            for li in idxs:
                sc = pl.core_of(base + li)
                for a in per:
                    tot += a[sc]
    return tot


def measure_delay(T, times, units, tabs):
    from delay_model import DelayModel
    dm = DelayModel(run_dir=DELAY_RUNS[T], verbose=False,
                    model_cls="SNN_synaptic_recurrent_delays")
    dm.model.eval()
    mods = dict(dm.model.named_modules())
    ts_ms = float(dm.config.time_step)
    sop, spk = 0, 0
    fh = {c[0]: [] for c in MESHES}
    with torch.no_grad():
        for i in range(N_TEST):
            x = bin_one(times[i], units[i], ts_ms, T)
            # input -> hidden1 (140 channels -> 352 neurons): this layer is before
            # the first spike_registrator, so nothing above counts it.  Every input
            # event triggers 352 MACs, same spike-driven convention as the layers
            # below (sop += spikes * fanout).
            sop += int(round(float(x.sum()))) * 352
            dm.model(torch.from_numpy(x).unsqueeze(1).float())
            rows = {}
            for mi, key, fan in ((3, "hidden1", 352), (8, "hidden2", 372)):
                s = mods[f"layers.{mi}"].spikes
                if s is None:
                    continue
                s = np.asarray(s.cpu())
                if s.ndim == 3:
                    s = s[:, 0, :]
                cnt = int((s > 0.5).sum())
                spk += cnt
                sop += cnt * fan
                for t in range(T):
                    idx = np.nonzero(s[t] > 0.5)[0].tolist()
                    if idx:
                        rows.setdefault(t, {})[key] = idx
            for cn, _, _ in MESHES:
                pl, tbl = tabs[cn]
                fh[cn].append(hops_of(rows, pl, tbl))
    return sop / N_TEST, spk / N_TEST, {k: np.mean(v) for k, v in fh.items()}


def measure_pmsn(which, T, times, units, tabs):
    from pmsn_model import SHDNet_ours
    ck = torch.load(PMSN_RUNS[(which, T)], map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = SHDNet_ours(bn=getattr(cfg, "bn", True), d_input=cfg.d_input, d_model=cfg.d_model,
                      d_out=20, d_state=cfg.d_state, dropout=cfg.dropout,
                      lr=cfg.neuron_lr, T=cfg.T)
    net.load_state_dict(ck["best_net"], strict=False)
    net.eval()
    captured, handles = {}, []
    for name, m in net.named_modules():
        if type(m).__name__ == "PMSN_neuron":
            handles.append(m.register_forward_hook(
                lambda mod, inp, out, n=name: captured.__setitem__(n, out.detach())))
    ts_ms = 1400.0 / T
    sop, spk = 0, 0
    fh = {c[0]: [] for c in MESHES}
    with torch.no_grad():
        for i in range(N_TEST):
            captured.clear()
            x = bin_one(times[i], units[i], ts_ms, T)
            # input -> hidden1 (140 -> 352): same correction as measure_delay().
            sop += int(round(float(x.sum()))) * 352
            net(torch.from_numpy(x).unsqueeze(0).permute(0, 2, 1))
            rows = {}
            for k, v in captured.items():
                s = v.reshape(T, 1, -1)
                cnt = int((s > 0.5).sum())
                spk += cnt
                sop += cnt * (352 if "hidden1" in k else 20)
                key = "hidden1" if "hidden1" in k else "hidden2"
                for t in range(T):
                    idx = torch.nonzero(s[t, 0, :] > 0.5, as_tuple=False).flatten().tolist()
                    if idx:
                        rows.setdefault(t, {})[key] = idx
            for cn, _, _ in MESHES:
                pl, tbl = tabs[cn]
                fh[cn].append(hops_of(rows, pl, tbl))
    for h in handles:
        h.remove()
    return sop / N_TEST, spk / N_TEST, {k: np.mean(v) for k, v in fh.items()}


def energy(model, T, sop, fh, mesh, ncores, slots):
    a = ARCH[model]
    N, ds = a["neurons"], a["d_state"]
    total_mac = sop + N * T * ds
    t_inf = total_mac / (ncores * F)

    syn = sop * E_AC
    neu = N * T * ds * E_MAC
    state = N * T * ds * 2 * E_ACC
    # buffer access = indexing into the shared spike-history buffer.  It scales with
    # the number of neurons that OWN a recurrent delay (hidden2 width, 256), never
    # with the number of delay parameters: per-synapse stores n_delay=65536 but the
    # buffer is still (256, L) in delay_layers_pytorch.py.
    n_buffer = a.get("n_buffer", a["n_delay"])
    buffer = n_buffer * T * 2 * E_ACC

    # Two weight regimes, see README §5.5
    #   weights resident —— Dampfhoffer's assumption: weights are spatially expanded,
    #                       so one synaptic op costs only the E_AC arithmetic;
    #   weights in SRAM  —— every synaptic op additionally reads its 32-bit weight,
    #                       costing E_ACC. This is an UPPER BOUND: a real core reads
    #                       whole lines and reuses them, reuse=1 being the worst case.
    w_read = sop * E_ACC

    sb_bits = N * ds * 32 + a["n_delay"] * slots
    w_bits = a["params"] * 32
    leak_sb = sb_bits * P_LEAK * t_inf * 1e12
    leak_w = w_bits * P_LEAK * t_inf * 1e12
    comm = fh * E_HOP

    study1 = syn + neu + state + buffer + leak_sb + comm      # Study 1 五项（权重驻留）
    study1_sram = study1 + w_read                             # 权重在 SRAM
    ts = state + buffer + leak_sb                              # 时间状态
    return dict(t_inf=t_inf * 1e6, syn=syn, neu=neu, state=state, buffer=buffer,
                leak_sb=leak_sb, comm=comm, leak_w=leak_w, w_read=w_read,
                study1=study1, study1_sram=study1_sram, ts=ts,
                ts_share=ts / study1, ts_share_sram=ts / study1_sram)


def main():
    times, units, labels = load_test()
    global N_TEST
    if N_TEST < 0:
        N_TEST = len(labels)
    print(f"测试集 {len(labels)} 个，本次用 {N_TEST} 个\n")
    tabs = build_tabs()

    print(f"实测（{N_TEST} 样本平均）")
    print(f"{'model':<7}{'T':>5}{'spike':>11}{'SOP':>12}", end="")
    for cn, _, _ in MESHES:
        print(f"{'fh-' + cn:>13}", end="")
    print()

    RAW = {}
    for T in (50, 100, 140):
        s, sp, f = measure_delay(T, times, units, tabs)
        RAW[("delay", T)] = (s, sp, f)
        print(f"{'delay':<7}{T:>5}{sp:>11,.0f}{s:>12,}", end="")
        for cn, _, _ in MESHES:
            print(f"{f[cn]:>13,.0f}", end="")
        print()
    for w in ("lif", "mc"):
        for T in (50, 100, 140):
            s, sp, f = measure_pmsn(w, T, times, units, tabs)
            RAW[(w, T)] = (s, sp, f)
            print(f"{w.upper():<7}{T:>5}{sp:>11,.0f}{s:>12,}", end="")
            for cn, _, _ in MESHES:
                print(f"{f[cn]:>13,.0f}", end="")
            print()

    for T in (50, 100, 140):
        print("\n" + "=" * 118)
        print(f"  T = {T}   ({1400/T:.1f} ms/frame)")
        print("=" * 118)
        for cn, nc, _ in MESHES:
            E = {}
            for m in ("delay", "lif", "mc"):
                sop, spk, fhm = RAW[(m, T)]
                E[m] = energy(m, T, sop, fhm[cn], cn, nc, SLOTS[T])
            # 按 spike 数（最粗）与 SOP（中）两种排序
            s_spk = sorted(E, key=lambda m: RAW[(m, T)][1])
            s_sop = sorted(E, key=lambda m: RAW[(m, T)][0])
            s_en = sorted(E, key=lambda m: E[m]["study1"])            # 权重驻留
            s_en_sram = sorted(E, key=lambda m: E[m]["study1_sram"])  # 权重在 SRAM

            print(f"\n--- {cn} ({nc} cores)   t_inf(delay)={E['delay']['t_inf']:.1f} us ---")
            hdr = (f"{'model':<7}{'acc':>7}{'spike':>10}{'syn':>7}{'neu':>7}"
                   f"{'state':>8}{'buf':>7}{'leak_sb':>9}{'comm':>7}"
                   f"{'w_read':>9}{'resident':>10}{'ts%':>6}"
                   f"{'+wread':>9}{'ts%':>6}{'wleak':>9}")
            print(hdr)
            print("-" * len(hdr))
            for m in ("delay", "lif", "mc"):
                e = E[m]
                print(f"{m:<7}{ACC[(m,T)]:>6.2f}%{RAW[(m,T)][1]:>10,.0f}"
                      f"{e['syn']/1e6:>7.3f}{e['neu']/1e6:>7.3f}"
                      f"{e['state']/1e6:>8.3f}{e['buffer']/1e6:>7.3f}"
                      f"{e['leak_sb']/1e6:>9.4f}{e['comm']/1e6:>7.3f}"
                      f"{e['w_read']/1e6:>9.3f}"
                      f"{e['study1']/1e6:>10.3f}{100*e['ts_share']:>6.1f}%"
                      f"{e['study1_sram']/1e6:>9.3f}{100*e['ts_share_sram']:>6.1f}%"
                      f"{e['leak_w']/1e6:>9.3f}")
            print(f"  spike 序 : {' < '.join(m for m in s_spk)}")
            print(f"  SOP   序 : {' < '.join(m for m in s_sop)}")
            print(f"  能量序(resident) : {' < '.join(m for m in s_en)}")
            print(f"  能量序(SRAM)     : {' < '.join(m for m in s_en_sram)}")
            # 判据 (a) 按白皮书原文："a non-trivial share of energy in at least one
            # brain-inspired model" -> any()，不是 all()。
            a_res = any(E[m]["ts_share"] >= 0.10 for m in E)
            a_sram_ = any(E[m]["ts_share_sram"] >= 0.10 for m in E)
            print(f"  (a) ts>=10% (any)  resident:{'Y' if a_res else 'N'}"
                  f"  SRAM:{'Y' if a_sram_ else 'N'}"
                  f"   |  (b) SOP!=E  resident:{'Y' if s_sop != s_en else 'N'}"
                  f"  SRAM:{'Y' if s_sop != s_en_sram else 'N'}")

            OUT[f"T{T}_{cn}"] = {
                "ncores": nc, "t_inf_us": E["delay"]["t_inf"],
                "sop_order": s_sop, "energy_order": s_en, "spike_order": s_spk,
                "energy_order_sram": s_en_sram,
                "a": a_res,                                              # 白皮书口径
                "a_all": all(E[m]["ts_share"] >= 0.10 for m in E),       # 更严格口径，参考
                "a_sram": a_sram_,                                       # 白皮书口径
                "a_all_sram": all(E[m]["ts_share_sram"] >= 0.10 for m in E),
                "b_sop": s_sop != s_en, "b_spike": s_spk != s_en,
                "b_sop_sram": s_sop != s_en_sram, "b_spike_sram": s_spk != s_en_sram,
                "models": {
                    m: {
                        "acc_pct": ACC[(m, T)],
                        "spikes": RAW[(m, T)][1],
                        "sop": RAW[(m, T)][0],
                        "flit_hops": RAW[(m, T)][2][cn],
                        "energy_uJ": {
                            "synaptic": E[m]["syn"] / 1e6,
                            "neuron": E[m]["neu"] / 1e6,
                            "state": E[m]["state"] / 1e6,
                            "delay_storage": E[m]["buffer"] / 1e6,
                            "leakage_sb": E[m]["leak_sb"] / 1e6,
                            "comm": E[m]["comm"] / 1e6,
                            "weight_read": E[m]["w_read"] / 1e6,
                            "total": E[m]["study1"] / 1e6,
                            "total_sram": E[m]["study1_sram"] / 1e6,
                            "weights_leak": E[m]["leak_w"] / 1e6,
                        },
                        "ts_share": E[m]["ts_share"],
                        "ts_share_sram": E[m]["ts_share_sram"],
                    } for m in ("delay", "lif", "mc")
                },
            }

    if SAVE:
        import json
        out = {
            "_meta": {
                "generated_by": "scripts/study1_measure.py --save",
                "n_test_samples": N_TEST,
                "dataset": "SHD test split",
                "units": "energy in microjoules (uJ) per inference",
                "precision": "FP32 (32-bit floating point)",
                "note": "weights_leak reported separately, excluded from total",
                "weight_read_modes": {
                    "resident": "weights spatially expanded; a synaptic operation "
                                "costs only E_AC (Dampfhoffer 2023, L2372)",
                    "sram": "weights resident in SRAM; every synaptic operation also "
                            "pays one E_access (10 pJ) to read its 32-bit weight. "
                            "This is an UPPER bound: weight reuse would divide it.",
                    "fields": "total / ts_share = resident; "
                              "total_sram / ts_share_sram = weights in SRAM",
                },
                "prices": {k: v for k, v in (
                    ("E_AC_pJ", E_AC), ("E_MAC_pJ", E_MAC), ("E_access_pJ", E_ACC),
                    ("P_leak_W_per_bit", P_LEAK), ("E_hop_pJ", E_HOP),
                    ("F_Hz", F), ("parallelism", 1))},
                "architecture": ARCH,
            },
            "configs": OUT,
            "summary": {
                # (a) 按白皮书口径：at least one model
                "a_holds": f"{sum(v['a'] for v in OUT.values())}/{len(OUT)}",
                "a_holds_sram": f"{sum(v['a_sram'] for v in OUT.values())}/{len(OUT)}",
                # 更严格的 all-模型口径，仅作参考
                "a_all_holds": f"{sum(v['a_all'] for v in OUT.values())}/{len(OUT)}",
                "a_all_holds_sram": f"{sum(v['a_all_sram'] for v in OUT.values())}/{len(OUT)}",
                "b_sop_flip": f"{sum(v['b_sop'] for v in OUT.values())}/{len(OUT)}",
                "b_spike_flip": f"{sum(v['b_spike'] for v in OUT.values())}/{len(OUT)}",
                "b_either_flip": f"{sum(v['b_sop'] or v['b_spike'] for v in OUT.values())}/{len(OUT)}",
                "b_sop_flip_sram": f"{sum(v['b_sop_sram'] for v in OUT.values())}/{len(OUT)}",
                "b_spike_flip_sram": f"{sum(v['b_spike_sram'] for v in OUT.values())}/{len(OUT)}",
                "b_either_flip_sram": f"{sum(v['b_sop_sram'] or v['b_spike_sram'] for v in OUT.values())}/{len(OUT)}",
            },
        }
        with open(SAVE, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n已保存 -> {SAVE}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="", help="把结果写成 JSON 到这个路径")
    ap.add_argument("--n", type=int, default=-1, help="样本数，-1 = 全部")
    _a = ap.parse_args()
    SAVE = _a.save
    if _a.n > 0:
        N_TEST = _a.n
    OUT = {}
    main()
