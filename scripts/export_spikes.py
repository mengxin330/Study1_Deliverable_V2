"""Export the spike trains of all nine checkpoints (advisor Part B, item 3).

    "Export the spike trains of all checkpoints in one file format (NPZ or HDF5).
     Use these fields: sample, time step, layer, neuron."

One NPZ per checkpoint, same format for all nine.  The four stored arrays are
exactly the requested fields:

    sample     int16   index into the SHD test split, 0..2263
    time_step  int16   frame index, 0..T-1
    layer      int8    0 = hidden1, 1 = hidden2
    neuron     int16   index within that layer (0..255 for delay, 0..351 for PMSN)

Only spiking events are stored, so each array has one entry per spike.  The
checkpoint identity (model, T) lives in the filename, not in a field, because
the request lists exactly these four fields.

Usage:
    python export_spikes.py                 # all nine
    python export_spikes.py --only delay    # one model family
    python export_spikes.py --samples 200   # quick subset
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from study1_measure import (DELAY_RUNS, PMSN_RUNS, MESHES, load_test, bin_one)

OUT_DIR = HERE.parent / "checkpoints" / "spikes"

LAYER_ID = {"hidden1": 0, "hidden2": 1}


def delay_spikes(T: int, n_samples: int, times, units):
    """Yield (layer_name, spikes[T, N] bool) for the delay network."""
    from delay_model import DelayModel
    dm = DelayModel(run_dir=DELAY_RUNS[T], verbose=False,
                    model_cls="SNN_synaptic_recurrent_delays")
    dm.model.eval()
    mods = dict(dm.model.named_modules())
    ts_ms = float(dm.config.time_step)

    rows = {"sample": [], "time_step": [], "layer": [], "neuron": []}
    with torch.no_grad():
        for i in range(n_samples):
            x = bin_one(times[i], units[i], ts_ms, T)
            dm.model(torch.from_numpy(x).unsqueeze(1).float())
            for mi, key in ((3, "hidden1"), (8, "hidden2")):
                s = mods[f"layers.{mi}"].spikes
                if s is None:
                    continue
                s = np.asarray(s.cpu())
                if s.ndim == 3:
                    s = s[:, 0, :]
                tt, nn = np.nonzero(s > 0.5)
                k = tt.size
                rows["sample"].extend([i] * k)
                rows["time_step"].extend(tt.tolist())
                rows["layer"].extend([LAYER_ID[key]] * k)
                rows["neuron"].extend(nn.tolist())
            if (i + 1) % 200 == 0:
                print(f"      {i+1}/{n_samples}")
    return rows


def pmsn_spikes(which: str, T: int, n_samples: int, times, units):
    """Same, for PMSN's two variants."""
    from pmsn_model import SHDNet_ours
    ck = torch.load(PMSN_RUNS[(which, T)], map_location="cpu", weights_only=False)
    cfg = ck["config"]
    net = SHDNet_ours(bn=getattr(cfg, "bn", True), d_input=cfg.d_input,
                      d_model=cfg.d_model, d_out=20, d_state=cfg.d_state,
                      dropout=cfg.dropout, lr=cfg.neuron_lr, T=cfg.T)
    net.load_state_dict(ck["best_net"], strict=False)
    net.eval()

    captured, handles = {}, []
    for name, m in net.named_modules():
        if type(m).__name__ == "PMSN_neuron":
            handles.append(m.register_forward_hook(
                lambda mod, inp, out, n=name: captured.__setitem__(n, out.detach())))

    ts_ms = 1400.0 / T
    rows = {"sample": [], "time_step": [], "layer": [], "neuron": []}
    with torch.no_grad():
        for i in range(n_samples):
            captured.clear()
            x = bin_one(times[i], units[i], ts_ms, T)
            net(torch.from_numpy(x).unsqueeze(0).permute(0, 2, 1))
            for name, v in captured.items():
                key = "hidden1" if "hidden1" in name else "hidden2"
                s = v.reshape(T, -1)
                tt, nn = torch.nonzero(s > 0.5, as_tuple=True)
                k = int(tt.numel())
                if k == 0:
                    continue
                rows["sample"].extend([i] * k)
                rows["time_step"].extend(tt.cpu().tolist())
                rows["layer"].extend([LAYER_ID[key]] * k)
                rows["neuron"].extend(nn.cpu().tolist())
            if (i + 1) % 200 == 0:
                print(f"      {i+1}/{n_samples}")

    for h in handles:
        h.remove()
    return rows


def save(rows: dict, path: pathlib.Path) -> dict:
    arrs = {k: np.asarray(v, dtype=np.int16) for k, v in rows.items()}
    arrs["layer"] = np.asarray(rows["layer"], dtype=np.int8)
    np.savez_compressed(path, **arrs)
    return {k: int(arrs[k].size) for k in arrs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=-1)
    ap.add_argument("--only", default="all", choices=["all", "delay", "lif", "mc"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    times, units, labels = load_test()
    n = len(labels) if args.samples < 0 else min(args.samples, len(labels))
    print(f"test samples: {len(labels)}, exporting {n}")

    jobs = []
    if args.only in ("all", "delay"):
        for T in (50, 100, 140):
            jobs.append(("delay", T, delay_spikes))
    if args.only in ("all", "lif"):
        for T in (50, 100, 140):
            jobs.append(("lif", T, lambda T, n_, t_, u_: pmsn_spikes("lif", T, n_, t_, u_)))
    if args.only in ("all", "mc"):
        for T in (50, 100, 140):
            jobs.append(("mc", T, lambda T, n_, t_, u_: pmsn_spikes("mc", T, n_, t_, u_)))

    summary = {}
    for model, T, fn in jobs:
        print(f"\n[{model} T={T}]")
        rows = fn(T, n, times, units)
        path = OUT_DIR / f"spikes_{model}_T{T}.npz"
        counts = save(rows, path)
        mb = path.stat().st_size / 1e6
        summary[f"{model}_T{T}"] = {"events": counts["sample"],
                                    "file": path.name, "MB": round(mb, 1)}
        print(f"    {counts['sample']:,} events -> {path.name} ({mb:.1f} MB)")

    meta = OUT_DIR / "spikes_index.json"
    meta.write_text(json.dumps({
        "format": "NPZ, one file per checkpoint",
        "fields": {
            "sample": "int16, index into the SHD test split",
            "time_step": "int16, frame index 0..T-1",
            "layer": "int8, 0=hidden1 1=hidden2",
            "neuron": "int16, index within the layer",
        },
        "n_samples": n,
        "files": summary,
    }, indent=2), encoding="utf-8")
    print(f"\nindex -> {meta}")
    total = sum(v["MB"] for v in summary.values())
    print(f"total {total:.0f} MB across {len(summary)} files")


if __name__ == "__main__":
    main()
