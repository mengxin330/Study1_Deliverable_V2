"""Study 1 deliverable figures: stacked energy breakdown + spike counts overlaid.

Every number is read from the results JSON produced by ``study1_measure.py``,
so the figures cannot drift away from the reported tables.

White-paper request: "Stacked energy breakdown for the three models, with spike
counts overlaid."

Usage:
    python plot_study1.py [--json <path>] [--out <dir>]
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_JSON = HERE.parent / "results" / "study1_h256_measure_full.json"
DEFAULT_OUT = HERE.parent / "figs"

MODELS = ["delay", "lif", "mc"]
SHOW = {"delay": "Delay-trained\n(DelRec, 256, per-synapse)",
        "lif": "Plain LIF\n(PMSN d_state=1)",
        "mc": "Multi-compartment\n(PMSN d_state=8)"}
SHORT = {"delay": "DL", "lif": "LIF", "mc": "MC"}
COLOR = {"delay": "#C44E52", "lif": "#4C72B0", "mc": "#DD8452"}

# JSON energy key -> (figure part name, colour)
PARTS = [("synaptic",       "synaptic ops (AC)",       "#4C72B0"),
         ("neuron",         "neuron update",           "#7FA8D6"),
         ("state",          "state memory",            "#DD8452"),
         ("delay_storage",  "delay storage",           "#C44E52"),
         ("leakage_sb",     "leakage (state+buffer)",  "#E8B04B"),
         ("comm",           "communication",           "#55A868")]

E_AC = 0.9   # pJ


def load(path: pathlib.Path):
    d = json.loads(path.read_text())
    print(f"source: {path}")
    print(f"        {d['_meta'].get('n_test_samples', '?')} test samples, "
          f"{len(d['configs'])} configurations")

    # E[T][mesh][model] -> dict of parts
    # spikes[T][model], acc[T][model], sop[T][model]
    E, SPK, ACC, SOP = {}, {}, {}, {}
    for key, cfg in d["configs"].items():
        T = cfg.get("T")
        if T is None:
            T = int(key.split("_")[0].lstrip("T"))
        mesh = key.split("_")[1]
        ncores = cfg["ncores"]
        E.setdefault(T, {}).setdefault(ncores, {})
        SPK.setdefault(T, {})
        ACC.setdefault(T, {})
        SOP.setdefault(T, {})
        for m in MODELS:
            mm = cfg["models"][m]
            eu = mm["energy_uJ"]
            E[T][ncores][m] = {k: eu[k] for k, _, _ in PARTS}
            E[T][ncores][m]["total"] = eu["total"]
            E[T][ncores][m]["total_sram"] = eu["total_sram"]
            if m not in SPK[T]:
                SPK[T][m] = mm.get("spikes", 0) or 0
                ACC[T][m] = mm["acc_pct"]
                # SOP is stored as the synaptic energy; recover the count.
                SOP[T][m] = eu["synaptic"] / E_AC * 1e6
    return E, SPK, ACC, SOP


def panel(ax, E, SPK, T, ncores, title):
    xs = np.arange(len(MODELS))
    bottoms = np.zeros(len(MODELS))
    for key, lab, col in PARTS:
        vals = np.array([E[T][ncores][m][key] for m in MODELS])
        ax.bar(xs, vals, 0.62, bottom=bottoms, label=lab, color=col,
               edgecolor="white", linewidth=0.6)
        bottoms += vals
    top = float(bottoms.max())
    ax.set_xticks(xs)
    ax.set_xticklabels([SHOW[m] for m in MODELS], fontsize=7.5)
    ax.set_ylabel("energy / inference (µJ)")
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_ylim(0, top * 1.32)
    ax.grid(axis="y", alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)

    for x, m in zip(xs, MODELS):
        ax.annotate(f"{E[T][ncores][m]['total']:.2f} µJ", (x, bottoms[x]),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, fontweight="bold")
        ax.annotate(f"{SPK[T][m]/1000:.1f}k spikes", (x, top * 1.24),
                    ha="center", fontsize=7.5, color="#222222", fontweight="bold")
        ax.annotate("●", (x, top * 1.155), ha="center", fontsize=9, color="#222222")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()

    jp = pathlib.Path(a.json)
    if not jp.exists():
        raise SystemExit(f"results JSON not found: {jp}\n"
                         f"Run: python study1_measure.py --n -1 --save {jp}")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    E, SPK, ACC, SOP = load(jp)
    meshes = sorted({int(c) for c in E[100].keys()})
    if 4 not in meshes:
        raise SystemExit(f"expected a 4-core (2x2) configuration, got {meshes}")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})

    # ---------- figure 1: main (T=100, 2x2) ----------
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    panel(ax, E, SPK, 100, 4,
          "Study 1 — T = 100 (14 ms/frame), 2×2 mesh (4 cores)\n"
          "stacked energy per inference, with spike counts overlaid")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7.4,
              frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "study1_main_T100_2x2.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- figure 2: 3 resolutions x 2 meshes ----------
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.4), sharex=False)
    for j, T in enumerate((50, 100, 140)):
        panel(axes[0, j], E, SPK, T, 4,
              f"T = {T} ({1400/T:.0f} ms/frame), 2×2")
        panel(axes[1, j], E, SPK, T, 64,
              f"T = {T} ({1400/T:.0f} ms/frame), 8×8")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Study 1 — energy breakdown across time-step resolutions and mesh sizes",
                 fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out / "study1_grid_resolution_mesh.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- figure 3: ranking by three estimates ----------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    names = ["spike count", "SOP (× fanout)", "full model energy"]
    for ax, T in zip(axes, (50, 100, 140)):
        sp = sorted(MODELS, key=lambda m: SPK[T][m])
        so = sorted(MODELS, key=lambda m: SOP[T][m])
        en = sorted(MODELS, key=lambda m: E[T][4][m]["total"])
        for i, r in enumerate([sp, so, en]):
            for pos, m in enumerate(r):
                ax.scatter(pos, -i, s=520, color=COLOR[m], zorder=3,
                           edgecolor="white", linewidth=1.5)
                ax.annotate(SHORT[m], (pos, -i), ha="center", va="center",
                            fontsize=9, color="white", fontweight="bold", zorder=4)
        ax.set_yticks([0, -1, -2])
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["1st (cheapest)", "2nd", "3rd (most expensive)"], fontsize=8)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(-2.5, 0.6)
        ax.set_title(f"T = {T}, 2×2 mesh\n"
                     f"spike{'≠' if sp != en else '='}energy   "
                     f"SOP{'≠' if so != en else '='}energy", fontsize=9)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Simplified estimates mis-rank the three models", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "study1_ranking_flip.png", bbox_inches="tight")
    plt.close(fig)

    print("\nsaved:")
    for f in ("study1_main_T100_2x2.png", "study1_grid_resolution_mesh.png",
              "study1_ranking_flip.png"):
        p = out / f
        print(f"  {p}  ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
