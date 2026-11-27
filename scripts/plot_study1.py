"""Study 1 figures — three plots, produced once per delay-access variant.

Two variants, identical layout:

    recurrent  delay access charged on the recurrent edges only  (§4.2, main tables)
    allsop     delay access charged on every synaptic operation  (§8.7, upper bound)

Within a variant the placement is chosen with --combo; the default charges the
delay access and shows it (weights resident, delay table in SRAM), because that
is the only placement in which the energy ranking can differ from the SOP ranking.

Usage
    python plot_study1.py                       # both variants, default combo
    python plot_study1.py --variant recurrent
    python plot_study1.py --combo w_res_d_res
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_JSON = ROOT / "results" / "study1_six_combos.json"
DEFAULT_OUT = ROOT / "figs"

MODELS = ["delay", "lif", "mc"]
SHOW = {"delay": "Delay-trained\n(DelRec, 256, per-synapse)",
        "lif": "Plain LIF\n(PMSN d_state=1)",
        "mc": "Multi-compartment\n(PMSN d_state=8)"}
SHORT = {"delay": "DL", "lif": "LIF", "mc": "MC"}
SHORT_FULL = {"delay": "Delay\n(per-synapse)",
              "lif": "LIF\n(d_state=1)",
              "mc": "MC\n(d_state=8)"}
COLOR = {"delay": "#C44E52", "lif": "#4C72B0", "mc": "#DD8452"}

# stacked-bar components: JSON key -> (label, colour)
PARTS_SRAM = [("synaptic",     "synaptic ops (AC)",   "#4C72B0"),
              ("neuron",       "neuron update",       "#7FA8D6"),
              ("state",        "state memory",        "#DD8452"),
              ("delay_access", "delay access",        "#C44E52"),
              ("leak_sb_d_sram", "leakage (state+table)", "#E8B04B"),
              ("comm",         "communication",       "#55A868")]
PARTS_RES = [("synaptic",    "synaptic ops (AC)",   "#4C72B0"),
             ("neuron",      "neuron update",       "#7FA8D6"),
             ("state",       "state memory",        "#DD8452"),
             ("delay_access","delay access",        "#C44E52"),
             ("leak_sb_d_res", "leakage (state+window)", "#E8B04B"),
             ("comm",        "communication",       "#55A868")]

E_AC = 0.9   # pJ

VARIANTS = {
    "recurrent": ("recurrent_only",
                  "delay access on the recurrent edges only (§4.2)"),
    "allsop": ("all_synaptic",
               "delay access on every synaptic operation (§8.7 upper bound)"),
}


def load(path: pathlib.Path, variant: str, combo: str):
    d = json.loads(path.read_text(encoding="utf-8"))
    src = d["variants"][VARIANTS[variant][0]]
    print(f"source: {path.name} / {variant} / combo={combo}")
    print(f"        {len(src['configs'])} configurations")

    parts = PARTS_SRAM if "sram" in combo.split("_d_")[-1] else PARTS_RES
    E, SPK, ACC, SOP = {}, {}, {}, {}
    for key, cfg in src["configs"].items():
        T = cfg["T"]
        ncores = cfg["ncores"]
        E.setdefault(T, {}).setdefault(ncores, {})
        SPK.setdefault(T, {})
        ACC.setdefault(T, {})
        SOP.setdefault(T, {})
        for m in MODELS:
            mm = cfg["models"][m]
            c = mm["components_uJ"]
            E[T][ncores][m] = {k: c[k] for k, _, _ in parts}
            E[T][ncores][m]["total"] = cfg["combos"][combo]["totals_uJ"][m]
            if m not in SPK[T]:
                SPK[T][m] = mm.get("spikes", 0) or 0
                ACC[T][m] = mm["acc_pct"]
                SOP[T][m] = mm["sop"]
    return E, SPK, ACC, SOP, parts


def panel(ax, E, SPK, T, ncores, parts, title, compact=False, rotate=0):
    xs = np.arange(len(MODELS))
    bottoms = np.zeros(len(MODELS))
    for key, lab, col in parts:
        vals = np.array([E[T][ncores][m][key] for m in MODELS])
        ax.bar(xs, vals, 0.62, bottom=bottoms, label=lab, color=col,
               edgecolor="white", linewidth=0.6)
        bottoms += vals
    top = float(bottoms.max())
    ax.set_xticks(xs)
    labels = [SHORT_FULL[m] if compact else SHOW[m] for m in MODELS]
    ax.set_xticklabels(labels, fontsize=7.5, rotation=rotate,
                       ha="right" if rotate else "center")
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


def make(jp: pathlib.Path, out: pathlib.Path, variant: str, combo: str, suffix: str):
    E, SPK, ACC, SOP, parts = load(jp, variant, combo)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    vlabel = VARIANTS[variant][1]
    combo_note = {
        "w_res_d_res":   "weights resident, delay table resident",
        "w_res_d_sram":  "weights resident, delay table in SRAM",
        "w_sram_r1_d_res":  "weights in SRAM (R=1), delay table resident",
        "w_sram_r1_d_sram": "weights in SRAM (R=1), delay table in SRAM",
    }.get(combo, combo)

    # ---------- figure 1: main (three resolutions, all 2x2) ----------
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0), sharey=False)
    for j, T in enumerate((50, 100, 140)):
        panel(axes[j], E, SPK, T, 4, parts,
              f"T = {T}  ({1400/T:.0f} ms/frame)\n2×2 mesh (4 cores)")
    axes[0].set_ylabel("energy / inference (µJ)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, fontsize=7.6, frameon=False,
               bbox_to_anchor=(0.5, -0.055))
    fig.suptitle("Study 1 — stacked energy per inference, 2×2 mesh, all three time resolutions\n"
                 f"{combo_note}; {vlabel}", fontsize=11, y=1.0)

    caption = (
        "Assumptions.  (1) FP32, 32-bit weights and state; the white paper's low-precision "
        "regime is not modelled.\n"
        f"(2) Placement: {combo_note}. The other three placements are in §5.5.\n"
        "(3) 2×2 mesh (4 cores) is the main configuration; 4×4 and 8×8 appear only as "
        "sensitivity cases (§5.5, fig. 2).\n"
        "(4) Sequential placement, core(i) = floor(i·n_cores/n_total); no optimisation (§8.1).\n"
        "(5) Communication counted as a unicast upper bound, flit_hops × 3.5 pJ; spikes in the "
        "same timestep are not merged (§8.2, §8.3).\n"
        "(6) par = 1, i.e. one core's worth of parallelism, no literature basis (§8.4); "
        "P_leak is an 8 MB cache value, so small-SRAM leakage is underestimated.\n"
        f"(7) {vlabel}. The delay table is 65,536 × 6 bit and is charged to delay storage, "
        "not to weights (§5.3).\n"
        "(8) Checkpoints are the best-validation epoch, not the best-test epoch.\n"
        "(9) The LIF bar is PMSN d_state=1, not the TASK 4 delay-free network: we could not "
        "train that network to usable accuracy in the time budget (§6.1, §7.5)."
    )
    fig.text(0.5, -0.22, caption, ha="center", va="top", fontsize=6.5,
             linespacing=1.5, wrap=True,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#F7F7F7",
                       edgecolor="#CCCCCC", linewidth=0.7))
    fig.tight_layout()
    fig.savefig(out / f"study1_main_2x2_allT_{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- figure 2: 3 resolutions x 3 meshes ----------
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.6), sharex=False)
    for i, nc in enumerate((4, 16, 64)):
        mesh = {4: "2×2", 16: "4×4", 64: "8×8"}[nc]
        for j, T in enumerate((50, 100, 140)):
            panel(axes[i, j], E, SPK, T, nc, parts,
                  f"T = {T} ({1400/T:.0f} ms/frame), {mesh} ({nc} cores)", compact=True)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("Study 1 — energy breakdown across time-step resolutions and mesh sizes\n"
                 f"{combo_note}; {vlabel}", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out / f"study1_grid_resolution_mesh_{suffix}.png", bbox_inches="tight")
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
    fig.suptitle(f"Simplified estimates vs the full model — {combo_note}\n{vlabel}",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(out / f"study1_ranking_flip_{suffix}.png", bbox_inches="tight")
    plt.close(fig)

    print("  saved:")
    for f in (f"study1_main_2x2_allT_{suffix}.png",
              f"study1_grid_resolution_mesh_{suffix}.png",
              f"study1_ranking_flip_{suffix}.png"):
        p = out / f
        print(f"    {p.name}  ({p.stat().st_size/1024:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--variant", default="both", choices=["both", "recurrent", "allsop"])
    ap.add_argument("--combo", default="w_res_d_sram",
                    choices=["w_res_d_res", "w_res_d_sram",
                             "w_sram_r1_d_res", "w_sram_r1_d_sram"])
    a = ap.parse_args()

    jp = pathlib.Path(a.json)
    if not jp.exists():
        raise SystemExit(f"results JSON not found: {jp}")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    todo = ["recurrent", "allsop"] if a.variant == "both" else [a.variant]
    for v in todo:
        print(f"\n=== {v} ===")
        make(jp, out, v, a.combo, v)


if __name__ == "__main__":
    main()
