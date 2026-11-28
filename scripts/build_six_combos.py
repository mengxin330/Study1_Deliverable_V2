"""Build the six hardware combinations (four placements x two weight-reuse factors).

Reads a single measurement JSON and writes results/study1_six_combos.json.

Combinations
------------
    weights  delay table  weight reuse   access cost
    ---------------------------------------------------------------
    resident resident     -              window read
    resident SRAM         -              window read + table read
    SRAM     resident     R = 1          weight read + window read
    SRAM     SRAM         R = 1          weight read + table read + window read
    SRAM     resident     R = 4          weight read/4 + window read
    SRAM     SRAM         R = 4          weight read/4 + table read + window read

R applies to the weight read only (advisor: "a column for weight reuse R = 4");
the delay access is not divided by R.

Two reads versus one
--------------------
Reading the spike window is unavoidable: `spike_i[t - d_ij]` has to come out of
storage, one access per activated recurrent connection.  Reading the delay value
is an additional access and happens only when the table is in SRAM.  So the
delay-table-in-SRAM placements pay two accesses and the resident placements pay
one.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
SOURCE = RES / "study1_partA_full.json"

# (label, combo key in the measurement, weight reuse factor or None)
COMBOS = [
    ("w_res_d_res",       "w_res_d_res",   None),
    ("w_res_d_sram",      "w_res_d_sram",  None),
    ("w_sram_r1_d_res",   "w_sram_d_res",  1),
    ("w_sram_r1_d_sram",  "w_sram_d_sram", 1),
    ("w_sram_r4_d_res",   "w_sram_d_res",  4),
    ("w_sram_r4_d_sram",  "w_sram_d_sram", 4),
]

MODELS = ("delay", "lif", "mc")


def build(src_path: pathlib.Path) -> dict:
    d = json.loads(src_path.read_text(encoding="utf-8"))
    out = {"source": src_path.name,
           "n_test_samples": d["_meta"]["n_test_samples"],
           "configs": {}, "summary": {}}

    for T in (50, 100, 140):
        for mesh in ("2x2", "4x4", "8x8"):
            name = f"T{T}_{mesh}"
            cfg = d["configs"][name]
            rec = {"T": T, "mesh": mesh, "ncores": cfg["ncores"],
                   "t_inf_us": cfg["t_inf_us"], "sop_order": cfg["sop_order"],
                   "spike_order": cfg["spike_order"], "combos": {},
                   "models": cfg["models"]}

            for label, key, R in COMBOS:
                c = cfg["combos"][key]
                totals = dict(c["totals"])
                ts = dict(c["ts"])
                if R not in (None, 1):
                    w_read = cfg["models"]["delay"]["components_uJ"]["weight_read"]
                    for m in MODELS:
                        totals[m] = totals[m] - w_read + w_read / R

                order = sorted(MODELS, key=lambda m: totals[m])
                shares = {m: ts[m] / totals[m] for m in MODELS}
                rec["combos"][label] = {
                    "weight_reuse": R,
                    "totals_uJ": totals,
                    "ts_uJ": ts,
                    "ts_share": shares,
                    "energy_order": order,
                    "a": any(v >= 0.10 for v in shares.values()),
                    "a_all": all(v >= 0.10 for v in shares.values()),
                    "b_sop": cfg["sop_order"] != order,
                    "b_spike": cfg["spike_order"] != order,
                }
            out["configs"][name] = rec

    n = len(out["configs"])
    for label, _, _ in COMBOS:
        out["summary"][label] = {
            "a": f"{sum(v['combos'][label]['a'] for v in out['configs'].values())}/{n}",
            "a_all": f"{sum(v['combos'][label]['a_all'] for v in out['configs'].values())}/{n}",
            "b_sop": f"{sum(v['combos'][label]['b_sop'] for v in out['configs'].values())}/{n}",
        }
    return out


def main():
    if not SOURCE.exists():
        raise SystemExit(f"measurement not found: {SOURCE}\n"
                         f"Run: python study1_measure.py --n -1 --save {SOURCE}")
    out = {"_meta": {
        "generated_by": "scripts/build_six_combos.py",
        "source": SOURCE.name,
        "note": ("R = 4 divides the weight read only; the delay access is unchanged. "
                 "Window read is paid in every placement; the table read is paid only "
                 "when the delay table is in SRAM."),
        "combos": [{"label": l, "weight_reuse": r} for l, _, r in COMBOS],
    }, "variants": {}}

    out["variants"]["recurrent_only"] = build(SOURCE)
    print(f"  {SOURCE.name}: {len(out['variants']['recurrent_only']['configs'])} configs")

    dst = RES / "study1_six_combos.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {dst}")

    print(f"\n  {'combination':<20}{'a':>6}{'a_all':>8}{'b_sop':>8}")
    for label, s in out["variants"]["recurrent_only"]["summary"].items():
        print(f"  {label:<20}{s['a']:>6}{s['a_all']:>8}{s['b_sop']:>8}")


if __name__ == "__main__":
    main()
