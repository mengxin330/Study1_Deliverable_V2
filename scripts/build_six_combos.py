"""Build the six hardware combinations (four placements x two weight-reuse factors).

Reads the two measurement JSONs produced by
    study1_measure.py         delay_access = recurrent edges only
    study1_measure_allsop.py  delay_access = every synaptic operation
and writes results/study1_six_combos.json.

Combinations
------------
    weights  delay table  weight reuse   access cost
    ---------------------------------------------------------------
    resident resident     -              none
    resident SRAM         -              delay_access
    SRAM     resident     R = 1          w_read
    SRAM     SRAM         R = 1          w_read + delay_access
    SRAM     resident     R = 4          w_read / 4
    SRAM     SRAM         R = 4          w_read / 4 + delay_access

R applies to the weight read only (advisor: "a column for weight reuse R = 4");
the delay-table access is not divided by R.  Reuse does not change temporal
state, so ts is taken from the measurement and only the denominator moves.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"

SOURCES = {
    "recurrent_only": RES / "study1_partA_full.json",
    "all_synaptic":   RES / "study1_allsop_full.json",
}

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
    out = {"source": src_path.name, "n_test_samples": d["_meta"]["n_test_samples"],
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

    for label, _, _ in COMBOS:
        hit_a = sum(v["combos"][label]["a"] for v in out["configs"].values())
        hit_aa = sum(v["combos"][label]["a_all"] for v in out["configs"].values())
        hit_b = sum(v["combos"][label]["b_sop"] for v in out["configs"].values())
        n = len(out["configs"])
        out["summary"][label] = {"a": f"{hit_a}/{n}", "a_all": f"{hit_aa}/{n}",
                                 "b_sop": f"{hit_b}/{n}"}
    return out


def main():
    out = {"_meta": {
        "generated_by": "scripts/build_six_combos.py",
        "note": ("R = 4 divides the weight read only; the delay-table access is "
                 "unchanged.  ts is copied from the measurement, so ts_share is "
                 "ts / the reuse-adjusted total."),
        "combos": [{"label": l, "weight_reuse": r} for l, _, r in COMBOS],
    }, "variants": {}}

    for variant, path in SOURCES.items():
        if not path.exists():
            print(f"  missing: {path.name}  -- skipped")
            continue
        out["variants"][variant] = build(path)
        print(f"  {variant}: {len(out['variants'][variant]['configs'])} configs")

    dst = RES / "study1_six_combos.json"
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {dst}")

    for variant, v in out["variants"].items():
        print(f"\n=== {variant} ===")
        print(f"  {'combination':<20}{'a':>6}{'a_all':>8}{'b_sop':>8}")
        for label, s in v["summary"].items():
            print(f"  {label:<20}{s['a']:>6}{s['a_all']:>8}{s['b_sop']:>8}")


if __name__ == "__main__":
    main()
