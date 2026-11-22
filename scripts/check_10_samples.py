"""Consistency check: 10-sample subset vs the full 2,264-sample average.

This script no longer re-implements the measurement loop. It imports
``study1_measure`` and calls the same ``measure_delay`` / ``measure_pmsn``
functions that produced the reported numbers, then compares the 10-sample
result against the stored full-set JSON. That way the reference values can
never drift away from the model version again.

Usage:
    python check_10_samples.py [--json <path>] [--n 10]

Default reference: ../results/study1_h256_measure_full.json
"""
import argparse
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import study1_measure as M   # noqa: E402

DEFAULT_JSON = HERE.parent / "results" / "study1_h256_measure_full.json"
E_AC = M.E_AC


def full_set_reference(path: pathlib.Path) -> dict:
    """Pull per-(model, T, mesh) SOP and flit_hops out of the saved JSON."""
    d = json.loads(path.read_text())
    ref = {}
    for key, cfg in d["configs"].items():
        # keys look like "T100_2x2"; the T is not always a stored field.
        T = cfg.get("T")
        if T is None:
            T = int(key.split("_")[0].lstrip("T"))
        for m in ("delay", "lif", "mc"):
            mm = cfg["models"][m]
            # SOP was stored as the synaptic energy; recover the count.
            sop = mm["energy_uJ"]["synaptic"] / E_AC * 1e6
            ref[(m, T, cfg["ncores"])] = {
                "sop": sop,
                "fh": mm.get("flit_hops"),
                "spikes": mm.get("spikes"),
            }
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()

    jp = pathlib.Path(a.json)
    if not jp.exists():
        sys.exit(f"reference JSON not found: {jp}\nRun study1_measure.py --save first.")
    ref = full_set_reference(jp)
    print(f"reference: {jp.name}  ({len(ref)} (model,T,ncores) cells)")
    print(f"subset   : first {a.n} of the test split\n")

    times, units, labels = M.load_test()
    M.N_TEST = a.n
    tabs = M.build_tabs()

    print(f"{'model':<7}{'T':>5}{'mesh':>6}{'SOP(10)':>13}{'SOP(full)':>13}"
          f"{'ratio':>8}{'fh(10)':>11}{'fh(full)':>11}{'ratio':>8}")
    print("-" * 92)

    rows = []
    for T in (50, 100, 140):
        for m, fn in (("delay", M.measure_delay), ("lif", M.measure_pmsn),
                      ("mc", M.measure_pmsn)):
            if m == "delay":
                sop, spk, fhm = fn(T, times, units, tabs)
            else:
                sop, spk, fhm = fn(m, T, times, units, tabs)
            for cn, nc, _ in M.MESHES:
                r = ref.get((m, T, nc))
                if r is None:
                    continue
                rs, rf = r["sop"], r["fh"]
                qs = sop / rs if rs else float("nan")
                qf = (fhm[cn] / rf) if rf else float("nan")
                rows.append((m, T, cn, qs, qf))
                print(f"{m:<7}{T:>5}{cn:>6}{sop:>13,.0f}{rs:>13,.0f}{qs:>8.3f}"
                      f"{fhm[cn]:>11,.0f}{rf if rf else 0:>11,.0f}{qf:>8.3f}")

    print()
    bad = [r for r in rows if not (0.90 <= r[3] <= 1.10)]
    badf = [r for r in rows if not (0.85 <= r[4] <= 1.15)]
    print(f"SOP   ratios in [0.90, 1.10]: {len(rows)-len(bad)}/{len(rows)}")
    print(f"hops  ratios in [0.85, 1.15]: {len(rows)-len(badf)}/{len(rows)}")
    if bad:
        print("\nSOP outliers:")
        for m, T, cn, qs, _ in bad:
            print(f"  {m} T={T} {cn}: {qs:.3f}")
    print("\nDone. A 10-sample subset is expected to sit within roughly ±10% of the")
    print("full-set mean; larger deviations mean the subset is unrepresentative.")


if __name__ == "__main__":
    main()
