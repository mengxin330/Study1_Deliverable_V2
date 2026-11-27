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

    3. delay storage         -> delay_access (per-synapse table lookup)

    4. state memory          -> state

    5. leakage               -> leak (state + spike window)



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

    "delay": dict(params=239124, neurons=512, d_state=1,

                  n_delay=256 * 256, n_buffer=256),

    "lif":   dict(params=187988, neurons=704, d_state=1, n_delay=0, width=352),

    "mc":    dict(params=200660, neurons=704, d_state=8, n_delay=0, width=352),

}



# Hidden width per model: the delay network uses per-synapse delays, which cost

# h^2 parameters, so it is 256 wide to stay parameter-matched with PMSN's 352.

MODEL_WIDTH = {"delay": 256, "lif": 352, "mc": 352}



# Fanout per layer, i.e. how many MACs one spike in that layer triggers.

#   delay  h1: 256 -> 256 (dense hidden1->hidden2)

#          h2: 20 (out) + 256 (recurrent) = 276

#   PMSN   h1: 352 -> 352

#          h2: 20 (fc), no recurrent term

FANOUT = {

    "delay": dict(input=256, hidden1=256, hidden2=276),

    "lif":   dict(input=352, hidden1=352, hidden2=20),

    "mc":    dict(input=352, hidden1=352, hidden2=20),

}



# test accuracy on the full 2,264-sample SHD test split (from each final_test.json)

ACC = {("delay", 50): 84.54, ("delay", 100): 84.30, ("delay", 140): 83.81,

       ("lif", 50): 88.96, ("lif", 100): 86.40, ("lif", 140): 87.10,

       ("mc", 50): 91.21, ("mc", 100): 88.30, ("mc", 140): 88.96}



# Ring-buffer depth L, from the canonical _build_mask in delay_layers_pytorch.py:

#     s = 1.0 + sigma ;  L = ceil(1 + max_d + s) + 1

# with sigma = 10.0 and max_d = 35.09 / 41.59 / 37.18 (from each final_delays.npz).

# This is the sliding-window length the kernel actually needs, and it is 11 slots

# deeper than ceil(max_d) + 2 because the triangular kernel has half-width 1+sigma.

SLOTS = {50: 49, 100: 55, 140: 51}





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

    """{mesh: (hop_matrix, {model: (placement, fanout_table)})}.



    The delay network is 256 wide, the two PMSN variants are 352 wide, so the

    sequential placement -- core(i) = floor(i * n_cores / n_total) -- maps them

    to different cores.  Building one table for all three (as an earlier version

    did with a hardcoded 352) silently mis-places the delay model.

    """

    tabs = {}

    for cn, nc, mn in MESHES:

        hop = make_hop(nc, mn)

        per_model = {}

        for model, hidden in MODEL_WIDTH.items():

            pl = Placement(layers_for(hidden=hidden, out=20), n_cores=nc)

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

            per_model[model] = (pl, tbl)

        tabs[cn] = (hop, per_model)

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

    sop, spk, h2 = 0, 0, 0

    fh = {c[0]: [] for c in MESHES}

    with torch.no_grad():

        for i in range(N_TEST):

            x = bin_one(times[i], units[i], ts_ms, T)

            # input -> hidden1 (140 pooled channels -> 256 neurons): this layer sits

            # before the first spike_registrator, so no hook above counts it.  Every

            # input event triggers 256 MACs, same spike-driven convention as below.

            sop += int(round(float(x.sum()))) * FANOUT["delay"]["input"]

            dm.model(torch.from_numpy(x).unsqueeze(1).float())

            rows = {}

            for mi, key in ((3, "hidden1"), (8, "hidden2")):

                s = mods[f"layers.{mi}"].spikes

                if s is None:

                    continue

                s = np.asarray(s.cpu())

                if s.ndim == 3:

                    s = s[:, 0, :]

                cnt = int((s > 0.5).sum())

                spk += cnt

                if key == "hidden2":

                    h2 += cnt          # only these consult the delay table

                sop += cnt * FANOUT["delay"][key]

                for t in range(T):

                    idx = np.nonzero(s[t] > 0.5)[0].tolist()

                    if idx:

                        rows.setdefault(t, {})[key] = idx

            for cn, _, _ in MESHES:

                _, per_model = tabs[cn]

                pl, tbl = per_model["delay"]

                fh[cn].append(hops_of(rows, pl, tbl))

    return sop / N_TEST, spk / N_TEST, h2 / N_TEST, {k: np.mean(v) for k, v in fh.items()}





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

    sop, spk, h2 = 0, 0, 0

    fh = {c[0]: [] for c in MESHES}

    with torch.no_grad():

        for i in range(N_TEST):

            captured.clear()

            x = bin_one(times[i], units[i], ts_ms, T)

            # input -> hidden1 (140 -> 352 for PMSN): same correction as measure_delay().

            sop += int(round(float(x.sum()))) * FANOUT[which]["input"]

            net(torch.from_numpy(x).unsqueeze(0).permute(0, 2, 1))

            rows = {}

            for k, v in captured.items():

                s = v.reshape(T, 1, -1)

                cnt = int((s > 0.5).sum())

                spk += cnt

                key = "hidden1" if "hidden1" in k else "hidden2"

                if key == "hidden2":

                    h2 += cnt

                sop += cnt * FANOUT[which][key]

                for t in range(T):

                    idx = torch.nonzero(s[t, 0, :] > 0.5, as_tuple=False).flatten().tolist()

                    if idx:

                        rows.setdefault(t, {})[key] = idx

            for cn, _, _ in MESHES:

                _, per_model = tabs[cn]

                pl, tbl = per_model[which]

                fh[cn].append(hops_of(rows, pl, tbl))

    for h in handles:

        h.remove()

    return sop / N_TEST, spk / N_TEST, h2 / N_TEST, {k: np.mean(v) for k, v in fh.items()}





def energy(model, T, sop, fh, mesh, ncores, slots, h2=0.0):

    """Energy under four hardware combinations of weight and delay-table placement.



    WHY FOUR, NOT TWO

    -----------------

    Two independent objects can each be either spatially expanded ("resident")

    or held in SRAM, so there are 2 x 2 = 4 combinations.  For the delay network

    all four are distinct; for PMSN's two variants there is no delay table, so

    they collapse to two.



    WHAT A DELAY ACCESS IS

    ----------------------

    Only the recurrent edges consult the delay table.  A spike from hidden2 fans

    out to 256 hidden2 targets, each with its own d_ij, so the lookup count is

    h2_spikes * width, NOT the full SOP.  The input layer (61 % of SOP in

    practice), the feedforward hidden1 -> hidden2 edges and the hidden2 -> out

    edges carry no delay and must not be charged for one.



    ACCESS COSTS ADD UP

    -------------------

    The weight and the delay table are separate objects, so their accesses are

    charged separately; they are NOT merged into one read.



        weights  delay table   per-inference access cost

        -------------------------------------------------------------------

        resident resident      none

        resident SRAM          delay_access                      (recurrent edges only)

        SRAM     resident      w_read                            (all edges)

        SRAM     SRAM          w_read + delay_access



    TEMPORAL STATE

    --------------

    ts never depends on the weight regime: per the white paper's category 3

    ("delay storage" = N_access * T * 2 * E_access) a delay-table access is part

    of the temporal state, but a weight access is not.  So ts depends only on

    where the delay table lives -- in SRAM both its lookup and its leakage count;

    resident, neither does.

    """

    a = ARCH[model]

    N, ds = a["neurons"], a["d_state"]

    total_mac = sop + N * T * ds

    t_inf = total_mac / (ncores * F)

    pj_per_bit = P_LEAK * t_inf * 1e12          # uJ leaked per stored bit



    DELAY_BITS = 6                              # 43 integer values, 0..42



    syn = sop * E_AC

    neu = N * T * ds * E_MAC

    state = N * T * ds * 2 * E_ACC

    w_read = sop * E_ACC                        # weights in SRAM: every edge reads one

    comm = fh * E_HOP



    n_buffer = a.get("n_buffer", a["n_delay"])

    delay_bits = a["n_delay"] * DELAY_BITS

    win_bits = n_buffer * max(slots - 1, 0)     # spike window, independent of the table

    w_bits = (a["params"] - a["n_delay"]) * 32



    # ---- recurrent-only delay lookup -------------------------------------

    # PMSN's variants are feedforward: no recurrent edge, no table, no lookup.

    has_table = a["n_delay"] > 0

    rec_ops = h2 * a.get("width", 256) if has_table else 0.0

    # VARIANT (this copy only): charge the delay access on EVERY synaptic
    # operation, i.e. assume every layer consults the delay table.  This is a
    # conservative upper bound; study1_measure.py charges only the recurrent
    # edges, which are the only ones that actually read a delay.
    delay_access = sop * E_ACC if has_table else 0.0



    # ---- leakage, both ways ----------------------------------------------

    sb_bits_d_sram = N * ds * 32 + delay_bits + win_bits   # table occupies runtime SRAM

    sb_bits_d_res = N * ds * 32 + win_bits                 # table spatially expanded

    leak_sb_d_sram = sb_bits_d_sram * pj_per_bit

    leak_sb_d_res = sb_bits_d_res * pj_per_bit

    leak_w = w_bits * pj_per_bit                           # weights only, always reported apart



    base = syn + neu + state + comm



    # ---- the four combinations -------------------------------------------

    # weights resident, delay table in SRAM

    tot_w_res_d_sram = base + delay_access + leak_sb_d_sram

    # weights resident, delay table resident

    tot_w_res_d_res = base + leak_sb_d_res

    # weights in SRAM, delay table resident

    tot_w_sram_d_res = base + w_read + leak_sb_d_res

    # weights in SRAM, delay table in SRAM -- charged separately, not merged

    tot_w_sram_d_sram = base + w_read + delay_access + leak_sb_d_sram



    # ---- temporal state: depends on the delay table only -----------------

    ts_d_sram = state + delay_access + leak_sb_d_sram

    ts_d_res = state + leak_sb_d_res



    return dict(

        t_inf=t_inf * 1e6, syn=syn, neu=neu, state=state, comm=comm,

        delay_access=delay_access, rec_ops=rec_ops,

        w_read=w_read, leak_w=leak_w,

        leak_sb_d_sram=leak_sb_d_sram, leak_sb_d_res=leak_sb_d_res,

        # four combinations

        tot_w_res_d_sram=tot_w_res_d_sram,

        tot_w_res_d_res=tot_w_res_d_res,

        tot_w_sram_d_res=tot_w_sram_d_res,

        tot_w_sram_d_sram=tot_w_sram_d_sram,

        # temporal state and its share, per delay-table placement

        ts_d_sram=ts_d_sram, ts_d_res=ts_d_res,

        ts_share_d_sram=ts_d_sram / tot_w_res_d_sram,

        ts_share_d_res=ts_d_res / tot_w_res_d_res,

        ts_share_d_sram_w_sram=ts_d_sram / tot_w_sram_d_sram,

        ts_share_d_res_w_sram=ts_d_res / tot_w_sram_d_res,

        # bits

        delay_bits=delay_bits, win_bits=win_bits, w_bits=w_bits,

        sb_bits_d_sram=sb_bits_d_sram, sb_bits_d_res=sb_bits_d_res,

        h2_spikes=h2,

    )





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

        s, sp, h2, f = measure_delay(T, times, units, tabs)

        RAW[("delay", T)] = (s, sp, h2, f)

        print(f"{'delay':<7}{T:>5}{sp:>11,.0f}{s:>12,}", end="")

        for cn, _, _ in MESHES:

            print(f"{f[cn]:>13,.0f}", end="")

        print()

    for w in ("lif", "mc"):

        for T in (50, 100, 140):

            s, sp, h2, f = measure_pmsn(w, T, times, units, tabs)

            RAW[(w, T)] = (s, sp, h2, f)

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

                sop, spk, h2, fhm = RAW[(m, T)]

                E[m] = energy(m, T, sop, fhm[cn], cn, nc, SLOTS[T], h2)

            # spike / SOP orderings

            s_spk = sorted(E, key=lambda m: RAW[(m, T)][1])

            s_sop = sorted(E, key=lambda m: RAW[(m, T)][0])

            # four hardware combinations of weight / delay-table placement

            s_wr_dr = sorted(E, key=lambda m: E[m]["tot_w_res_d_res"])

            s_wr_ds = sorted(E, key=lambda m: E[m]["tot_w_res_d_sram"])

            s_ws_dr = sorted(E, key=lambda m: E[m]["tot_w_sram_d_res"])

            s_ws_ds = sorted(E, key=lambda m: E[m]["tot_w_sram_d_sram"])



            print(f"\n--- {cn} ({nc} cores)   t_inf(delay)={E['delay']['t_inf']:.1f} us ---")

            hdr = (f"{'model':<7}{'acc':>7}{'spike':>10}{'syn':>7}{'neu':>7}"

                   f"{'state':>8}{'dacc':>8}{'leak':>8}{'comm':>7}{'w_read':>9}{'wleak':>9}")

            print(hdr)

            print("-" * len(hdr))

            for m in ("delay", "lif", "mc"):

                e = E[m]

                print(f"{m:<7}{ACC[(m,T)]:>6.2f}%{RAW[(m,T)][1]:>10,.0f}"

                      f"{e['syn']/1e6:>7.3f}{e['neu']/1e6:>7.3f}"

                      f"{e['state']/1e6:>8.3f}{e['delay_access']/1e6:>8.3f}"

                      f"{e['leak_sb_d_sram']/1e6:>8.4f}{e['comm']/1e6:>7.3f}"

                      f"{e['w_read']/1e6:>9.3f}{e['leak_w']/1e6:>9.3f}")



            print(f"  {'combination':<26}" + "".join(f"{m:>9}" for m in ("delay","lif","mc")) + f"{'ts%(delay)':>12}")

            for tag, key, tskey in (

                    ("W resident, D resident", "tot_w_res_d_res",  "ts_d_res"),

                    ("W resident, D SRAM",     "tot_w_res_d_sram", "ts_d_sram"),

                    ("W SRAM,     D resident", "tot_w_sram_d_res", "ts_d_res"),

                    ("W SRAM,     D SRAM",     "tot_w_sram_d_sram","ts_d_sram")):

                vals = [E[m][key] for m in ("delay","lif","mc")]

                tsv = E["delay"][tskey]

                print(f"  {tag:<26}" + "".join(f"{v/1e6:>9.3f}" for v in vals)

                      + f"{100*tsv/vals[0]:>11.1f}%")



            print(f"  spike order : {' < '.join(s_spk)}")

            print(f"  SOP   order : {' < '.join(s_sop)}")

            for tag, s in (("Wres Dres", s_wr_dr), ("Wres Dsram", s_wr_ds),

                           ("Wsram Dres", s_ws_dr), ("Wsram Dsram", s_ws_ds)):

                flip = "FLIP" if s_sop != s else "same"

                print(f"  energy {tag:<11}: {' < '.join(s):<22}  vs SOP -> {flip}")



            combos = {

                "w_res_d_res":   ("tot_w_res_d_res",   "ts_d_res",  s_wr_dr),

                "w_res_d_sram":  ("tot_w_res_d_sram",  "ts_d_sram", s_wr_ds),

                "w_sram_d_res":  ("tot_w_sram_d_res",  "ts_d_res",  s_ws_dr),

                "w_sram_d_sram": ("tot_w_sram_d_sram", "ts_d_sram", s_ws_ds),

            }

            cfg = {"ncores": nc, "t_inf_us": E["delay"]["t_inf"],

                   "sop_order": s_sop, "spike_order": s_spk, "combos": {}}

            for cname, (totkey, tskey, order) in combos.items():

                shares = {m: E[m][tskey] / E[m][totkey] for m in ("delay","lif","mc")}

                cfg["combos"][cname] = {

                    "energy_order": order,

                    "b_sop": s_sop != order,

                    "b_spike": s_spk != order,

                    "a": any(v >= 0.10 for v in shares.values()),

                    "a_all": all(v >= 0.10 for v in shares.values()),

                    "ts_share": shares,

                    "totals": {m: E[m][totkey] / 1e6 for m in ("delay","lif","mc")},

                    "ts": {m: E[m][tskey] / 1e6 for m in ("delay","lif","mc")},

                }



            cfg["models"] = {}

            for m in ("delay","lif","mc"):

                e = E[m]

                cfg["models"][m] = {

                    "acc_pct": ACC[(m,T)], "spikes": RAW[(m,T)][1],

                    "sop": RAW[(m,T)][0], "h2_spikes": e["h2_spikes"],

                    "rec_ops": e["rec_ops"], "flit_hops": RAW[(m,T)][3][cn],

                    "components_uJ": {

                        "synaptic": e["syn"]/1e6, "neuron": e["neu"]/1e6,

                        "state": e["state"]/1e6, "delay_access": e["delay_access"]/1e6,

                        "comm": e["comm"]/1e6, "weight_read": e["w_read"]/1e6,

                        "weights_leak": e["leak_w"]/1e6,

                        "leak_sb_d_sram": e["leak_sb_d_sram"]/1e6,

                        "leak_sb_d_res": e["leak_sb_d_res"]/1e6,

                    },

                    "bits": {

                        "delay_table": e["delay_bits"], "spike_window": e["win_bits"],

                        "weights": e["w_bits"],

                        "timestate_d_sram": e["sb_bits_d_sram"],

                        "timestate_d_res": e["sb_bits_d_res"],

                    },

                }

            OUT[f"T{T}_{cn}"] = cfg

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
                "combos": {
                    c: {
                        "a": f"{sum(v['combos'][c]['a'] for v in OUT.values())}/{len(OUT)}",
                        "a_all": f"{sum(v['combos'][c]['a_all'] for v in OUT.values())}/{len(OUT)}",
                        "b_sop": f"{sum(v['combos'][c]['b_sop'] for v in OUT.values())}/{len(OUT)}",
                        "b_spike": f"{sum(v['combos'][c]['b_spike'] for v in OUT.values())}/{len(OUT)}",
                    }
                    for c in ("w_res_d_res", "w_res_d_sram", "w_sram_d_res", "w_sram_d_sram")
                },
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

