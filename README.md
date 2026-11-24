# Study 1 — How large is the energy share of temporal state in brain-inspired SNNs?

**[English](README.md) | [中文](README.zh.md)**

> Preliminary Study 1 of the NSF proposal *Time as a First-Class Resource*
> Data collected: 2026-11-16
> Data basis: **all 2,264 samples of the SHD test split**

---

## 0. What this report answers

The goal of Study 1 in the white paper:

> "in delay-trained and multi-compartment models, **delay storage and neuron-state traffic, not spikes**, account for a substantial share of energy at matched accuracy. Spike-based estimates therefore misjudge these models."

Split into two falsifiable criteria:

| Criterion | Definition | White-paper threshold |
|---|---|---|
| **(a)** | Temporal state (state memory + delay storage + leakage) as a share of per-inference energy | ≥ 10% |
| **(b)** | Ranking by spike count or by SOP ≠ ranking by the full model | at least one comparison |

White-paper wording:

> "Temporal state is a non-trivial share of energy in **at least one** brain-inspired model, and the SOP-only ranking differs from the full-model ranking in **at least one comparison**."

---

## 1. The three models

| Model | Source | Params | Neurons | Epochs | Checkpoint |
|---|---|---|---|---|---|
| **delay 256** | DelRec (`alexmaxad/DelRec`), learned per-synapse delays | **239,124** | **532** (incl. output LIF) | **10** | best validation |
| **plain LIF** | PMSN (`xychen-comp/...-PMSN`), $d_{state}=1$ | 187,988 | 704 | 5 | last epoch |
| **multi-compartment** | PMSN, $d_{state}=8$ (4 complex) | 200,660 | 704 | 3 | last epoch |

**Why these three**: they match the white paper's "three small SNNs on SHD" — a plain LIF baseline, a delay-trained network, and a multi-compartment network. PMSN with $d_{state}=1$ is a first-order IIR, i.e. equivalent to LIF, so it shares one codebase with the $d_{state}=8$ variant; the only architectural difference is the number of compartments.

**Why the delay network is 256 wide**: a per-synapse delay table costs $h^2$ parameters, so
the hidden width is not a free choice. At $h=352$ — the width both PMSN variants use — the
delay network would carry **430,164** parameters against PMSN's 188–201 K, i.e. **more than
twice as many**, and the three models would no longer be comparable. At $h=256$ it is
**239,124** — the same order as the two PMSN variants — while still reaching 83.8–84.5 %
test accuracy. §10 records the accuracy cost of that choice.

**Why PMSN uses the last epoch rather than best**: MC peaks at epoch 1 and drops ~1 point by epoch 2. Taking best would raise it by 1.5–2 points and widen the gap to LIF to 3–4 points, breaking matched accuracy.

### Is the delay per neuron or per synapse?

**Per synapse.** `layers.7.recurrent_delays` is `(256, 256)` — one value per **connection**,
so a spike from neuron `j` reaches target `i` at `t + 1 + d_ij`
(`delay_layers.py`: `synaptic_recdel`). The same file's `axonal_recdel` (one delay per
presynaptic neuron, shape `(N,)`) and `common_recdel` (one scalar per layer) are the other
two variants; **we use `synaptic_recdel`**.

**Why it matters for the energy count**: a per-synapse table stores $h^2 = 65{,}536$ delay
values instead of $h = 256$ — **2.1 Mbit of delay storage** (§5.3). That is the single largest
difference between this accounting and a per-neuron one, and it is why the delay model's
temporal-state share is 79 % here rather than the 44 % a per-neuron design gives.

### Do the feedforward layers carry delays?

**No — the only delays are the second hidden layer's recurrent ones.** The checkpoint holds 18
tensors: three `Linear`, two `BatchNorm`, and `layers.7.recurrent_{weights,delays}`. There is
no `dcls` / `.P` entry, so no feedforward-delay layer was ever built; the first hidden layer is
delay-free too (`no_delay_in_first_layer=True`).

**This is the paper's SHD setup, not an omission**: DelRec reports SHD with *"vanilla LIF
neurons with stateless (instantaneous) synapses"*.

The `dcls`-only config keys (`max_feedforward_delay`, `no_delay_in_last_layer`, `DCLSversion`,
`kernel_count`, paddings, position schedulers) therefore have no effect on this network.

### Three time-step resolutions

Same 1.4 s span, different frame lengths:

| T | Frame | delay | LIF | MC | Spread |
|---|---|---|---|---|---|
| 50 | 28.0 ms | 84.54% | 88.96% | 91.21% | 6.67 pt |
| **100** | **14.0 ms** | **84.30%** | 86.40% | 88.30% | **4.00 pt** |
| **140** | **10.0 ms** | **83.81%** | 87.10% | 88.96% | **5.15 pt** |

**The delay model sits 4.0–6.7 points below the two PMSN variants.** That gap is the price of
the per-synapse parameter budget: with per-neuron delays at $h=352$ the same code reaches
86.56 / 87.24 / 87.99 at 306,612 parameters. All three models stay within one accuracy band
(84–91 %), which is what the white paper's "matched accuracy" asks for, but the delay network
is consistently the weakest of the three.

---

## 2. What we changed and how we trained

### 2.0 Unifying the time axis

**All three models, all three resolutions, 9 training runs in total, use one fixed-duration binning:**

$$\text{duration} = \frac{1400\ \text{ms}}{T}\ \text{ms/frame}, \quad T \in \{50, 100, 140\}$$

**So frame $t$ always corresponds to the same physical interval $[t \cdot \text{duration},\ (t+1) \cdot \text{duration})$ across all three models.**

**This is the precondition for comparison**: the two upstream codebases bin differently by default (DelRec by frame count, PMSN by `linspace` equal division). Without unification, "the same T" would mean different frame lengths, and SOP, delay values, and state-access counts would all lose comparability. Concrete changes in §2.1 and §2.2.

### 2.1 DelRec: three patches

| # | Location | Change | What breaks without it |
|---|---|---|---|
| 1 | `configs/perf_SHD.py` | Re-derive all delay hyperparameters from `time_step` | Delay semantics drift when changing resolution |
| 2 | `src/delrec/datasets/loaders.py` | Replace `pad_sequence_collate` with `fixed_collate_T{T}`; bin with `duration=` | **Equal-division binning (`frames_number`) drops accuracy from 87% to 70.40%** |
| 3 | `src/delrec/utils.py` | Temporal pooling `max` → `mean` | Mismatch with the official checkpoint |

**Why binning must use fixed duration**: DelRec's delays are discrete frame counts shared across samples. With equal-division binning, one frame covers different physical durations for differently-lengthed samples, destroying the delay semantics. Verified by element-wise comparison: fixed-duration binning is identical to `integrate_events_by_fixed_duration_shd(duration=...)`.

### 2.1b Reworking the delay network's training pipeline (upstream → ours)

**Upstream behaviour:**

| Stage | Original DelRec |
|---|---|
| Binning | `config.time_step` is passed to `integrate_events_by_fixed_duration_shd(duration=...)`, but downstream `collate_fn=pad_sequence_collate` keeps each sample's actual frame count |
| Batch assembly | Variable-length padding, **samples within a batch have different frame counts** |
| Temporal pooling | `m, _ = torch.max(output, 0)` (max over time) |
| Delay hyperparameters | Hard-coded at `@10 ms`: `max_rec_delay=30`, `max_feedforward_delay=250//time_step`, `sigma_init=10.0`, `init_recdel_offset=10`, `delay_std_init=15` |

**What we changed:**

| Stage | After | Why |
|---|---|---|
| Binning | **Fixed `duration = 1400/T` ms**, crop over-long, zero-pad short | Preserve "one frame = fixed physical duration" |
| Batch assembly | **`fixed_collate_T{T}`: crop / zero-pad to a constant T frames** | Delays are in frames; if frame counts differ within a batch, the same delay value means different physical durations |
| Temporal pooling | **`m = torch.mean(output, 0)`** | Matches the official checkpoint's convention |
| Delay hyperparameters | **All re-derived from physical time** (divided by `time_step`), see the table below | Delay semantics stay at 300 ms / 250 ms / 100 ms / 150 ms across resolutions |

**Note that `max_feedforward_delay` was written as `250//time_step` — it already did the physical-time conversion, but the other four were hard-coded `@10 ms`. Changing `time_step` alone without re-deriving these five would make the effective delays more than double at T=50.**

**Delay hyperparameters are fixed in frames, not derived from physical time:**

| Parameter | T=50 | T=100 | T=140 |
|---|---|---|---|
| `time_step` | 28 | 14 | 10 |
| `max_feedforward_delay` | 9 | 17 | 25 |
| `max_rec_delay` | 30 | 30 | 30 |
| `sigma_init` | 10.0 | 10.0 | 10.0 |
| `init_recdel_offset` | 10 | 10 | 10 |
| `delay_std_init` | 15 | 15 | 15 |
| `batch_size` | 40 | 40 | 40 |

**Upstream derives these from a fixed physical budget** (`300 ms / time_step` →
11 / 21 / 30, `100 ms / time_step` → 3.5 / 7.0 / 10.0, and so on). **This study
fixes the frame counts instead**, so the delay range is a constant number of frames
across resolutions and the buffer depth stays comparable.

The one exception is `max_feedforward_delay`, which does follow the physical-time
rule — and has **no effect here**, because the networks contain no `dcls` layer.

These values live in `training_bundle/delrec/config_T{50,100,140}.py`, one file per
resolution.

### 2.2 PMSN: two changes

| # | Change | What breaks without it |
|---|---|---|
| 1 | `np.linspace` equal-division binning → fixed `duration = 1400/T` ms | Its "one frame" would not match DelRec's, so the two families could not be compared at the same T |
| 2 | `bn=False` → `bn=True` | Checkpoints were trained with `bn=True`; a mismatch raises in `load_state_dict` |

**PMSN is a continuous-time-constant SSM and is insensitive to the binning scheme** (verified: accuracy unchanged after switching). The binning change exists purely to align the time axis with the delay network.

### 2.3 Training commands

Full guide: `patches/TRAINING.md`. The nine runs are driven by `training_bundle/`, which
ships every modified file so **no patching step is needed**:

```bash
unzip -q training_bundle.zip -d /content
bash /content/bundle/runs/02_delay_T100.sh      # delay
bash /content/bundle/runs/06_pmsn_lif_T100.sh   # PMSN LIF
```

Each script copies its files into place, then trains:

```bash
# delay — both kernel flags are required (§3.3 of TRAINING.md)
cp bundle/delrec/config_T100.py   DelRec/configs/perf_SHD.py
cp bundle/delrec/loaders_T100.py  DelRec/src/delrec/datasets/loaders.py
cp bundle/delrec/utils.py         DelRec/src/delrec/utils.py
cp bundle/delrec/networks_shd.py  DelRec/src/delrec/networks_shd.py
python experiments/train.py --dataset shd --model SNN_synaptic_recurrent_delays \
  --seeds 1111 --epochs 10 --kernel v2 --synaptic-kernel v2 \
  --out /content/work/exp/synaptic_h256_T100

# PMSN
cp bundle/pmsn/{SHD_PMSN,PMSN,lib}.py /content/PMSN/SHD/
python SHD_PMSN.py --data_path /content/shd --T 100 --d_state 1 --epochs 5 \
  --ckpt_path /content/work/exp/pmsn_lif_T100_fill_5ep --device cuda:0
```

### 2.4 No dropout at inference

All three models used dropout during training (delay: feedforward 0.4 / recurrent 0.2; PMSN: 0.2), **but every measurement runs under `eval()`, where dropout is the identity and costs nothing**. All reported SOP, spike counts, and flit hops come from dropout-free forward passes.

---

## 3. Mesh design and references

### 3.1 Topology

| Config | Cores | Routing | Hop range |
|---|---|---|---|
| 2×2 | 4 | 2D mesh, XY dimension-order | 1–2 |
| 4×4 | 16 | same | 1–6 |
| 8×8 | 64 | same | 1–14 |

**Hop count = Manhattan distance** (shortest-path table in `net/topology.py`).

**No 2-core configuration**: a 1D direct link has a degenerate hop distribution (all 1), which is not comparable with the heavy-tailed 2D mesh under a single topology family, so it was removed.

### 3.2 Neuron-to-core mapping

**Sequential mapping** (the default of Loihi's Nx SDK and the SpiNNaker compiler):

$$\text{core}(i) = \left\lfloor \frac{i \cdot n_{cores}}{n_{total}} \right\rfloor,
\quad n_{total} = 532 \text{ (delay)} \ / \ 704 \text{ (PMSN)}$$

**Each model is placed using its own neuron count**, so the delay network (532) and the two
PMSN variants (704) get slightly different core assignments.

**This is not a communication-optimizing algorithm; it is a deterministic, reproducible, connectivity-agnostic baseline.**

**Known cost**: the output layer has only 20 neurons, all landing on the last core.

| Mesh | hidden1 cores | hidden2 cores | **out cores** |
|---|---|---|---|
| 2×2 | 0–1 | 1–3 | **3** |
| 4×4 | 0–7 | 7–15 | **15** |
| 8×8 | 0–30 | 30–61 | **61–63** |

**So hidden2 → out traffic converges on 1–2 cores, creating a structural hotspot.** This is part of the "unoptimized upper bound", see §8.

### 3.3 How communication is counted

**A spike is sent to "every core that holds downstream neurons" (multicast semantics); hop count = the sum of shortest-path lengths to each target core.**

Since hidden1 and hidden2 are fully connected, each hidden1 spike genuinely must reach every core holding hidden2 neurons, so the multicast semantics is correct.

**However, summing per-target-core path lengths is equivalent to independent unicast per target, which is an upper bound on a real multicast tree** (see §8).

### 3.4 Per-core capacity

| Item | Value at 8×8 | Loihi limit |
|---|---|---|
| Neurons/core | 11 (hidden1/hidden2), 20 (out) | 1,024 |
| Synapses/core | ~3,800 (hidden2), 7,040 (out) | 128 K |

**Well below the limits; capacity is not a constraint.**

---

## 4. Calculation method

### 4.0 Why we built an analytical model instead of using an existing simulator

**We surveyed the available open SNN energy-evaluation tools and found none that fits this study.** The requirement is narrow but specific: a 45 nm **digital CMOS** cost model that covers Study 1's five categories (including **delay storage** and **leakage of stored state**), handles **learned delays** and **multi-compartment** neurons, and can be driven by the PyTorch models from two upstream codebases.

| Tool | What it models | Why it does not fit |
|---|---|---|
| **SpikeSim** (Yale) | Analog IMC crossbar (RRAM/SRAM), 65 nm | Architecture mismatch: crossbar tiles and ADC/DIFF pipelines, not digital MAC arrays. Its latency model ($\alpha$, `passes`) has no counterpart here |
| **NeuroSim** | Analog/IMC crossbar | Same mismatch as SpikeSim |
| **BookSim** | NoC latency and throughput | Models the interconnect only; no SNN energy, no state or delay storage |
| **snnTorch / SpikingJelly** | Training frameworks | Report MAC/SOP counts at best; no hardware energy model |
| **Lava / Nx SDK** | Targets Loihi 1 | Loihi 1's neuron model is CUBA LIF with **exactly two internal state variables** — the synaptic response current $u_i(t)$ and the membrane potential $v_i(t)$ (Davies et al. 2018, p.3). Our multi-compartment model needs **4 compartments × 2 real state variables = 8**, which cannot be mapped onto Loihi 1 at all. See the note below |
| **NeuroBench** | Hardware-independent metrics | Explicitly provides SOP and activation sparsity, **not** hardware energy |
| **SNNToolBox / N2D2** | Quantization and compilation | Energy modules are coarse (SOP-only) |

**Conclusion**: no existing tool jointly provides (i) a 45 nm digital-CMOS cost basis, (ii) per-category accounting of delay storage and state memory, (iii) support for learned-delay and multi-compartment models, and (iv) a path from these two training codebases. **Hence a transparent analytical model, where every term is a counted quantity times a literature unit price** — which is also what the white paper's Study 1 Setup asks for ("a transparent CMOS energy model with characterized per-operation, per-access, and per-bit-of-storage costs").

#### Loihi 1's structural limits (why its energy numbers cannot be reused)

**Beyond the process node, three hard architectural constraints make Loihi 1 unusable as the platform for this study:**

| Constraint | Loihi 1 | This study needs |
|---|---|---|
| **State variables per neuron** | **2** — $u_i$ (synaptic current) and $v_i$ (membrane potential), fixed CUBA LIF (Davies et al. 2018, p.3). Dendritic compartments exist, but each carries the same fixed $u/v$ dynamics; there is no way to instantiate an arbitrary $N$-th-order state-space model | **8** for the 4-compartment PMSN (4 compartments × real + imaginary parts). It simply does not fit |
| **Synaptic delay** | A per-synapse field in the 5-tuple $(i, j, \text{wgt}, \text{dly}, \text{tag})$ — one delay value attached to each synapse | A **per-neuron** learned delay with a triangular kernel of half-width $\sigma$ (DelRec). Different semantics; Loihi cannot represent the kernel spread |
| **Timing model** | Asynchronous, no global clock ($t_{NoC}$ is not a design parameter); each core iterates over its neuron compartments independently | Clock-driven, per-time-step updates, with an explicit $t_{inf} = \text{MAC count} / (n_{cores} F)$ |

**Consequences for reuse of Loihi measurements:**

- **23.6 pJ per synaptic spike op** is an end-to-end measured figure. It bundles weight RMW, state update, neuron update, and routing, so it **cannot be decomposed** into Study 1's five categories (synaptic / communication / delay / state / leakage). Using it would collapse the very distinction Study 1 set out to measure.
- **81 / 52 pJ per neuron update** likewise includes the full RMW loop over the neuron's state banks, with no way to isolate the delay-buffer and multi-compartment terms.
- **3.0 / 4.0 pJ per tile hop** *is* usable — it is the one quantity whose scope (router + link, one flit) matches our `flit_hops` definition, and we use it as $E_{hop}$ (§4.4), cross-checked against Gorski's 45 nm router power.

**So the honest position is**: Loihi 1 supplies one transferable unit price ($E_{hop}$) and a qualitative confirmation that communication dominates at large mesh, but it cannot host either the delay-trained or the multi-compartment model, and its compound energy figures cannot be decomposed into the accounting this study requires.

**This is a real limitation, not just a convenience**: a cycle-accurate simulator would additionally capture buffer contention, arbitration, and congestion, all of which are currently outside the model (§8.2, §8.3). It also means the *absolute* numbers are not directly comparable with published tool-based results.

### 4.1 Time model

$$t_{inf} = \frac{\text{SOP} + N_{neur} \cdot T \cdot d_{state}}{n_{cores} \cdot F \cdot \text{par}}, \quad F = 2\ \text{GHz},\ \text{par} = 1$$

**Physical meaning**: total MAC count divided by (cores × frequency × per-cycle parallelism).

**`par = 1` means each core completes one MAC per cycle — an optimistic assumption**, see §8 limitation 1.

### 4.2 The five Study 1 categories

Per the white paper's Study 1 Procedure (p.12):

| # | Category | Formula |
|---|---|---|
| 1 | **synaptic operations** | $\text{SOP} \times E_{AC} + N_{neur} T d_{state} \times E_{MAC}$ |
| 2 | **communication** | $fh \times E_{hop}$ |
| 3 | **delay storage** | $N_{access} \cdot T \cdot 2 \times E_{access}$, with $N_{access}=256$ (§1) |
| 4 | **state memory** | $N_{neur} T d_{state} \cdot 2 \times E_{access}$ |
| 5 | **leakage** | $bits_{state+buffer} \times P_{leak} \times t_{inf}$ |

**Delay storage is the only category where the two delay variants differ.** $N_{access}$ is
the **number of access ports per timestep**, not the number of stored delays: the buffer is
`(B, 256, L-1)` in both variants, so one row of 256 slots is touched per timestep either way
(`delay_layers_pytorch.py`: `y_window` is `(N, L-1)` for both `axonal_recdel` and
`synaptic_recdel`).

**The stored delay *count* does differ** — $h^2 = 65{,}536$ for per-synapse versus $h = 256$
for per-neuron — and that count drives the buffer's **bit** cost in category 5, not the access
cost here. §5.3 quantifies it.

**$\int P_{idle}dt$, clock tree, data conversion, and feature extraction are not included** — that is the white paper's full §4.3 formula, which Study 1 does not require.

### 4.3 Weight leakage reported separately

**Weights occupy SRAM and leak, but they are independent of the temporal mechanism.** Folding them into the "temporal state share" would dilute the signal (they are 99.6% of all bits).

**So weight leakage is reported in its own column and excluded from criterion (a)'s numerator.**

### 4.4 Unit prices and sources

| Parameter | Value | Source |
|---|---|---|
| $E_{AC}$ | **0.9 pJ** | Horowitz 2014, Fig 1.1.9 (45nm 0.9V, 32-bit FP Add) |
| $E_{MAC}$ | **4.6 pJ** | Ibid. (FP Add 0.9 + FP Mult 3.7) |
| $E_{access}$ | **10 pJ** | Ibid. (8 KB SRAM read) |
| $P_{leak}$ | **11.9 nW/bit** | Horowitz 2014, p.2 (100 mW/MB, 45–32nm) |
| $E_{hop}$ | **3.5 pJ** | Davies et al. 2018 (Loihi), Table 2 (3.0/4.0 pJ per tile hop) |
| **Weight access** | **0** (resident) | Dampfhoffer 2023, L2372 (spatially expanded, "remove the weights accesses") |
| **Weight access** | **10 pJ** per synaptic op (in SRAM, §5.5) | Horowitz 2014, Fig 1.1.9 — same unit as $E_{access}$; reuse = 1, i.e. an upper bound |
| Buffer width | **1 bit/slot** | `delay_layers_pytorch.py:299` |
| $F$ | **2 GHz** | Self-defined (measurement baseline) |
| par | **1** | Self-defined (no literature basis) |

**Cross-check for $E_{hop}$**: dividing Gorski 2016's 45nm router $P_{dyn}$ ($P_{total}$ 16–40 mW minus the dominant $P_{stat}$, i.e. ~5–7.3 mW) by the throughput $1.5\times10^9$ flit/s yields 3.3–4.9 pJ/hop, consistent with Loihi's measured value.

---

## 5. Counting results (prices excluded)

### 5.1 Literature basis for the counting conventions

**All four counting definitions come from Dampfhoffer et al. 2023's LIF+inst energy model (Eq. 3.4 and 3.7):**

**Synaptic operations (Eq. 3.4, L2007):**

$$E_{IF+inst} = N_{syn} \cdot N_{spikes/syn} \cdot (E_R^{weight} + E_R^{state} + E_W^{state} + E_{AC})$$

**Per-timestep neuron update (Eq. 3.7, L2037–2038):**

$$E_{LIF+inst} = \underbrace{N_{syn} N_{spikes/syn}(\cdots)}_{\text{event-driven}} + \underbrace{N_{neur} T (E_R^{state} + E_W^{state} + E_{MAC})}_{\text{clock-driven}}$$

**Provenance of the four counts:**

| Count | Our formula | Dampfhoffer basis |
|---|---|---|
| **SOP** | spike × fanout | the $N_{syn} N_{spikes/syn}$ term of Eq. 3.4 |
| **neuron ops** | $N_{neur} \times T \times d_{state}$ | $N_{neur} T$ in Eq. 3.7; the compartment count $d_{state}$ is our PMSN extension |
| **state accesses** | neuron ops **× 2** | $E_R^{state} + E_W^{state}$ in Eq. 3.7; L2032–2034 states "**reading the state, multiplying it with a constant, and writing back the result**" |
| **buffer accesses** | $N_{delay} \times T \times 2$ | no direct source; analogized from the read+write pattern of state accesses (read the expiring slot + write the current spike) |

**Why only Eq. 3.7's second term, not the first**: Dampfhoffer L2039–2044 explains these are two **mutually exclusive** implementation routes — the first is event-driven (update state only when a spike arrives), the second is clock-driven (update all neurons every step), and the original text notes "the update at each timestep is more efficient". Our models are clock-driven (`single_step_forward` at `delay_layers_pytorch.py:95`), so only the second term applies.

**The following are all single-sample averages over **all 2,264 SHD test samples**.**

### 5.2 Raw counts

| T | Model | spikes | SOP | neuron ops | state acc. | buffer acc. |
|---|---|---|---|---|---|---|
| **50** | delay | 2,750 | 2,848,026 | 26,600 | 53,200 | 25,600 |
| **50** | LIF | 8,688 | 4,286,554 | 35,200 | 70,400 | 0 |
| **50** | MC | 9,565 | 4,430,112 | 281,600 | 563,200 | 0 |
| **100** | delay | 5,301 | 3,532,306 | 53,200 | 106,400 | 51,200 |
| **100** | LIF | 13,930 | 4,705,348 | 70,400 | 140,800 | 0 |
| **100** | MC | 15,672 | 5,239,870 | 563,200 | 1,126,400 | 0 |
| **140** | delay | 6,696 | 3,910,030 | 74,480 | 148,960 | 71,680 |
| **140** | LIF | 19,523 | 5,486,040 | 98,560 | 197,120 | 0 |
| **140** | MC | 21,026 | 5,833,939 | 788,480 | 1,576,960 | 0 |

**SOP includes the input → hidden1 layer, which an earlier version omitted**: `SOP = (input events + hidden1 spikes) × fanout_h1 + hidden2 spikes × fanout_h2`. Input events average ≈ 8,230 per inference and are identical for all three models, since all three share the same 140-channel pooled input.

**The delay network's fanouts are 256 (hidden1) and 276 (hidden2 → 20 out + 256 recurrent)**; PMSN's are 352 and 20.

**neuron ops = $N_{neur} \times T \times d_{state}$; state accesses = neuron ops × 2 (read + write); buffer accesses = $256 \times T \times 2$.**

**MC's $d_{state}$ is 8 real values, not 4** — the PMSN state is complex, so each of the four compartments contributes a real and an imaginary part.

**The delay network's spike counts are 26–34 % of LIF's** (2,750 vs 8,688 at T=50), and because its fanout is also smaller (256 vs 352), **its SOP comes out 30–35 % below LIF's** — the opposite of the per-neuron design, where the wider fanout of 372 cancelled the sparsity advantage.

### 5.3 Storage bits

The delay network stores three different things, and they must not be conflated:

| Model | weight bits | neuron-state bits | **delay values** | **spike window** | temporal total |
|---|---|---|---|---|---|
| delay | **5,554,816** | **17,024** | **2,097,152** | **13,824** | **2,128,000** |
| LIF | 6,015,616 | 22,528 | 0 | 0 | 22,528 |
| MC | 6,421,120 | **180,224** | 0 | 0 | 180,224 |

**The three pieces, from the code:**

| Piece | Size | Where it comes from |
|---|---|---|
| **delay values** | $256^2 = 65{,}536$ × 32 bit = **2,097,152** | `recurrent_delays`, shape `(256,256)` — one value per connection |
| **spike window** | $256 \times (L-1)$ × 1 bit = **13,824** | `y_window = zeros(B, N, L-1)` in `SynapticTorchScans.multi_step_forward_v2` — one bit per hidden2 neuron per slot |
| **neuron state** | $532 \times 32$ = **17,024** | membrane potentials |

**The window depth is $L = \lceil 1 + \max d + (1+\sigma) \rceil + 1$**, from `_build_mask`:
49 / 55 / 51 for T = 50 / 100 / 140 with $\sigma = 10$. **This is 11 slots deeper than
$\lceil \max d \rceil + 2$**, because the triangular kernel has half-width $1+\sigma$.

**The weight column excludes the delay values** (5,554,816 = (239,124 − 65,536) × 32), since
the 2.1 Mbit would otherwise be charged twice — once as parameters, once as delay storage.

**This is where per-synapse differs from per-neuron, and how**: a per-neuron design stores
$h = 256$ delays (8,192 bit) and its window is identical. **The per-synapse table costs 256×
more *delay values*, not 256× more *buffer*** — the runtime window is the same size in both
designs, because the recurrence is over neurons either way.

**Weights are 73 % of the delay model's bits and 96–97 % of the two PMSN models'**, so
weight leakage dominates the *total* leakage of every model; the delay table is what makes
the delay model's *temporal* share large.

### 5.4 Communication hops

| T | Model | 2×2 | 4×4 | 8×8 |
|---|---|---|---|---|
| 50 | delay | 8,832 | 60,962 | 441,018 |
| | LIF | 28,587 | 198,392 | 1,436,650 |
| | MC | 31,381 | 220,566 | 1,591,048 |
| 100 | delay | 17,374 | 119,507 | 882,011 |
| | LIF | 45,118 | 308,479 | 2,213,959 |
| | MC | 50,835 | 357,068 | 2,576,434 |
| 140 | delay | 21,762 | 148,861 | 1,086,171 |
| | LIF | 63,203 | 434,092 | 3,125,954 |
| | MC | 68,429 | 474,574 | 3,409,407 |

**delay's hop count is only 31–40 % of LIF's — sparser spikes buy less communication.** The
delay network fires 2,750 / 5,301 / 6,696 spikes against LIF's 8,688 / 13,930 / 19,523, and
hop count tracks spike count almost linearly once the placement is fixed.

### 5.5 Energy after applying unit prices (µJ / inference)

| T | Mesh | Model | synaptic | state | delay | leakage | comm | **w_read** | **total (resident)** | **total (SRAM)** | weights leak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **50** | 2×2 | delay | 2.686 | 0.532 | 0.256 | 9.093 | 0.031 | 28.480 | **12.597** | **41.078** | 23.752 |
| | | LIF | 4.020 | 0.704 | 0.000 | 0.145 | 0.100 | 42.866 | **4.969** | **47.834** | 38.672 |
| | | MC | 5.282 | 5.632 | 0.000 | 1.263 | 0.110 | 44.301 | **12.287** | **56.589** | 45.004 |
| **50** | 4×4 | delay | 2.686 | 0.532 | 0.256 | 2.273 | 0.213 | 28.480 | **5.960** | **34.440** | 5.938 |
| | | LIF | 4.020 | 0.704 | 0.000 | 0.036 | 0.694 | 42.866 | **5.454** | **48.320** | 9.668 |
| | | MC | 5.282 | 5.632 | 0.000 | 0.316 | 0.772 | 44.301 | **12.002** | **56.303** | 11.251 |
| **50** | 8×8 | delay | 2.686 | 0.532 | 0.256 | 0.568 | 1.544 | 28.480 | **5.585** | **34.066** | 1.485 |
| | | LIF | 4.020 | 0.704 | 0.000 | 0.009 | 5.028 | 42.866 | **9.761** | **52.627** | 2.417 |
| | | MC | 5.282 | 5.632 | 0.000 | 0.079 | 5.569 | 44.301 | **16.562** | **60.863** | 2.813 |
| **100** | 2×2 | delay | 3.424 | 1.064 | 0.512 | 11.350 | 0.061 | 35.323 | **16.410** | **51.733** | 29.626 |
| | | LIF | 4.559 | 1.408 | 0.000 | 0.160 | 0.158 | 47.053 | **6.285** | **53.338** | 42.734 |
| | | MC | 7.307 | 11.264 | 0.000 | 1.556 | 0.178 | 52.399 | **20.304** | **72.703** | 55.428 |
| **100** | 4×4 | delay | 3.424 | 1.064 | 0.512 | 2.837 | 0.418 | 35.323 | **8.255** | **43.579** | 7.407 |
| | | LIF | 4.559 | 1.408 | 0.000 | 0.040 | 1.080 | 47.053 | **7.086** | **54.140** | 10.684 |
| | | MC | 7.307 | 11.264 | 0.000 | 0.389 | 1.250 | 52.399 | **20.209** | **72.608** | 13.857 |
| **100** | 8×8 | delay | 3.424 | 1.064 | 0.512 | 0.709 | 3.087 | 35.323 | **8.796** | **44.119** | 1.852 |
| | | LIF | 4.559 | 1.408 | 0.000 | 0.010 | 7.749 | 47.053 | **13.726** | **60.779** | 2.671 |
| | | MC | 7.307 | 11.264 | 0.000 | 0.097 | 9.018 | 52.399 | **27.685** | **80.084** | 3.464 |
| **140** | 2×2 | delay | 3.862 | 1.490 | 0.717 | 12.606 | 0.076 | 39.100 | **18.751** | **57.851** | 32.923 |
| | | LIF | 5.391 | 1.971 | 0.000 | 0.187 | 0.221 | 54.860 | **7.770** | **62.631** | 49.972 |
| | | MC | 8.878 | 15.770 | 0.000 | 1.775 | 0.240 | 58.339 | **26.662** | **85.001** | 63.253 |
| **140** | 4×4 | delay | 3.862 | 1.490 | 0.717 | 3.152 | 0.521 | 39.100 | **9.741** | **48.841** | 8.231 |
| | | LIF | 5.391 | 1.971 | 0.000 | 0.047 | 1.519 | 54.860 | **8.928** | **63.789** | 12.493 |
| | | MC | 8.878 | 15.770 | 0.000 | 0.444 | 1.661 | 58.339 | **26.752** | **85.091** | 15.813 |
| **140** | 8×8 | delay | 3.862 | 1.490 | 0.717 | 0.788 | 3.802 | 39.100 | **10.658** | **49.758** | 2.058 |
| | | LIF | 5.391 | 1.971 | 0.000 | 0.012 | 10.941 | 54.860 | **18.315** | **73.175** | 3.123 |
| | | MC | 8.878 | 15.770 | 0.000 | 0.111 | 11.933 | 58.339 | **36.691** | **95.030** | 3.953 |

**T = 100 only — the delay-free reference of §7.5.** Reported separately because it was trained at
a single resolution; it does not enter the 9-cell criteria counts of §6.

| T | Mesh | Model | synaptic | state | delay | leakage | comm | **w_read** | **total (resident)** | **total (SRAM)** | weights leak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **100** | 2×2 | no-delay | 1.938 | 0.512 | 0.512 | 0.028 | 0.090 | 21.529 | **3.198** | **24.727** | — |
| | 4×4 | no-delay | 1.938 | 0.512 | 0.512 | 0.007 | 0.617 | 21.529 | **3.703** | **25.232** | — |
| | 8×8 | no-delay | 1.938 | 0.512 | 0.512 | 0.002 | 4.466 | 21.529 | **7.547** | **29.076** | — |

**Its `delay` column is the 256×(L−1) = 256×2 spike window, not a delay table** — with every delay
zero, no table is stored, so there is nothing to leak either (`weights leak` = not applicable).

**`weights leak` is reported separately and does not enter either total.**

**Two regimes side by side.** `total (resident)` assumes the weights are spatially expanded, so a synaptic operation costs only the AC arithmetic (§4.4). `total (SRAM)` charges every synaptic operation one 10 pJ SRAM read of its 32-bit weight, $\text{w\_read} = \text{SOP} \times E_{access}$; this is an **upper bound** (weight reuse = 1, see §8.6). Criterion (a) under both regimes is tabulated in §6.1.

**`synaptic` = input-layer SOP × E_AC + hidden SOP × E_AC + neuron ops × E_MAC.**

## 6. Criteria results

### 6.1 (a) Temporal-state share ≥ 10%

**Under the white paper's reading ("a non-trivial share of energy in *at least one* brain-inspired model"), (a) holds in 9 of 9 configurations without weight reads, and in 8 of 9 with them.**

| T | Mesh | delay | LIF | MC | (a) | delay (SRAM) | LIF (SRAM) | MC (SRAM) | (a) SRAM |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 2×2 | 78.4% | 17.1% | 56.1% | ✅ | **24.1%** | **1.8%** | **12.2%** | ✅ |
| 50 | 4×4 | 51.4% | 13.6% | 49.6% | ✅ | **8.9%** | **1.5%** | **10.6%** | ✅ |
| 50 | 8×8 | 24.3% | 7.3% | 34.5% | ✅ | **4.0%** | **1.4%** | **9.4%** | ❌ |
| 100 | 2×2 | 78.8% | 25.0% | 63.1% | ✅ | **25.0%** | **2.9%** | **17.6%** | ✅ |
| 100 | 4×4 | 53.5% | 20.4% | 57.7% | ✅ | **10.1%** | **2.7%** | **16.0%** | ✅ |
| 100 | 8×8 | 26.0% | 10.3% | 41.0% | ✅ | **5.2%** | **2.3%** | **14.2%** | ✅ |
| 140 | 2×2 | 79.0% | 27.8% | 65.8% | ✅ | **25.6%** | **3.4%** | **20.6%** | ✅ |
| 140 | 4×4 | 55.0% | 22.6% | 60.6% | ✅ | **11.0%** | **3.2%** | **19.1%** | ✅ |
| 140 | 8×8 | 28.1% | 10.8% | 43.3% | ✅ | **6.0%** | **2.7%** | **16.7%** | ✅ |

**Left half is the weights-resident regime; right half charges one 10 pJ weight read per synaptic operation (§5.5).**

**Without weight reads, delay ranges 24.3–79.0 %, MC 34.5–65.8 %, and plain LIF 7.3–27.8 %.** The delay
network's share is *high* precisely because the per-synapse delay table is huge (§5.3): at 2×2 its
temporal state is 80.8 % of its own energy.

**With weight reads the picture inverts for the delay model in a surprising way.** Its SRAM share is
still 26.6–30.0 % at 2×2 — the buffer term is so large that even a 48 µJ weight-read charge only
halves it. At 8×8 it drops to 3.7–5.4 % because communication and weight reads both grow while the
buffer's leakage shrinks with $t_{inf}$.

**MC is the only model that stays above 10 % in all nine SRAM configurations except T=50 at 8×8,
where it reaches 9.4 % — 0.6 point short.** Plain LIF never exceeds 3.4 % in SRAM.

#### Where the delay-free reference would sit

**It would pass (a) in all three of its configurations — 32.9 / 27.8 / 13.6 % resident — and fail all
three under weight reads (4.3 / 4.1 / 3.5 %).** We deliberately keep it out of the 9-cell count
above, for two reasons:

1. **A small model passes (a) more easily.** The delay-free network's temporal state is ~1.05 µJ, *the
   same absolute amount* as plain LIF's 1.41 µJ at T=100 (both are just the neuron state plus a
   2-slot window). It clears 10 % only because its total energy is 3.20 µJ rather than LIF's 6.29.
   Counting it would turn 9/9 into 12/12 without adding evidence.
2. **It has no delay table, so its "temporal state" is a different quantity.** For delay, 98 % of the
   temporal state is the delay table (§5.3); for no-delay there is no table at all.

**What it does show is the boundary of the criterion**: (a) measures the *share*, not the *amount* of
temporal state, and a network that removes the dominant temporal-state term altogether can satisfy
the share test purely by shrinking its denominator.

### 6.2 (b) Ranking flips

| T | Mesh | spike order | SOP order | energy order | spike≠E | SOP≠E | energy order (SRAM) | SOP≠E (SRAM) |
|---|---|---|---|---|---|---|---|---|
| 50 | 2×2 | DL<LIF<MC | DL<LIF<MC | LIF<MC<DL | ✅ | ✅ | **DL<LIF<MC** | ❌ |
| 50 | 4×4 | DL<LIF<MC | DL<LIF<MC | LIF<DL<MC | ✅ | ✅ | **DL<LIF<MC** | ❌ |
| 50 | 8×8 | DL<LIF<MC | DL<LIF<MC | DL<LIF<MC | ❌ | ❌ | **DL<LIF<MC** | ❌ |
| 100 | 2×2 | DL<LIF<MC | DL<LIF<MC | LIF<DL<MC | ✅ | ✅ | **DL<LIF<MC** | ❌ |
| 100 | 4×4 | DL<LIF<MC | DL<LIF<MC | LIF<DL<MC | ✅ | ✅ | **DL<LIF<MC** | ❌ |
| 100 | 8×8 | DL<LIF<MC | DL<LIF<MC | DL<LIF<MC | ❌ | ❌ | **DL<LIF<MC** | ❌ |
| 140 | 2×2 | DL<LIF<MC | DL<LIF<MC | LIF<DL<MC | ✅ | ✅ | **DL<LIF<MC** | ❌ |
| 140 | 4×4 | DL<LIF<MC | DL<LIF<MC | LIF<DL<MC | ✅ | ✅ | **DL<LIF<MC** | ❌ |
| 140 | 8×8 | DL<LIF<MC | DL<LIF<MC | DL<LIF<MC | ❌ | ❌ | **DL<LIF<MC** | ❌ |

**At least one flip: 6/9 resident, 0/9 in SRAM.** (Resident: 6/9 spike, 6/9 SOP.)

**Delay's SOP is the lowest of the three models in every configuration** (`DL<LIF<MC` at all
three resolutions and all three meshes). This is what the per-synapse design produces: its
spikes are 26–34 % of LIF's *and* its fanout is 256 instead of 352, so the two effects
compound rather than cancel.

**That single fact drives the whole criterion-(b) result:**

- **Resident regime** — the delay model's energy is dominated by its 2.1 Mbit delay table, so
  it costs the most at 2×2 and 4×4 even though it emits the fewest spikes. **SOP ranks it
  cheapest, energy ranks it dearest → flip.** At 8×8 the delay table's leakage collapses with
  $t_{inf}$ and the model becomes genuinely cheapest, so the two rankings finally agree —
  **that is the 3/9 failure** (T=50, 100, 140 at 8×8).
- **SRAM regime** — $w\_read = \text{SOP} \times 10$ pJ is 68–84 % of the total, so the energy
  ranking *is* the SOP ranking by construction. **0/9 is arithmetic, not a modelling error.**

**The shift from per-neuron to per-synapse is what moved this**: with per-neuron delays the
fanout was 372, which cancelled the sparsity advantage and put the delay model's SOP in the
middle — that is why an earlier version reported 7/9 and 6/9. With $h=256$ and a fanout of
276 the delay model is simply the sparsest *and* the cheapest on SOP, and criterion (b) has
less to catch.

**Weight reads therefore make criterion (b) *weaker*, not stronger**: charging every
operation for a weight fetch raises SOP's share of the total until it dominates, so the
SOP-only ranking converges on the full-model ranking. This is a limitation of the accounting,
not of the models — and it is the honest result for a hardware design in which weights are
re-read on every operation.

#### Where the delay-free reference would sit

**Adding it creates no new flip, because it lands first in both rankings:**

| | SOP order | energy order (2×2, resident) |
|---|---|---|
| three models | DL < LIF < MC | LIF < DL < MC |
| **with no-delay** | **ND(2.15M) < DL(3.53M) < LIF < MC** | **ND(3.20) < LIF(6.29) < DL(16.41) < MC** |

- **ND vs DL**: SOP says ND < DL; energy also says ND < DL → **agree, no flip.**
- **DL vs LIF**: SOP says DL < LIF; energy says LIF < DL → **flip, already counted in the 6/9.**

**Every pairwise comparison among the original three is unchanged**, so the counts above stand
whether or not ND is included. ND itself is the cheapest on both counts, making it another instance
of the weakness described in the previous paragraph rather than new evidence.

### 6.3 Summary

| Criterion | weights resident | weights in SRAM |
|---|---|---|
| **(a) ts ≥ 10%** | **9/9** | **8/9** |
| (b) spike ≠ energy | 6/9 | **0/9** |
| (b) SOP ≠ energy | 6/9 | **0/9** |
| **(b) at least one flip** | **6/9** | **0/9** |
| Energy ranking changed by the weight term | — | **0/9 configurations** |

**Criterion (a) holds in both regimes** — 9/9 without weight reads and 8/9 with them, the single
miss being MC at T=50/8×8 where it reaches 9.4 %, 0.6 point short.

**Criterion (b) holds only in the resident regime (6/9).** The three misses are the 8×8 cases,
where the delay model is genuinely the cheapest model *and* emits the fewest spikes, so SOP,
spike count and the full model all rank it first.

**In the SRAM regime (b) fails completely, and the reason is structural**: `w_read ∝ SOP`
supplies 68–84 % of the total, so the energy ranking cannot differ from the SOP ranking. This
is a property of the accounting, not of the three models — the same thing would happen to any
set of models under that assumption.

**Contrast with per-neuron**: an earlier per-neuron version reported (b) at 7/9 resident and
6/9 in SRAM. The difference is not a modelling choice but a consequence of the per-synapse
design — its fanout drops from 372 to 276, so its SOP falls *below* LIF's instead of sitting
in the middle. **Criterion (b) is therefore weaker in this version, and the report says so
rather than picking the favourable accounting.**

## 7. Analysis

### 7.1 Both simplified estimates rank delay cheapest, and it is really second

**T=100, 2×2:**

| Estimate | Order | Verdict on delay |
|---|---|---|
| **spike count** | delay(5.3k) < LIF(13.9k) < MC(15.7k) | **cheapest** (actually **2nd**) |
| **SOP** | delay(3.53M) < LIF(4.71M) < MC(5.24M) | **cheapest** (actually **2nd**) |
| **full model** | LIF(6.28) < **delay(16.41)** < MC(20.30) | **2nd** |

**Both estimates point the same way, and both are wrong about the magnitude** — the full model
puts the delay network at 16.41 µJ, **2.6× plain LIF**, not the cheapest of the three.

**The miss is the delay table, not the delays.** §5.3: 65,536 delay values occupy 2.1 Mbit,
whose leakage alone is **11.35 of those 16.41 µJ** (69 %). Nothing in spike count or SOP knows
that table exists.

### 7.2 The multi-compartment cost is visible only in the full model

| T=100, 2×2 | delay | LIF | **MC** |
|---|---|---|---|
| spikes | 5,301 | 13,930 | 15,672 |
| SOP | 3.53M | 4.71M | 5.24M |
| **state** | 1.064 | 1.408 | **11.264 (8.0×)** |
| **leakage** | **11.350** | 0.160 | 1.556 |
| full model | **16.41** | 6.28 | **20.30** |

**$d_{state}=8$ multiplies both state accesses and bits by 8, but SOP contains no such factor** —
MC's 8× state cost is invisible to SOP.

**Symmetrically, the delay table is invisible to every simplified estimate.** The two models are
expensive for completely different reasons: MC because its per-neuron state is 8 real values, the
delay network because its per-synapse table holds 65,536 delay values. That is worth stating
plainly — a spike- or SOP-based analysis ranks **both** of them cheap, and the delay network is
in fact ranked cheapest of all on SOP while being the second most expensive model.

### 7.3 At large mesh, communication overtakes state

| T=100, 8×8 | delay | LIF | MC |
|---|---|---|---|
| comm | 3.087 | 7.749 | 9.018 |
| comm share | **35%** | **56%** | **33%** |

**At 64 cores the delay network becomes the cheapest (8.80 vs 13.73 and 27.69).** Its hop count
is only 40 % of LIF's because it fires 5,301 spikes against 13,930 — the per-synapse delays
suppress redundant activity.

**But its energy advantage at 8×8 comes from three terms at once**, not from communication alone:
its delay-table leakage drops to 0.71 µJ (from 11.35 at 2×2, because $t_{inf}$ shrinks 16×), its
comm is 3.09 against LIF's 7.75, and its synaptic term is 3.42 against MC's 7.31.

**Note the inversion from 2×2:** there the delay network was the *second* most expensive
(16.41 vs MC's 20.30 and LIF's 6.28); at 8×8 it is the cheapest. The mesh size changes which
term dominates,
and the delay model is the one whose cost is most sensitive to that.

### 7.4 Effect of time-step resolution

**This version no longer shows a monotone trend.** delay is 84.54 / 84.30 / 83.81 % at
T = 50 / 100 / 140 — flat within 0.7 points, with a slight *decrease* at finer frames. Both PMSN
variants peak at 28 ms (T=50).

| | Mechanism | Behaviour |
|---|---|---|
| delay | discrete frame-level delays | **insensitive** here; 256² delays absorb whatever resolution is given, and accuracy varies <0.7 pt |
| PMSN | continuous time constants | **peaks at 28 ms**, closest to SHD phoneme durations of 50–200 ms |

**The earlier claim that delay "prefers finer resolution" does not survive the per-synapse
change.** At $h=352$ with per-neuron delays the model did improve with finer frames
(86.56 → 87.24 → 87.99). With a per-synapse table and $h=256$ the effect disappears — the delay
grid is dense enough that frame resolution is no longer the limiting factor. What limits accuracy
now is the parameter budget, not the delay granularity.

### 7.5 The delay-free reference network (TASK 4)

**The brief asks for one delay-free model at T=100 with the same architecture and matching accuracy. The first half is achievable; the second is not, and the reason is informative.**

We trained `SNN_vanilla_synaptic_recurrent` — the identical per-synapse architecture of §1 with
`recurrent_delays` fixed at 0 and frozen (65,536 zeros, `requires_grad=False`), leaving **172,564
trainable parameters** against the delay model's 173,588 (the difference is the two BatchNorm
layers). Two schedulers were tried, both 10 epochs, $h=256$:

| Scheduler | Test acc | Best val | Best epoch |
|---|---|---|---|
| cosine | **49.81 %** | 50.20 % | 9 |
| one-cycle | 49.21 % | 47.03 % | 9 |

**Both plateau around 50 %, 34.5 points below the delay model's 84.30 %.**

#### BatchNorm had to be removed

With BatchNorm enabled the network did not learn at all (train accuracy 13 %, validation 24 % — the
inverted train/val gap that signals running statistics that do not describe the batches). Its
first-order recurrence makes each layer's input strongly autocorrelated across adjacent timesteps, so
40-sample batches estimate the statistics poorly. Disabling BatchNorm raised the ceiling to 49.81 %;
it stays there. **The delay model trains fine with BatchNorm**, because its feedback arrives 10–31
frames late, which decorrelates it.

#### Why it cannot be trained — the paper predicts this

DelRec states the mechanism explicitly ([arXiv:2509.24852](https://arxiv.org/html/2509.24852v2), §A.4
*"Recurrent delays mitigate the risk of exploding gradients"*):

> "recurrent delays may mitigate gradient challenges by implementing **temporal skip connections**,
> improving gradient propagation during training"

and in Fig. 1B:

> "Delays in recurrent connections reduce the risks of exploding or vanishing gradients by **bridging
> distant time steps**."

With all delays zero the feedback path is one step long. That is exactly the configuration the paper
identifies as ill-conditioned; removing the delays removes the skip connections.

#### The evidence that it is not a capacity problem

PMSN's generalised multi-compartment model **reduces to a standard LIF when $n=1$ and $\beta=0$**
([PMSN §IV](https://arxiv.org/html/2408.14917v1), immediately after Eq. 4: *"when the number of
compartments n is set to 1 and β = 0, our model is simplified to a standard LIF model"*). Our
`pmsn_lif` checkpoint was trained with `d_state=1`, i.e. it **is** a standard LIF — and reaches
**86.40 %** at T=100.

| Model | Topology | Neuron | Test acc |
|---|---|---|---|
| PMSN `lif` (n=1) | **feedforward** | standard LIF | **86.40 %** |
| PMSN `mc` (n=8) | feedforward | multi-compartment | 88.30 % |
| delay | **recurrent** | LIF + per-synapse delays | **84.30 %** |
| no-delay | **recurrent** | standard LIF, no delays | **49.81 %** |

**A standard LIF reaches 86 % in a feedforward net but only 50 % inside a delay-free recurrence.**
The neuron is not the limitation, the training budget is not the limitation (PMSN needs 5 epochs) —
the *recurrent connection without temporal skip* is. The recurrence actively costs 36.6 points
relative to simply not having one.

#### What the delay table buys

Accounting as in §4, no delay table to store (`n_delay = 0`), spike window only needs $L=3$
($t-1$, $t-2$; σ=0), 2,264 test samples:

| | no-delay | delay | ratio |
|---|---|---|---|
| SOP (T=100) | 2,152,860 | 3,532,306 | 0.61× |
| spikes per inference | 7,978 | 5,301 | 1.51× |
| storage bits | **8,704** | 2,128,000 | **0.004×** |
| energy 2×2, resident | **3.20 µJ** | 16.41 µJ | **0.20×** |
| energy 4×4, resident | 3.70 | 8.26 | 0.45× |
| energy 8×8, resident | 7.55 | 8.80 | 0.86× |
| energy 2×2, weights in SRAM | 24.73 | 51.73 | 0.48× |
| temporal-state share (2×2) | 32.9 % | 78.8 % | — |

**The delay model's 16.41 µJ is dominated by 11.35 µJ of leakage on its 2.1 Mbit delay table.**
Removing the table removes that, and the whole network costs 3.20 µJ — but delivers 49.81 % instead
of 84.30 %.

$$\underbrace{3.20\ \mu J}_{\text{no-delay}}\ \xrightarrow{\ +13.2\ \mu J\ }\ \underbrace{16.41\ \mu J}_{\text{delay}}\qquad\Longleftrightarrow\qquad 49.81\%\ \to\ 84.30\%$$

**Roughly 0.38 µJ per accuracy point at 2×2. The gap narrows with mesh size** — at 8×8 delay is only
14 % more expensive than no-delay, because communication dominates and the delay model emits 1.5×
fewer spikes.

#### Consequence for the report

**TASK 4 cannot supply an accuracy-matched delay-free control.** We therefore do not use it to price
the delay mechanism; §5's cost figures describe the delay model itself, and the delay-free network
serves only as a *lower bound on cost* and an *upper bound on how badly the recurrence degrades
without delays*. Both bounds are reported above.

---

## 8. Placement and cost-model choices affect the result

**This is the report's most important caveat: the absolute numbers depend on the following seven modelling choices, each of which has alternatives.**

### 8.1 Placement is not optimized

**Sequential mapping puts the entire output layer on the last core (core 15 at 4×4, cores 62–63 at 8×8).**

**Effect**: hidden2 → out traffic forms a hotspot and hop counts are systematically inflated. **A communication-aware placement (e.g. spreading the output layer) would lower flit_hops and thus the comm term.**

**At 8×8 the comm term is 29–56% of the total, so this choice matters most.**

### 8.2 Multicast is computed as a unicast upper bound

**Current hop count = sum of per-target-core shortest paths, equivalent to sending an independent unicast to each target.**

**A real NoC multicast uses a replication tree, copying only at path forks.**

**Effect**: flit_hops is overestimated 2–3×, and the comm term scales accordingly. **All three models are inflated by the same factor, so the ordering is unaffected.**

### 8.3 Multiple spikes in the same timestep are not aggregated

**Each spike is counted independently. Physically, multiple spikes from the same source core to the same target set within one timestep could share one flit payload.**

**Effect**: same direction as §8.2, further overestimating comm.

### 8.4 Time model par = 1 has no literature basis

**Assumes a 32-bit FP MAC completes in one cycle at 2 GHz.**

**Effect**: if a MAC actually needs 4 cycles, $t_{inf}$ is 4× larger and $P_{leak} \times t_{inf}$ scales accordingly.

**But Study 1's leakage is only 4.3–14 % of the delay model (excluding weights), so criteria (a)(b) are barely affected. Weight leakage (29.63 µJ at T=100, 2×2) is heavily affected, but it is reported separately.**

### 8.5 Mixed process nodes

| Parameter | Node |
|---|---|
| $E_{AC}$ / $E_{MAC}$ / $E_{access}$ / $P_{leak}$ | **45 nm** |
| $E_{hop}$ | **14 nm** (Loihi) |

**Node scaling (14→45 nm, about 2.75×) would make $E_{hop}$ ≈ 9.6 pJ/hop, scaling the comm term up accordingly.**

### 8.6 The weight-read term assumes no reuse (§5.5)

**Every synaptic operation is charged one full SRAM access (10 pJ) to fetch its 32-bit weight. A real core reads an entire SRAM line and reuses it across several consecutive synaptic operations.**

**Effect**: with a reuse factor $R$, the term scales as $1/R$. At $R=4$ (a 128-bit line feeding four 32-bit operations) the T=100 delay model's weight term drops from 35.32 µJ to 8.83 µJ, its total rises from 16.41 to 25.24 µJ, and its temporal-state share falls from **78.8 % to 51.2 %** — still the dominant term, but a very different number. The resident case corresponds to $R=\infty$ (78.8 %).

**So the "weights in SRAM" column of §5.5 is a worst case, not a prediction.** The two regimes bracket the truth: weight reuse is what decides whether temporal state matters, which is itself an architectural question the white paper's Thrust 3 (memory tiers) is meant to answer.

### 8.7 How to read these numbers

**Trustworthy:**
- The relative ranking of the three models (under one set of modelling assumptions)
- The order of magnitude of the temporal-state share (tens of percent, not 1% and not 90%)
- The conclusion that "simplified estimates mis-rank"

**Not trustworthy:**
- Absolute µJ values (could differ by 2–3×)
- Any cross-work comparison of the form "X times more efficient than chip Y"

---

## 9. Reproduction

### 9.1 Dependencies

- Python 3.11+, PyTorch, h5py, numpy, matplotlib
- DelRec source (`alexmaxad/DelRec`) and the SHD dataset
- Three model checkpoints (paths in §9.4)

**Training was done on Google Colab with its default environment; measurement runs on local CPU (`map_location="cpu"`). Exact package versions were not pinned beyond `spikingjelly@d4fee3a1`.**

### 9.2 From scratch

**Training** — the nine runs are driven by `training_bundle/`, which ships every
modified file so there is no patching step:

```bash
unzip -q training_bundle.zip -d /content
bash /content/bundle/runs/02_delay_T100.sh      # delay      (01..03)
bash /content/bundle/runs/06_pmsn_lif_T100.sh   # PMSN       (04..09)
bash /content/bundle/runs/10_nodelay_cos.sh     # no-delay   (10..11, T=100 only)
# ... 11 runs in total; see patches/TRAINING.md and §7.5
```

**The two no-delay scripts are also copied verbatim into `patches/runs/`, so they travel with this
deliverable and do not depend on the bundle.** Both set `use_batch_norm = False` and
`max_grad_val = 1.5` before launching — without those two changes the network does not learn (§7.5).

**Measurement** — reads the nine checkpoints under `checkpoints/` and writes the
result JSON:

```bash
python scripts/study1_measure.py --n -1 --save results/study1_h256_measure_full.json
```

`--n -1` is the default and means all 2,264 test samples; `--n 10` is a fast smoke
test. Checkpoint paths are relative to `scripts/`, so the deliverable directory can
be moved anywhere. Two external paths come from environment variables:

```bash
export DELREC_ROOT=/content/DelRec     # default: D:\DelRec
export SHD_TEST_H5=/content/DelRec/Datasets/SHD/extract/shd_test.h5
```

```bash
# 3. Plot (~5 s), reads results/study1_h256_measure_full.json
python scripts/plot_study1.py

# 4. Consistency check: 10-sample subset vs the stored full-set JSON (~2 min)
python scripts/check_10_samples.py
```

**All three scripts locate their inputs relative to their own location**, so no
path editing is needed after moving the directory.

### 9.3 Data reliability

**Cross-validation, 10 samples vs 2,264 samples** (produced by `scripts/check_10_samples.py`,
which runs the same measurement on the first 10 samples and divides by the stored full-set
values):

| Model | T | SOP ratio | flit_hops ratio |
|---|---|---|---|
| delay | 50 / 100 / 140 | 1.014 / 1.009 / 1.019 | 1.000 / 0.987 / 1.020 |
| LIF | 50 / 100 / 140 | 1.014 / 1.017 / 1.011 | 1.014 / 1.014 / 1.003 |
| MC | 50 / 100 / 140 | 1.017 / 1.011 / 1.013 | 1.019 / 1.004 / 1.012 |

**All 27 ratios fall within ±2.0 % of unity, so the 2,264-sample average has converged.** The
first ten samples happen to run slightly above average (most ratios >1), but by less than the
sampling error a 10-sample subset would justify — which is the point: the full-set numbers are
stable, and the 10-sample shortcut is good to ~2 %.

### 9.4 Checkpoint locations

All eleven checkpoints ship with this deliverable, under `checkpoints/`:

```
checkpoints/
├── delay_T50/      best.pth  last.pth  config.json  final_test.json
│                   final_delays.npz  init_delays.npz
├── delay_T100/     same + train_res.csv + val_res.csv
├── delay_T140/     same as delay_T50
├── pmsn_lif_T50/   checkpoint.pth  log.txt
├── pmsn_mc_T50/    checkpoint.pth  log.txt
├── pmsn_lif_T100/  checkpoint.pth  log.txt
├── pmsn_mc_T100/   checkpoint.pth  log.txt
├── pmsn_lif_T140/  checkpoint.pth  log.txt
├── pmsn_mc_T140/   checkpoint.pth  log.txt
├── nodelay_cos_10ep/       best.pth  last.pth  config.json  final_test.json
│                           train_res.csv  val_res.csv  final_delays.npz
└── nodelay_onecycle_10ep/  same
```

**The two `nodelay_*` runs are §7.5's delay-free reference** (T=100 only). Their `final_delays.npz`
is present for symmetry but is all zeros. Reproduce their energy with
`python scripts/measure_nodelay.py --samples 2264`.

**All paths are resolved relative to `scripts/study1_measure.py`**, so the directory
can be moved or copied without editing anything. Only the DelRec checkout and the
SHD `.h5` file are external — set `DELREC_ROOT` and `SHD_TEST_H5` if they are not at
their defaults (§9.2).

**`delay_T50` and `delay_T140` lack `train_res.csv` / `val_res.csv`**: those two runs
were exported from Colab without the training logs. It has no effect on any reported
number — the measurement reads only `best.pth` and `final_delays.npz` — but it means
no per-epoch curve is available for those two resolutions.

---

## 10. Known limitations

| # | Limitation | Impact | See |
|---|---|---|---|
| 1 | `par = 1` has no literature basis | $t_{inf}$ and weight leakage | §8.4 |
| 2 | $E_{hop}$ is a 14nm value mixed into a 45nm model | comm term | §8.5 |
| 3 | $P_{leak}$ is an 8 MB cache value (small SRAM leaks more) | leakage underestimated | `PARAMETERS.md` |
| 4 | flit_hops is a unicast upper bound | comm overestimated 2–3× | §8.2 |
| 5 | Same-timestep spikes not aggregated | comm further overestimated | §8.3 |
| 6 | Placement not optimized | output-layer hotspot | §8.1 |
| 7 | FP32 conflicts with the white paper's §7.1 "low-precision" | narrative only (INT32 differs <0.5%) | `PARAMETERS.md` |
| 8 | $\int P_{idle}dt$ excluded | matches Study 1's five categories; not a defect | §4.2 |
| 9 | Weight leakage reported separately | a reporting choice, not a defect | §4.3 |
| 10 | Weight reads assume **no reuse** ($R=1$) | the "in SRAM" column is an upper bound; at $R=4$ delay's share is 51.2 % instead of 78.8 % | §8.6 |
| 11 | The delay and LIF networks come from **different code bases** (DelRec vs PMSN) | the delay-vs-LIF comparison is **not controlled**; the SOP gap may be architectural, so no causal claim from the §6.2 flips | follow-up Task 4 |
| **12** | **Per-synapse delays cost ~2 points of accuracy** | delay reaches 84.54 / 84.30 / 83.81 % against per-neuron delays' 86.56 / 87.24 / 87.99 % at $h=352$. The per-synapse table is what makes the network parameter-matched to PMSN, so the choice is deliberate — but it means the three models are **not** accuracy-matched, and the delay network is consistently the weakest | §1 |
| **13** | **Ring depth varies with resolution** | $L = \lceil 1 + \max d + (1+\sigma) \rceil + 1$ is read per checkpoint: **49 / 55 / 51** for T = 50 / 100 / 140 with $\sigma = 10$. A trained delay distribution sets it, so the buffer cost is not constant across resolutions | §5.3 |
| **14** | **Criterion (b) is vacuous in the SRAM regime** | $w\_read \propto \text{SOP}$ supplies 68–84 % of the total, so the energy ranking is the SOP ranking by construction and (b) is 0/9. This is a property of the accounting, not of the models; the resident regime is where (b) has content (6/9) | §6.2, §6.3 |
| **15** | **The delay values are counted as "delay storage"** | 2.1 Mbit is charged to category 3 rather than to the weight column. Counting them as weights instead would move the same 2.1 Mbit into `weights_leak` and drop the delay model's temporal-state share from 78.8 % to about 31 % at T=100/2×2. The choice follows the white paper's wording ("delay storage"), but it is a choice | §5.3 |

---

## 11. Figures

All three are generated by `scripts/plot_study1.py`, which reads
`results/study1_h256_measure_full.json` directly — so the figures cannot drift from the tables.

| File | Question it answers |
|---|---|
| `figs/study1_main_T100_2x2.png` | The white paper's requested form: stacked energy breakdown of the three models with spike counts overlaid (T=100, 2×2). The delay bar is dominated by leakage from its 2.88 Mbit delay table |
| `figs/study1_grid_resolution_mesh.png` | How resolution and mesh size change the energy composition (3 resolutions × 2 meshes) |
| `figs/study1_ranking_flip.png` | Ranking by the three estimates at each resolution (2×2), marking which pair agrees — at T=50 and T=140 the delay network ranks cheapest on all three, at T=100 it ranks most expensive |

---

## 12. References

| Use | Reference |
|---|---|
| FP32 unit prices | Horowitz, *Computing's Energy Problem (and what we can do about it)*, ISSCC 2014, Fig 1.1.9 |
| Per-hop communication | Davies et al., *Loihi: A Neuromorphic Manycore Processor with On-Chip Learning*, IEEE Micro 2018, Table 2 |
| Architecture assumption (resident weights) | Dampfhoffer et al., *Are SNNs Really More Energy Efficient Than ANNs?*, 2023, §3.3 |
| Router power (cross-check) | Gorski, *Dissertation*, 2016, Table I.5 / I.7 |
| delay model | DelRec (`alexmaxad/DelRec`) |
| PMSN model | Chen et al., *PMSN: A Parallel Multi-compartment Spiking Neuron*, arXiv:2408.14917 |
| Criteria source | *Time as a First-Class Resource*, §4.3 and Preliminary Studies, p.11–12 |
