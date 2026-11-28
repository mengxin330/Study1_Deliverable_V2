# Study 1 — How large is the energy share of temporal state in brain-inspired SNNs?

**[English](README.md) | [中文](README.zh.md)**

> Preliminary Study 1 of the NSF proposal *Time as a First-Class Resource*
> Data collected: 2026-11-27
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
values instead of $h = 256$ — **393 kbit of delay storage at 6 bits per value** (§5.3). That is the single largest
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
| **Synaptic delay** | A per-synapse field in the 5-tuple $(i, j, \text{wgt}, \text{dly}, \text{tag})$ — one delay value per synapse | **Also per-synapse** since the correction (§1). Loihi stores integer delays with no kernel spread; DelRec trains a continuous delay and convolves it with a triangular kernel of half-width $\sigma$, so the *representation* still differs |
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
| 3 | **delay storage** | $N_{access} \times E_{access}$ (see below) |
| 4 | **state memory** | $N_{neur} T d_{state} \cdot 2 \times E_{access}$ |
| 5 | **leakage** | $bits_{state+buffer} \times P_{leak} \times t_{inf}$ |

**Category 3 changes with the delay variant, and its count is a per-synapse count.**

The white paper writes $N_{access}\cdot T\cdot 2$ without defining $N_{access}$. The earlier
version of this report filled in $N_{access}=256$ — one row of the `(B, 256, L-1)` window
per timestep — **which is the PyTorch implementation, not a per-synapse hardware count.**

With one delay per connection, every recurrent edge reads a different time offset
(`y_i[t-d_ij]`), so the count is per connection, not per neuron:

$$N_{access} = h_2^{\text{spikes}} \times 256$$

i.e. the number of hidden2 spikes times the number of hidden2 targets. At T=100 that is
877,744 lookups instead of 25,600.

**Only the recurrent edges carry a delay.** The input layer (61% of SOP), the feedforward
hidden1→hidden2 edges and the hidden2→out edges have no delay entry to read and are not
charged here.

**The stored delay *count* still differs** — $h^2 = 65{,}536$ for per-synapse versus $h = 256$
for per-neuron — and that count drives the delay table's **bit** cost in category 5, not the
access cost here. §5.3 quantifies it.

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

### 4.5 Number labels

**Every number in this report carries one of four labels:**

| Quantity | Label | Source |
|---|---|---|
| accuracy | **measured** | trained checkpoint evaluated on the 2,264-sample SHD test split |
| spike count | **measured** | forward pass |
| SOP | **measured** | spikes × fanout |
| `flit_hops` | **measured** | spikes + placement + topology |
| $E_{AC}$ = 0.9 pJ | **characterized** | Horowitz 2014 |
| $E_{MAC}$ = 4.6 pJ | **characterized** | Horowitz 2014 |
| $E_{ACC}$ = 10 pJ | **characterized** | Horowitz 2014 |
| $P_{leak}$ = 11.9 nW/bit | **characterized** | 8 MB cache leakage |
| $E_{hop}$ = 3.5 pJ | **characterized** | Loihi / Gorski |
| $F$ = 2 GHz, par = 1 | **characterized** | assumption, §8.4 |
| every µJ in §5.5 | **projected** | measured counts × characterized prices |
| every bit count in §5.3 | **projected** | parameter / state counting |
| $t_{inf}$ | **projected** | $(\text{SOP} + N T d)/(n F)$ |
| realized transit latency | **simulated** | none in Study 1; Study 2 will use it |

**How the tables refer to it**: §5.2 is `all measured`; §5.3, §5.5 and §6 are `projected (§4.5)`.

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
| delay | **5,554,816** | **16,384** | **393,216** | **13,824** | **423,424** |
| LIF | 6,015,616 | 22,528 | 0 | 0 | 22,528 |
| MC | 6,421,120 | **180,224** | 0 | 0 | 180,224 |

**The three pieces, from the code:**

| Piece | Size | Where it comes from |
|---|---|---|
| **delay values** | $256^2 = 65{,}536$ × **6 bit** = **393,216** | `recurrent_delays`, shape `(256,256)` — one value per connection |
| **spike window** | $256 \times (L-1)$ × 1 bit = **13,824** | `y_window = zeros(B, N, L-1)` in `SynapticTorchScans.multi_step_forward_v2` — one bit per hidden2 neuron per slot |
| **neuron state** | $512 \times 32$ = **16,384** | membrane potentials; the 20 output units are a plain `Linear` and carry no state |

**The delay table is stored at 6 bits per value, not 32.** The trained delays take **43 integer
values (0–42)**, so 6 bits suffice; storing them as FP32 over-provisioned the dominant term by
5.3×. This is the single largest correction in this version: the delay table drops from
2,097,152 to 393,216 bits, and its leakage from 11.35 µJ to 2.26 µJ at T=100/2×2.

**The window depth is $L = \lceil 1 + \max d + (1+\sigma) \rceil + 1$**, from `_build_mask`:
49 / 55 / 51 for T = 50 / 100 / 140 with $\sigma = 10$. **This is 11 slots deeper than
$\lceil \max d \rceil + 2$**, because the triangular kernel has half-width $1+\sigma$.

**The weight column excludes the delay values** (5,554,816 = (239,124 − 65,536) × 32), since
the 393 kbit would otherwise be charged twice — once as parameters, once as delay storage.

**This is where per-synapse differs from per-neuron, and how**: a per-neuron design stores
$h = 256$ delays (1,536 bit at 6 bits each) and its window is identical. **The per-synapse table
costs 256× more *delay values*, not 256× more *buffer*** — the runtime window is the same size in
both designs, because the recurrence is over neurons either way.

**Weights are 93 % of the delay model's bits and 96–97 % of the two PMSN models'**, so
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

**Two accesses per activated connection, not one.**

- **`window read`** — `spike_i[t − d_ij]` has to come out of storage. One access per
  activated recurrent connection, **unavoidable in every placement**.
- **`table read`** — reading the delay value `d_ij` itself. One more access, **only when
  the delay table sits in SRAM**. When the table is resident the offset is a constant.

`877,744` activated connections at T=100 (3,428.7 hidden2 spikes × 256 targets), so each
read costs **8.778 µJ** at 10 pJ. The measurements record their sum as `delay access`
(17.555 µJ); the two columns below split it.

**One table per delay-table placement, both weight placements side by side.** Every column
enters the total, so each row can be checked by addition. **The two totals differ by exactly
`w_read`.**

#### Delay table resident

| T | Mesh | Model | synaptic | neuron | state | window read | table read | leakage | comm | w_read | **total (W resident)** | **total (W in SRAM)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 2×2 | delay | 2.563 | 0.118 | 0.512 | 4.769 | 0.000 | 0.123 | 0.031 | 28.480 | **8.116** | **36.596** |
|  |  | lif | 3.858 | 0.162 | 0.704 | 0.000 | 0.000 | 0.145 | 0.100 | 42.866 | **4.969** | **47.834** |
|  |  | mc | 3.987 | 1.295 | 5.632 | 0.000 | 0.000 | 1.263 | 0.110 | 44.301 | **12.287** | **56.589** |
| 50 | 4×4 | delay | 2.563 | 0.118 | 0.512 | 4.769 | 0.000 | 0.031 | 0.213 | 28.480 | **8.206** | **36.687** |
|  |  | lif | 3.858 | 0.162 | 0.704 | 0.000 | 0.000 | 0.036 | 0.694 | 42.866 | **5.454** | **48.320** |
|  |  | mc | 3.987 | 1.295 | 5.632 | 0.000 | 0.000 | 0.316 | 0.772 | 44.301 | **12.002** | **56.303** |
| 50 | 8×8 | delay | 2.563 | 0.118 | 0.512 | 4.769 | 0.000 | 0.008 | 1.544 | 28.480 | **9.514** | **37.994** |
|  |  | lif | 3.858 | 0.162 | 0.704 | 0.000 | 0.000 | 0.009 | 5.028 | 42.866 | **9.761** | **52.627** |
|  |  | mc | 3.987 | 1.295 | 5.632 | 0.000 | 0.000 | 0.079 | 5.569 | 44.301 | **16.562** | **60.863** |
| 100 | 2×2 | delay | 3.179 | 0.236 | 1.024 | 8.777 | 0.000 | 0.161 | 0.061 | 35.323 | **13.438** | **48.761** |
|  |  | lif | 4.235 | 0.324 | 1.408 | 0.000 | 0.000 | 0.160 | 0.158 | 47.053 | **6.285** | **53.338** |
|  |  | mc | 4.716 | 2.591 | 11.264 | 0.000 | 0.000 | 1.556 | 0.178 | 52.399 | **20.304** | **72.703** |
| 100 | 4×4 | delay | 3.179 | 0.236 | 1.024 | 8.777 | 0.000 | 0.040 | 0.418 | 35.323 | **13.675** | **48.998** |
|  |  | lif | 4.235 | 0.324 | 1.408 | 0.000 | 0.000 | 0.040 | 1.080 | 47.053 | **7.086** | **54.140** |
|  |  | mc | 4.716 | 2.591 | 11.264 | 0.000 | 0.000 | 0.389 | 1.250 | 52.399 | **20.209** | **72.608** |
| 100 | 8×8 | delay | 3.179 | 0.236 | 1.024 | 8.777 | 0.000 | 0.010 | 3.087 | 35.323 | **16.313** | **51.636** |
|  |  | lif | 4.235 | 0.324 | 1.408 | 0.000 | 0.000 | 0.010 | 7.749 | 47.053 | **13.726** | **60.779** |
|  |  | mc | 4.716 | 2.591 | 11.264 | 0.000 | 0.000 | 0.097 | 9.018 | 52.399 | **27.685** | **80.084** |
| 140 | 2×2 | delay | 3.519 | 0.330 | 1.434 | 11.412 | 0.000 | 0.173 | 0.076 | 39.100 | **16.943** | **56.044** |
|  |  | lif | 4.937 | 0.453 | 1.971 | 0.000 | 0.000 | 0.187 | 0.221 | 54.860 | **7.770** | **62.631** |
|  |  | mc | 5.251 | 3.627 | 15.770 | 0.000 | 0.000 | 1.775 | 0.240 | 58.339 | **26.662** | **85.001** |
| 140 | 4×4 | delay | 3.519 | 0.330 | 1.434 | 11.412 | 0.000 | 0.043 | 0.521 | 39.100 | **17.259** | **56.359** |
|  |  | lif | 4.937 | 0.453 | 1.971 | 0.000 | 0.000 | 0.047 | 1.519 | 54.860 | **8.928** | **63.789** |
|  |  | mc | 5.251 | 3.627 | 15.770 | 0.000 | 0.000 | 0.444 | 1.661 | 58.339 | **26.752** | **85.091** |
| 140 | 8×8 | delay | 3.519 | 0.330 | 1.434 | 11.412 | 0.000 | 0.011 | 3.802 | 39.100 | **20.507** | **59.607** |
|  |  | lif | 4.937 | 0.453 | 1.971 | 0.000 | 0.000 | 0.012 | 10.941 | 54.860 | **18.315** | **73.175** |
|  |  | mc | 5.251 | 3.627 | 15.770 | 0.000 | 0.000 | 0.111 | 11.933 | 58.339 | **36.691** | **95.030** |

#### Delay table in SRAM

| T | Mesh | Model | synaptic | neuron | state | window read | table read | leakage | comm | w_read | **total (W resident)** | **total (W in SRAM)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 2×2 | delay | 2.563 | 0.118 | 0.512 | 4.769 | 4.769 | 1.803 | 0.031 | 28.480 | **14.566** | **43.046** |
|  |  | lif | 3.858 | 0.162 | 0.704 | 0.000 | 0.000 | 0.145 | 0.100 | 42.866 | **4.969** | **47.834** |
|  |  | mc | 3.987 | 1.295 | 5.632 | 0.000 | 0.000 | 1.263 | 0.110 | 44.301 | **12.287** | **56.589** |
| 50 | 4×4 | delay | 2.563 | 0.118 | 0.512 | 4.769 | 4.769 | 0.451 | 0.213 | 28.480 | **13.396** | **41.876** |
|  |  | lif | 3.858 | 0.162 | 0.704 | 0.000 | 0.000 | 0.036 | 0.694 | 42.866 | **5.454** | **48.320** |
|  |  | mc | 3.987 | 1.295 | 5.632 | 0.000 | 0.000 | 0.316 | 0.772 | 44.301 | **12.002** | **56.303** |
| 50 | 8×8 | delay | 2.563 | 0.118 | 0.512 | 4.769 | 4.769 | 0.113 | 1.544 | 28.480 | **14.388** | **42.868** |
|  |  | lif | 3.858 | 0.162 | 0.704 | 0.000 | 0.000 | 0.009 | 5.028 | 42.866 | **9.761** | **52.627** |
|  |  | mc | 3.987 | 1.295 | 5.632 | 0.000 | 0.000 | 0.079 | 5.569 | 44.301 | **16.562** | **60.863** |
| 100 | 2×2 | delay | 3.179 | 0.236 | 1.024 | 8.777 | 8.777 | 2.257 | 0.061 | 35.323 | **24.311** | **59.634** |
|  |  | lif | 4.235 | 0.324 | 1.408 | 0.000 | 0.000 | 0.160 | 0.158 | 47.053 | **6.285** | **53.338** |
|  |  | mc | 4.716 | 2.591 | 11.264 | 0.000 | 0.000 | 1.556 | 0.178 | 52.399 | **20.304** | **72.703** |
| 100 | 4×4 | delay | 3.179 | 0.236 | 1.024 | 8.777 | 8.777 | 0.564 | 0.418 | 35.323 | **22.976** | **58.299** |
|  |  | lif | 4.235 | 0.324 | 1.408 | 0.000 | 0.000 | 0.040 | 1.080 | 47.053 | **7.086** | **54.140** |
|  |  | mc | 4.716 | 2.591 | 11.264 | 0.000 | 0.000 | 0.389 | 1.250 | 52.399 | **20.209** | **72.608** |
| 100 | 8×8 | delay | 3.179 | 0.236 | 1.024 | 8.777 | 8.777 | 0.141 | 3.087 | 35.323 | **25.222** | **60.545** |
|  |  | lif | 4.235 | 0.324 | 1.408 | 0.000 | 0.000 | 0.010 | 7.749 | 47.053 | **13.726** | **60.779** |
|  |  | mc | 4.716 | 2.591 | 11.264 | 0.000 | 0.000 | 0.097 | 9.018 | 52.399 | **27.685** | **80.084** |
| 140 | 2×2 | delay | 3.519 | 0.330 | 1.434 | 11.412 | 11.412 | 2.502 | 0.076 | 39.100 | **30.684** | **69.785** |
|  |  | lif | 4.937 | 0.453 | 1.971 | 0.000 | 0.000 | 0.187 | 0.221 | 54.860 | **7.770** | **62.631** |
|  |  | mc | 5.251 | 3.627 | 15.770 | 0.000 | 0.000 | 1.775 | 0.240 | 58.339 | **26.662** | **85.001** |
| 140 | 4×4 | delay | 3.519 | 0.330 | 1.434 | 11.412 | 11.412 | 0.625 | 0.521 | 39.100 | **29.253** | **68.353** |
|  |  | lif | 4.937 | 0.453 | 1.971 | 0.000 | 0.000 | 0.047 | 1.519 | 54.860 | **8.928** | **63.789** |
|  |  | mc | 5.251 | 3.627 | 15.770 | 0.000 | 0.000 | 0.444 | 1.661 | 58.339 | **26.752** | **85.091** |
| 140 | 8×8 | delay | 3.519 | 0.330 | 1.434 | 11.412 | 11.412 | 0.156 | 3.802 | 39.100 | **32.064** | **71.165** |
|  |  | lif | 4.937 | 0.453 | 1.971 | 0.000 | 0.000 | 0.012 | 10.941 | 54.860 | **18.315** | **73.175** |
|  |  | mc | 5.251 | 3.627 | 15.770 | 0.000 | 0.000 | 0.111 | 11.933 | 58.339 | **36.691** | **95.030** |

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

**Four panels, not two regimes.** The weights and the delay table can each be resident or in
SRAM, so there are four placements; every column shown is what enters the total, which is why
the columns add up to 	otal in each panel. §6.3 reports the criteria per placement.

**`synaptic` = input-layer SOP × E_AC + hidden SOP × E_AC + neuron ops × E_MAC.**

**`delay access` is non-zero only for the delay model** (LIF and MC have no table) and only
when the table is in SRAM. **`w_read` is non-zero only when the weights are in SRAM**; the
weight bits and their leakage still exist when resident (§5.3), they are reported apart.

## 6. Criteria results

### 6.1 (a) Temporal-state share ≥ 10%

**Three weight configurations per table**: resident, in SRAM with reuse $R=1$, and in SRAM
with $R=4$. **$R$ divides the weight read only.** The delay access — both reads — is unchanged
by $R$.

**(a) — delay table resident**

| T | Mesh | delay (W res) | delay (W R=1) | delay (W R=4) | LIF (W res) | LIF (W R=1) | LIF (W R=4) | MC (W res) | MC (W R=1) | MC (W R=4) | (a) W res | (a) W R=1 | (a) W R=4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 2×2 | 66.6% | 14.8% | 35.5% | 17.1% | 1.8% | 3.2% | 56.1% | 12.2% | 19.6% | ✅ | ✅ | ✅ |
| 50 | 4×4 | 64.7% | 14.5% | 34.7% | 13.6% | 1.5% | 2.7% | 49.6% | 10.6% | 17.0% | ✅ | ✅ | ✅ |
| 50 | 8×8 | 55.6% | 13.9% | 31.8% | 7.3% | 1.4% | 2.3% | 34.5% | 9.4% | 14.5% | ✅ | ✅ | ✅ |
| 100 | 2×2 | 74.1% | 20.4% | 44.7% | 25.0% | 2.9% | 5.8% | 63.1% | 17.6% | 27.7% | ✅ | ✅ | ✅ |
| 100 | 4×4 | 72.0% | 20.1% | 43.7% | 20.4% | 2.7% | 5.2% | 57.7% | 16.0% | 25.3% | ✅ | ✅ | ✅ |
| 100 | 8×8 | 60.1% | 19.0% | 39.0% | 10.3% | 2.3% | 4.1% | 41.0% | 14.2% | 21.2% | ✅ | ✅ | ✅ |
| 140 | 2×2 | 76.8% | 23.2% | 48.7% | 27.8% | 3.4% | 6.5% | 65.8% | 20.6% | 31.5% | ✅ | ✅ | ✅ |
| 140 | 4×4 | 74.7% | 22.9% | 47.7% | 22.6% | 3.2% | 5.9% | 60.6% | 19.1% | 29.1% | ✅ | ✅ | ✅ |
| 140 | 8×8 | 62.7% | 21.6% | 42.5% | 10.8% | 2.7% | 4.5% | 43.3% | 16.7% | 24.2% | ✅ | ✅ | ✅ |

**(a) — delay table in SRAM**

| T | Mesh | delay (W res) | delay (W R=1) | delay (W R=4) | LIF (W res) | LIF (W R=1) | LIF (W R=4) | MC (W res) | MC (W R=1) | MC (W R=4) | (a) W res | (a) W R=1 | (a) W R=4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 2×2 | 81.4% | 27.5% | 54.7% | 17.1% | 1.8% | 3.2% | 56.1% | 12.2% | 19.6% | ✅ | ✅ | ✅ |
| 50 | 4×4 | 78.4% | 25.1% | 51.2% | 13.6% | 1.5% | 2.7% | 49.6% | 10.6% | 17.0% | ✅ | ✅ | ✅ |
| 50 | 8×8 | 70.6% | 23.7% | 47.3% | 7.3% | 1.4% | 2.3% | 34.5% | 9.4% | 14.5% | ✅ | ✅ | ✅ |
| 100 | 2×2 | 85.7% | 34.9% | 62.9% | 25.0% | 2.9% | 5.8% | 63.1% | 17.6% | 27.7% | ✅ | ✅ | ✅ |
| 100 | 4×4 | 83.3% | 32.8% | 60.2% | 20.4% | 2.7% | 5.2% | 57.7% | 16.0% | 25.3% | ✅ | ✅ | ✅ |
| 100 | 8×8 | 74.2% | 30.9% | 55.0% | 10.3% | 2.3% | 4.1% | 41.0% | 14.2% | 21.2% | ✅ | ✅ | ✅ |
| 140 | 2×2 | 87.2% | 38.3% | 66.1% | 27.8% | 3.4% | 6.5% | 65.8% | 20.6% | 31.5% | ✅ | ✅ | ✅ |
| 140 | 4×4 | 85.1% | 36.4% | 63.8% | 22.6% | 3.2% | 5.9% | 60.6% | 19.1% | 29.1% | ✅ | ✅ | ✅ |
| 140 | 8×8 | 76.1% | 34.3% | 58.4% | 10.8% | 2.7% | 4.5% | 43.3% | 16.7% | 24.2% | ✅ | ✅ | ✅ |

### 6.2 (b) Ranking flips

**Three weight configurations per table.** The weight read is proportional to SOP — the one
quantity an SOP-only estimate already sees — so raising $R$ shrinks it and lets the energy
order depart from the SOP order.

**(b) — delay table resident**

| T | Mesh | energy order (W res) | energy order (W R=1) | energy order (W R=4) | SOP ≠ E (W res) | SOP ≠ E (W R=1) | SOP ≠ E (W R=4) |
|---|---|---|---|---|---|---|---|
| 50 | 2×2 | LIF < DL < MC | DL < LIF < MC | DL < LIF < MC | ✅ | ❌ | ❌ |
| 50 | 4×4 | LIF < DL < MC | DL < LIF < MC | DL < LIF < MC | ✅ | ❌ | ❌ |
| 50 | 8×8 | DL < LIF < MC | DL < LIF < MC | DL < LIF < MC | ❌ | ❌ | ❌ |
| 100 | 2×2 | LIF < DL < MC | DL < LIF < MC | DL < LIF < MC | ✅ | ❌ | ❌ |
| 100 | 4×4 | LIF < DL < MC | DL < LIF < MC | DL < LIF < MC | ✅ | ❌ | ❌ |
| 100 | 8×8 | LIF < DL < MC | DL < LIF < MC | DL < LIF < MC | ✅ | ❌ | ❌ |
| 140 | 2×2 | LIF < DL < MC | DL < LIF < MC | DL < LIF < MC | ✅ | ❌ | ❌ |
| 140 | 4×4 | LIF < DL < MC | DL < LIF < MC | DL < LIF < MC | ✅ | ❌ | ❌ |
| 140 | 8×8 | LIF < DL < MC | DL < LIF < MC | DL < LIF < MC | ✅ | ❌ | ❌ |

**(b) — delay table in SRAM**

| T | Mesh | energy order (W res) | energy order (W R=1) | energy order (W R=4) | SOP ≠ E (W res) | SOP ≠ E (W R=1) | SOP ≠ E (W R=4) |
|---|---|---|---|---|---|---|---|
| 50 | 2×2 | LIF < MC < DL | DL < LIF < MC | DL < LIF < MC | ✅ | ❌ | ❌ |
| 50 | 4×4 | LIF < MC < DL | DL < LIF < MC | DL < LIF < MC | ✅ | ❌ | ❌ |
| 50 | 8×8 | LIF < DL < MC | DL < LIF < MC | DL < LIF < MC | ✅ | ❌ | ❌ |
| 100 | 2×2 | LIF < MC < DL | LIF < DL < MC | LIF < DL < MC | ✅ | ✅ | ✅ |
| 100 | 4×4 | LIF < MC < DL | LIF < DL < MC | LIF < DL < MC | ✅ | ✅ | ✅ |
| 100 | 8×8 | LIF < DL < MC | DL < LIF < MC | DL < LIF < MC | ✅ | ❌ | ❌ |
| 140 | 2×2 | LIF < MC < DL | LIF < DL < MC | LIF < DL < MC | ✅ | ✅ | ✅ |
| 140 | 4×4 | LIF < MC < DL | LIF < DL < MC | LIF < DL < MC | ✅ | ✅ | ✅ |
| 140 | 8×8 | LIF < DL < MC | DL < LIF < MC | DL < LIF < MC | ✅ | ❌ | ❌ |

**The window read is what makes (b) hold in the resident placement.** Resident removes
only the table read; the window read (8.78 µJ at T=100/2×2) is still paid, lifting the delay
model from 4.66 to **13.44 µJ** — above LIF's 6.28 µJ — so its energy is no longer
lowest and the SOP ranking disagrees.

**`W SRAM` removes the effect again.** The weight read is 35.32 µJ and scales
with SOP, inflating all three models alike, so the energy order falls back onto the SOP order.

**Only `W resident` keeps (b) alive: 8–9 of 9 in both delay-table placements.**

### 6.3 Summary

| Placement | (a) | (a) all three models | (b) SOP ≠ energy |
|---|---|---|---|
| **W resident, D resident** | 9/9 | 8/9 | **8/9** |
| **W resident, D in SRAM** | 9/9 | 8/9 | **9/9** |
| **W SRAM R=1, D resident** | 9/9 | 0/9 | **0/9** |
| **W SRAM R=1, D in SRAM** | 9/9 | 0/9 | **4/9** |
| **W SRAM R=4, D resident** | 9/9 | 0/9 | **0/9** |
| **W SRAM R=4, D in SRAM** | 9/9 | 0/9 | **4/9** |

**Criterion (a) holds in every placement now** (9/9 for all six), because the window
read puts real energy into the temporal state: the delay model's share is
74.1 % resident and 85.7 % with the table in SRAM at T=100/2×2.

**Criterion (b) depends on where the weights live, not where the delay table lives.**

| weights | D resident | D in SRAM |
|---|---|---|
| resident | **8/9** | **9/9** |
| SRAM R=1 | 0/9 | 4/9 |
| SRAM R=4 | 0/9 | 4/9 |

**The `a_all` column is the stricter reading** (all three models above 10 %), which only the two
`W resident` placements satisfy; the white paper's criterion asks for *at least one* model, so
`(a)` is the one to quote.

## 7. Analysis

### 7.1 With the delay table resident, the delay model is the cheapest of the three

**T=100, 2×2, weights resident, delay table resident:**

| Estimate | Order | Verdict on delay |
|---|---|---|
| **spike count** | delay(5.3k) < LIF(13.9k) < MC(15.7k) | cheapest |
| **SOP** | delay(3.53M) < LIF(4.71M) < MC(5.24M) | cheapest |
| **full model** | **delay(4.66)** < LIF(6.28) < MC(20.30) | cheapest |

**All three agree. In this placement the delay model is genuinely the cheapest**, because every
one of its cost terms is proportional to its low spike count, and the delay table itself
occupies no runtime storage.

**The delay access is what can change that.** Moving the table into SRAM adds a term that is
*not* proportional to SOP — it is proportional to the number of recurrent lookups — so the three
estimates can then disagree:

| Estimate | Order | Verdict on delay |
|---|---|---|
| **SOP** | delay(3.53M) < LIF(4.71M) < MC(5.24M) | cheapest |
| **full model, D in SRAM** | LIF(6.28) < **delay(15.53)** < MC(20.30) | **2nd** |

**The delay network climbs from 4.66 to 15.53 µJ** — a factor of
3.3 — on the same spike counts.

**So the answer to "does an SOP-only estimate mis-rank these models?" is conditional**: it does
not, while the delay table is resident; it does, once the table is accessed at run time. §6.2
maps this across all six placements.

### 7.2 The multi-compartment cost is visible only in the full model

**T=100, 2×2, weights resident, delay table resident:**

| | delay | LIF | MC |
|---|---|---|---|
| spikes | 5,301 | 13,930 | 15,672 |
| SOP | 3.53M | 4.71M | 5.24M |
| **state** | 1.024 | 1.408 | **11.264 (8.0×)** |
| **leakage (state + window)** | 0.161 | 0.160 | 1.556 |
| full model | **4.66** | 6.28 | **20.30** |

**MC emits 13 % more spikes than LIF and 3× more than delay, yet costs 3.2× LIF and 4.4× delay.**
The gap is the eight-state compartment: `state` alone is 11.26 µJ, 8× the single-state models,
because every hidden neuron updates eight real values every frame.

**Neither spike count nor SOP sees this.** MC has the *highest* SOP of the three but only by
11 % over LIF, while its energy is 223 % higher. **A sparsity-based estimate ranks MC second;
the full model puts it last by a wide margin.**

### 7.3 At large mesh, communication overtakes state

**Delay model, weights and delay table both resident:**

| mesh | synaptic | state | delay access | leakage | comm | total | comm share |
|---|---|---|---|---|---|---|---|
| 2×2 | 3.18 | 1.02 | 8.78 | 0.16 | 0.06 | **4.66** | 1% |
| 4×4 | 3.18 | 1.02 | 8.78 | 0.04 | 0.42 | **4.90** | 9% |
| 8×8 | 3.18 | 1.02 | 8.78 | 0.01 | 3.09 | **7.54** | 41% |

As the core count grows, `t_inf` shrinks (448 → 107 → 40 µs), so leakage falls with it, while
communication grows with the number of hops. **Communication goes from 2 % of the delay model
at 2×2 to 41 % at 8×8**, and the total first rises then falls.

**The mesh size therefore decides which cost term matters**, and it is the one design parameter
that flips the ranking between models (§6.2, the 8×8 rows).

### 7.4 Effect of time-step resolution

**Delay accuracy is flat across resolutions** — 84.54 / 84.30 / 83.81 % at T = 50 / 100 / 140,
within 0.7 points. The 65,536-entry table absorbs whatever resolution it is given.

| T | delay | LIF | MC | delay ts% (D res) |
|---|---|---|---|---|
| 50 | 3.35 | 4.97 | 12.29 | 19.0% |
| 100 | 4.66 | 6.28 | 20.30 | 25.4% |
| 140 | 5.53 | 7.77 | 26.66 | 29.0% |

**The delay model's cost grows with T** (3.35 → 4.66 → 5.53 µJ) because every term except
leakage scales with the number of frames. **Its temporal-state share grows too** (19 → 25 → 29 %)
because the spike window and the delay access both scale with T while `comm` does not.

**The earlier claim that delay "prefers finer resolution" does not survive the per-synapse
change.** At $h=352$ with per-neuron delays the model did improve with finer frames
(86.56 → 87.24 → 87.99). With a per-synapse table and $h=256$ the effect disappears.
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

#### Why it did not train — the paper predicts this

DelRec states the mechanism explicitly ([arXiv:2509.24852](https://arxiv.org/html/2509.24852v2), §A.4
*"Recurrent delays mitigate the risk of exploding gradients"*):

> "recurrent delays may mitigate gradient challenges by implementing **temporal skip connections**,
> improving gradient propagation during training"

and in Fig. 1B:

> "Delays in recurrent connections reduce the risks of exploding or vanishing gradients by **bridging
> distant time steps**."

With all delays zero the feedback path is one step long. That is exactly the configuration the paper
identifies as ill-conditioned; removing the delays removes the skip connections.

**This explains our runs; it does not prove the architecture unusable.** Recurrent SNNs without
delays have been trained on SHD above 50 % elsewhere. **What we report is that we could not train
this network to usable accuracy in the time budget**, not that the network cannot be trained.

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
| energy 2×2, resident | **3.20 µJ** | 4.66 µJ | **0.69×** |
| energy 4×4, resident | 3.70 | 4.90 | 0.76× |
| energy 8×8, resident | 7.55 | 7.54 | 1.00× |
| energy 2×2, weights in SRAM | 24.73 | 50.86 | 0.49× |
| temporal-state share (2×2) | 32.9 % | 23.7 % | — |

**With the delay table in SRAM the delay model's 15.53 µJ is dominated by 8.78 µJ of delay access, not by leakage** — the 6-bit table leaks only 2.26 µJ, because 393 kbit at 11.9 nW/bit over 448 µs is small compared with 877,744 SRAM lookups at 10 pJ each.
Removing the table removes that, and the whole network costs 3.20 µJ — but delivers 49.81 % instead
of 84.30 %.

$$\underbrace{3.20\ \mu J}_{\text{no-delay}}\ \xrightarrow{\ +1.46\ \mu J\ }\ \underbrace{4.66\ \mu J}_{\text{delay}}\qquad\Longleftrightarrow\qquad 49.81\%\ \to\ 84.30\%$$

**Roughly 0.04 µJ per accuracy point at 2×2.** With the delay table resident the two are almost
identical in cost (3.35 vs 3.20 µJ) and the delay model additionally delivers 34.5 points more
accuracy; with the table in SRAM it costs 1.5× as much. **That ratio — cost of delay storage versus
cost of accuracy — is what Study 1 measures.**

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

**Effect on this study**: $t_{inf}$ scales linearly, so every leakage term does too. At T=100/2×2 the temporal-state leakage of the delay model is 2.26 µJ with the table in SRAM and 0.16 µJ without — **2.9 % and 0.2 % of the 13.5–15.5 µJ total** — so (a) and (b) move by a fraction of a point. Weight leakage (29.6 µJ) is affected proportionally but is reported separately and excluded from both criteria.

### 8.5 Mixed process nodes

| Parameter | Node |
|---|---|
| $E_{AC}$ / $E_{MAC}$ / $E_{access}$ / $P_{leak}$ | **45 nm** |
| $E_{hop}$ | **14 nm** (Loihi) |

**Node scaling (14→45 nm, about 2.75×) would make $E_{hop}$ ≈ 9.6 pJ/hop, scaling the comm term up accordingly.**

### 8.6 The weight-read term assumes no reuse (§5.5)

**Every synaptic operation is charged one full SRAM access (10 pJ) to fetch its 32-bit weight. A real core reads an entire SRAM line and reuses it across several consecutive synaptic operations.**

**Effect**: with a reuse factor $R$, the term scales as $1/R$. At $R=4$ (a 128-bit line feeding four 32-bit operations) the T=100 delay model's weight read drops from 35.32 µJ to 8.83 µJ. **With the delay table in SRAM that moves the total from 50.86 µJ to 24.36 µJ and the temporal-state share from 23.7 % to 50.4 %; with the table resident it moves 39.98 → 13.49 µJ and 3.0 % → 8.8 %.** The resident-weight case corresponds to $R=\infty$. **§6.1 and §6.2 carry the R=4 rows for both.**

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
python scripts/study1_measure.py         --n -1 --save results/study1_partA_full.json
python scripts/build_six_combos.py                       # -> results/study1_six_combos.json
```

`--n -1` is the default and means all 2,264 test samples; `--n 10` is a fast smoke
test. Checkpoint paths are relative to `scripts/`, so the deliverable directory can
be moved anywhere. Two external paths come from environment variables:

```bash
export DELREC_ROOT=/content/DelRec     # default: D:\DelRec
export SHD_TEST_H5=/content/DelRec/Datasets/SHD/extract/shd_test.h5
```

```bash
# 3. Plot (~5 s), reads results/study1_six_combos.json
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

#### Spike trains (Part B item 3)

`checkpoints/spikes/` holds the spike trains of **all nine** checkpoints in one format, one
NPZ per checkpoint:

```
checkpoints/spikes/
├── spikes_delay_T{50,100,140}.npz
├── spikes_lif_T{50,100,140}.npz
├── spikes_mc_T{50,100,140}.npz
├── spikes_index.json      format description + event counts + file sizes
└── (all ten files)        233.5 M events, 138 MB total
```

**Four fields per file, exactly as requested:**

| field | dtype | meaning |
|---|---|---|
| `sample` | int16 | index into the 2,264-sample SHD test split |
| `time_step` | int16 | frame index, `0 .. T-1` |
| `layer` | int8 | **0** = hidden1, **1** = hidden2 |
| `neuron` | int16 | index within that layer (0–255 for delay, 0–351 for PMSN) |

**Only spiking events are stored** — each array holds one entry per spike, so the file is
sparse and compresses 12× from the dense triple. The checkpoint identity (model, resolution)
lives in the filename, not in a field, because the request lists exactly these four.

**Reading:**

```python
z = np.load("checkpoints/spikes/spikes_delay_T100.npz")
# z["sample"], z["time_step"], z["layer"], z["neuron"] -> equal-length int arrays
```

**`layer` widths differ by family**: delay's hidden layers are 256 wide, PMSN's 352.
`spikes_index.json` records the event count and size of each file.

**Regenerate with `python scripts/export_spikes.py`** (about 12 minutes; `--samples N` for a
subset, `--only delay|lif|mc` for one family).

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
| 10 | Weight reads assume **no reuse** ($R=1$) in §5.5 | $R=4$ rows are given in §6.1/§6.2 and cut the weight read 4×; the delay access is **not** divided by $R$ | §8.6, §6.1 |
| 11 | The delay and LIF networks come from **different code bases** (DelRec vs PMSN) | the delay-vs-LIF comparison is **not controlled**; the SOP gap may be architectural, so no causal claim from the §6.2 flips | follow-up Task 4 |
| **12** | **Per-synapse delays cost ~2 points of accuracy** | delay reaches 84.54 / 84.30 / 83.81 % against per-neuron delays' 86.56 / 87.24 / 87.99 % at $h=352$. The per-synapse table is what makes the network parameter-matched to PMSN, so the choice is deliberate — but it means the three models are **not** accuracy-matched, and the delay network is consistently the weakest | §1 |
| **13** | **Ring depth varies with resolution** | $L = \lceil 1 + \max d + (1+\sigma) \rceil + 1$ is read per checkpoint: **49 / 55 / 51** for T = 50 / 100 / 140 with $\sigma = 10$. A trained delay distribution sets it, so the buffer cost is not constant across resolutions | §5.3 |
| **14** | **Criterion (b) fails whenever the delay table is resident** | with no delay access the delay model's energy is proportional to its SOP, so the two rankings agree — 0/9 in all three resident placements | §6.2, §6.3 |
| **15** | **The delay values are counted as "delay storage"** | 393 kbit is charged to category 3 rather than to the weight column, per the advisor's correction; the alternative placement is one of the four panels in §5.5 | §4.2, §5.3 |
| **17** | **The 6-bit delay encoding is a design choice, not a measurement** | 43 trained values need 6 bits; storing them as FP32 would cost 5.3× more leakage and change (a) by ~15 points | §5.3 |
| **18** | **Spike trains are exported for all nine checkpoints** | 233.5 M events, 138 MB across nine NPZ files; `layer` is 0/1 and layer widths differ (256 vs 352) | §9.4 |

---

## 11. Figures

All six are generated by `scripts/plot_study1.py`, which reads
`results/study1_six_combos.json` directly — so the figures cannot drift from the tables.

| File | Question it answers |
|---|---|
| `figs/study1_main_2x2_allT_.png` | The white paper's requested form: stacked energy breakdown of the three models with spike counts overlaid, 2×2 mesh, all three resolutions |
| `figs/study1_grid_resolution_mesh_.png` | How resolution and mesh size change the energy composition — the full 3 × 3 sensitivity grid (3 resolutions × 3 meshes) |
| `figs/study1_ranking_flip_.png` | Ranking by the three estimates at each resolution (2×2), marking which pair agrees — the delay model is cheapest by spike count and SOP but second by full-model energy in this placement |

**Placement in the figures: weights resident, delay table in SRAM** — the only placement in which
the energy ranking can differ from the SOP ranking (§6.2). Pass `--combo w_res_d_res` for the
all-resident panel.

```bash
python scripts/plot_study1.py                        # all six figures
python scripts/plot_study1.py --variant recurrent    # three only
python scripts/plot_study1.py --combo w_res_d_res    # a different placement
```

**TASK 5 is discharged as follows.**

1. **2×2 is the main figure.** `study1_main_2x2_allT.png` carries all three time resolutions on four
   cores — the mesh is held fixed at the main configuration so the time axis varies.
   4×4 and 8×8 appear only in `study1_grid_resolution_mesh.png` (the complete 3 × 3 grid) and in the
   §5.5 table as sensitivity cases; no conclusion rests on them.
2. **Every assumption is in the caption.** The main figure carries a nine-point caption box below the
   plot, covering precision, the weight regime, the mesh choice, placement, the multicast bound, the
   parallelism factor, the leakage source, the delay-table encoding and placement, and how
   checkpoints were chosen. Each item points at the section that discusses it.
3. **Checkpoints are selected by validation accuracy.** `experiments/train.py` saves `best.pth` when
   `val_acc >= best_val_acc` (L211-213) and then evaluates that checkpoint on the test split (L220-221,
   commented *"final test on the best validation checkpoint"*). **No checkpoint anywhere in this report
   was chosen by test accuracy.**

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
