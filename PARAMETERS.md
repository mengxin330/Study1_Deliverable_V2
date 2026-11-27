# Parameters used in Study 1

Every quantity the energy model needs, with its value, its label, and where it comes from.
The four labels are defined in README §4.5:

- **measured** — produced by running the models
- **characterized** — taken from cited literature
- **projected** — computed from the two above
- **simulated** — produced by a cycle-level NoC model (none in Study 1)

---

## 1. Unit prices (characterized)

| Symbol | Value | Unit | Source | Used by |
|---|---|---|---|---|
| $E_{AC}$ | 0.9 | pJ / AC operation | Horowitz, *Computing's Energy Problem*, ISSCC 2014, Fig. 1.1 (45 nm) | synaptic term |
| $E_{MAC}$ | 4.6 | pJ / MAC operation | Horowitz 2014, Fig. 1.1 (45 nm FP32) | neuron-update term |
| $E_{ACC}$ | 10.0 | pJ / SRAM access | Horowitz 2014, Fig. 1.1 (32-bit, 8 KB–1 MB range) | weight read, neuron-state access, delay access |
| $P_{leak}$ | 11.9 | nW / bit | derived from an 8 MB SRAM at ~0.8 W (see §4.4) | all leakage terms |
| $E_{hop}$ | 3.5 | pJ / flit-hop | Loihi (Davies et al. 2018) Table 2, scaled to 45 nm | communication term |
| $F$ | 2.0 | GHz | assumption, no literature basis (§8.4) | $t_{inf}$ |
| par | 1 | — | assumption: one core's worth of parallelism (§8.4) | $t_{inf}$ |

**These are the only characterized numbers in the report.** Everything else is either measured
or projected from a measured count and one of these prices.

---

## 2. Architecture (measured from the checkpoints)

| Model | parameters | neurons with state | $d_{state}$ | delay values | window width | hidden width |
|---|---|---|---|---|---|---|
| delay | 239,124 | **512** | 1 | **65,536** | 256 | 256 |
| LIF | 187,988 | 704 | 1 | 0 | 0 | 352 |
| MC | 200,660 | 704 | 8 | 0 | 0 | 352 |

**Why `neurons = 512` for the delay model, not 532:** the network is
`Linear(140,256) → LIF(256) → Linear(256,256) → LIF(256) → Linear(256,20)`. The 20 output
units are a plain `Linear` layer with no membrane potential, so they carry no state and
contribute nothing to `state` or to the state-memory bit count.

**Why the delay table is 6 bits, not 32:** the trained delays take **43 integer values
(0–42)**, so 6 bits represent them exactly. See README §5.3.

---

## 3. Fanout (measured, used to turn spike counts into SOP)

| Model | input → hidden1 | hidden1 → hidden2 | hidden2 → out | hidden2 → hidden2 |
|---|---|---|---|---|
| delay | 256 | 256 | 20 | **256** |
| LIF / MC | 352 | 352 | 20 | — (feedforward) |

**Delay's hidden2 fanout is 276 = 256 recurrent + 20 output.** The recurrent part is the only
one that consults the delay table, so `delay_access` counts $h_2^{spikes}\times256$, not
`SOP` (§4.2, §8.7).

---

## 4. Spike-window depth (measured from the checkpoints)

$$L = \lceil 1 + \max d + (1+\sigma) \rceil + 1$$

| T | $\max d$ | $\sigma$ | $L$ | window bits |
|---|---|---|---|---|
| 50 | — | 10 | **49** | 256 × 48 = 12,288 |
| 100 | 42 | 10 | **55** | 256 × 54 = 13,824 |
| 140 | — | 10 | **51** | 256 × 50 = 12,800 |

---

## 5. Accuracy (measured, SHD test split, 2,264 samples)

| Model | T = 50 | T = 100 | T = 140 |
|---|---|---|---|
| delay | 84.54 % | 84.30 % | 83.81 % |
| LIF | 88.96 % | 86.40 % | 87.10 % |
| MC | 91.21 % | 88.30 % | 88.96 % |

**Delay-free reference** (TASK 4): 49.81 % (cos schedule) / 49.21 % (one-cycle). We could not
train it to usable accuracy in the time budget (README §7.5).

---

## 6. Raw measured counts (measured)

Averaged over all 2,264 test samples:

| Model | T | spikes | SOP | flit-hops 2×2 | flit-hops 4×4 | flit-hops 8×8 |
|---|---|---|---|---|---|---|
| delay | 50 | 2,750 | 2,848,026 | 8,832 | 60,962 | 441,018 |
| delay | 100 | 5,301 | 3,532,306 | 17,374 | 119,507 | 882,011 |
| delay | 140 | 6,696 | 3,910,030 | 21,762 | 148,861 | 1,086,171 |
| LIF | 50 | 8,688 | 4,286,554 | 28,587 | 198,392 | 1,436,650 |
| LIF | 100 | 13,930 | 4,705,348 | 45,118 | 308,479 | 2,213,959 |
| LIF | 140 | 19,523 | 5,486,040 | 63,203 | 434,092 | 3,125,954 |
| MC | 50 | 9,565 | 4,430,112 | 31,381 | 220,566 | 1,591,048 |
| MC | 100 | 15,672 | 5,239,870 | 50,835 | 357,068 | 2,576,434 |
| MC | 140 | 21,026 | 5,833,939 | 68,429 | 474,574 | 3,409,407 |

**Delay's hidden2 spikes** (the delay-access count before multiplying by 256):

| T | $h_2$ spikes | $h_2 \times 256$ |
|---|---|---|
| 50 | 1,863 | 476,944 |
| 100 | 3,429 | 877,744 |
| 140 | 4,458 | 1,141,202 |

---

## 7. Bit counts (projected)

| Model | weights | neuron state | delay table | spike window | **temporal total** |
|---|---|---|---|---|---|
| delay | 5,554,816 | 16,384 | **393,216** | 13,824 | **423,424** |
| LIF | 6,015,616 | 22,528 | 0 | 0 | 22,528 |
| MC | 6,421,120 | **180,224** | 0 | 0 | 180,224 |

Weight bits = (parameters − delay values) × 32, so the delay table is not counted twice
(README §5.3).

---

## 8. Time model (projected)

$$t_{inf} = \frac{\text{SOP} + N_{neur}\,T\,d_{state}}{n_{cores}\cdot F\cdot \text{par}}$$

| T | 2×2 (4 cores) | 4×4 (16) | 8×8 (64) |
|---|---|---|---|
| 50 | 359 µs | 90 µs | 22 µs |
| 100 | 448 µs | 112 µs | 28 µs |
| 140 | 498 µs | 124 µs | 31 µs |

(delay model, weights resident; LIF and MC are slightly higher because their SOP is larger.)

**Leakage** = bits × $P_{leak}$ × $t_{inf}$, so it scales linearly with $t_{inf}$ and therefore
falls as the core count rises.

---

## 9. Mesh and placement (projected)

| Config | Cores | Mesh | Placement |
|---|---|---|---|
| 2×2 | 4 | 2 × 2 | sequential, `core(i) = floor(i·n/512)` |
| 4×4 | 16 | 4 × 4 | same |
| 8×8 | 64 | 8 × 8 | same |

**Sequential placement, not optimised** (README §8.1). Communication is a **unicast upper
bound**: same-timestep spikes are not merged, and multicast is not exploited (§8.2, §8.3).

---

## 10. What is *not* modelled

| Item | Why |
|---|---|
| $\int P_{idle}\,dt$ | outside the white paper's Study 1 five categories |
| clock tree, data conversion, feature extraction | same |
| weight leakage **in the totals** | reported separately (§4.3) |
| low-precision (INT8/INT4) regimes | the white paper's §7.1 regime is not modelled (§10) |

---

## Reproducing

```bash
python scripts/study1_measure.py        --n -1 --save results/study1_partA_full.json
python scripts/study1_measure_allsop.py --n -1 --save results/study1_allsop_full.json
python scripts/build_six_combos.py
python scripts/plot_study1.py
python scripts/export_spikes.py
```

**Every number in README §5.2–§6.3 comes from these five commands.** No value is typed in by
hand.
