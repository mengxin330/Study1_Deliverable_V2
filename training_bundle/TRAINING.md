# Training Reproduction Guide

Three time resolutions × two code bases = **9 training runs**: 3 for the delay
network (one per resolution) and 6 for PMSN (3 resolutions × 2 variants).

```bash
# Directory convention
/content/DelRec          # alexmaxad/DelRec
/content/PMSN            # xychen-comp/Parallel-Multi-compartment-Spiking-Neuron-PMSN
/content/shd             # {shd_train.h5, shd_test.h5}
/content/work/exp        # outputs
```

---

## 0. Quick start — the training bundle

The nine runs are driven by `training_bundle/`. **Every file in it is already the
final version, so there is no patching step for either code base.**

```
bundle/
├── delrec/
│   ├── networks_shd.py           per-synapse model class (shared by all T)
│   ├── utils.py                  temporal pooling max->mean (shared)
│   ├── config_T{50,100,140}.py   one per resolution
│   └── loaders_T{50,100,140}.py  fixed_collate_T{T}
├── pmsn/
│   ├── SHD_PMSN.py               pre-patched: bn=True + fixed-duration binning
│   ├── SHD_PMSN.diff             what changed (+10 / −7 lines, one file)
│   ├── PMSN.py                   unmodified upstream
│   └── lib.py                    unmodified upstream
└── runs/
    ├── 01_delay_T50.sh           01..03  delay network
    ├── 02_delay_T100.sh
    ├── 03_delay_T140.sh
    ├── 04_pmsn_lif_T50.sh        04..09  PMSN
    ├── 05_pmsn_mc_T50.sh
    ├── 06_pmsn_lif_T100.sh
    ├── 07_pmsn_mc_T100.sh
    ├── 08_pmsn_lif_T140.sh
    └── 09_pmsn_mc_T140.sh
```

After the environment is set up (§1):

```bash
unzip -q /content/bundle.zip -d /content
bash /content/bundle/runs/02_delay_T100.sh      # delay, T=100
bash /content/bundle/runs/06_pmsn_lif_T100.sh   # PMSN LIF, T=100
```

Each script copies its files into place, then trains. §3 and §4 explain what each
group of scripts does; §6 explains why whole files are shipped instead of patches.

---

## 1. Environment

```bash
# --- DelRec ---
cd /content
git clone https://github.com/alexmaxad/DelRec.git
cd DelRec && pip install -r requirements.txt
mkdir -p Datasets/SHD

# requirements.txt pulls numpy 2.x, which breaks spikingjelly's compiled extension:
#   ValueError: numpy.dtype size changed ... Expected 96 from C header, got 88
# Pin numpy < 2 AND restart the session -- the .so binds the ABI at process start.
pip install -q --force-reinstall "numpy==1.26.4"
# >>> Runtime -> Restart session <<<

# --- PMSN ---
cd /content
git clone https://github.com/xychen-comp/Parallel-Multi-compartment-Spiking-Neuron-PMSN.git PMSN
pip install -q einops tqdm h5py
pip install -q "spikingjelly @ git+https://github.com/fangwei123456/spikingjelly.git@d4fee3a1715bf42ce15f52fde1b1d4709d7ea25e"

# --- SHD ---
mkdir -p /content/shd && cd /content/shd
wget -q https://zenkelab.org/datasets/shd_train.h5.zip
wget -q https://zenkelab.org/datasets/shd_test.h5.zip
unzip -oq shd_train.h5.zip && unzip -oq shd_test.h5.zip
```

### 1.1 Three attributes missing from `perf_SHD.py`

`loaders.py` reads three attributes that the SHD config never defines:

```python
# append inside the Config class in configs/perf_SHD.py
mask_T_stepsi = None
mask_T_steps  = 0
masked_proba  = None
```

Without them: `AttributeError: 'Config' object has no attribute 'mask_T_stepsi'`.

---

## 2. Hyperparameters — values actually used

**These are the values recorded in each run's `config.json`.** The three
resolutions were not trained with one identical set; the differences are listed
rather than smoothed over.

| Parameter | T=50 | T=100 | T=140 |
|---|---|---|---|
| `time_step` | 28 | 14 | 10 |
| `max_feedforward_delay` | 9 | 17 | 25 |
| `batch_size` | 40 | 40 | 40 |
| `hidden_layers` | [256, 256] | [256, 256] | [256, 256] |
| `epochs` | 10 | 10 | 10 |
| `lr_w` | 5e-3 | 5e-3 | 5e-3 |
| `lr_positions` | 5e-2 | 5e-2 | 5e-2 |
| `sigma_init` | 10.0 | 10.0 | 10.0 |
| `init_rec_delay` | uniform | uniform | uniform |
| `delay_std_init` | 15 | 15 | 15 |
| `init_recdel_offset` | 10 | 10 | 10 |
| `max_rec_delay` | 30 | 30 | 30 |
| `tau` | 1.05 | 1.05 | 1.05 |
| `seed` | 1111 | 1111 | 1111 |

**Only the first two rows differ across resolutions.** From `batch_size` down
every value is identical, and all three runs share seed 1111.

**`batch_size = 40` at all three resolutions** — a Colab T4 cannot hold 256 at
T=100 together with the `(N, N)` per-synapse delay tensor, and 40 is used
everywhere for consistency.

**`max_feedforward_delay`** is `250 ms / time_step` rounded (9 / 17 / 25). It has
**no effect on these networks**: there is no `dcls` layer, so the parameter is
never read. It is listed only so the config round-trips.

**`max_rec_delay = 30` at all three resolutions.** The upstream recipe derives it
from a fixed 300 ms budget (`300 / time_step` → 11 / 21 / 30). This study fixes it
at **30 frames** for all three runs, so the delay *range* is a constant number of
frames instead of a constant physical time. That keeps the number of delay slots
comparable across resolutions.

---

## 3. Delay network (DelRec, per-synapse)

### 3.1 The per-synapse model does not exist upstream

`networks_shd.py` ships `SNN_recurrent_delays` (per-neuron) only. The per-synapse
variant is the vendored file `bundle/delrec/networks_shd.py` (source
`alexmaxad/DelRec @ dfce0992`, **+66 / −1 lines**).

The 66 added lines are one class, `SNN_synaptic_recurrent_delays`, which is
`SNN_recurrent_delays` with a single substitution — `axonal_recdel` →
`synaptic_recdel` — so `layers.7.recurrent_delays` has shape `(N, N)` instead of
`(N,)`. The exact diff is in `patches/vendor/networks_shd.diff`.

### 3.2 Training

Each run script copies four files, then trains. Nothing is patched:

```bash
cd /content/DelRec

# --- T=100 as the example; the other two differ only in the file names ---
cp /content/bundle/delrec/config_T100.py   configs/perf_SHD.py
cp /content/bundle/delrec/loaders_T100.py  src/delrec/datasets/loaders.py
cp /content/bundle/delrec/utils.py         src/delrec/utils.py
cp /content/bundle/delrec/networks_shd.py  src/delrec/networks_shd.py

python experiments/train.py --dataset shd --model SNN_synaptic_recurrent_delays \
  --seeds 1111 --epochs 10 --kernel v2 --synaptic-kernel v2 \
  --out /content/work/exp/synaptic_h256_T100
```

For T=50 and T=140 replace every `T100` above with `T50` / `T140`
(`runs/01_delay_T50.sh` and `runs/03_delay_T140.sh` do exactly this).

**Two sanity checks before the first epoch:**

```
recurrent kernels: axonal=v2, synaptic=v2
Total Trainable Params: 239124
```

`synaptic=default` means `--synaptic-kernel v2` did not take effect — that is the
Triton path and it crashes mid-run (§3.3).

### 3.3 `--kernel v2` and `--synaptic-kernel v2` are two different flags

`--kernel` selects the kernel for `axonal_recdel`; `--synaptic-kernel` selects the
one for `synaptic_recdel`. **Omit the second and it falls back to
`SYNAPTIC_CUDA_DEFAULT = 'eventdriven'`** (`delay_layers.py:83`), a Triton kernel:

| Omission | Symptom |
|---|---|
| `--kernel v2` on a wide net | `triton.runtime.errors.OutOfResources: Required: 135168, limit 65536` |
| `--synaptic-kernel v2` | `LLVM ERROR: pthread_join failed: Invalid argument`, then `RuntimeError: DataLoader worker is killed by signal: Aborted` — observed at T=140, epoch 5 |

**A correct run prints:** `recurrent kernels: axonal=v2, synaptic=v2`

### 3.4 Expected parameter count

The run prints `Total Trainable Params: 239124`:

```
140·256 + 256 (bias)  = 36,096      layers.0
256·2 (BatchNorm)     =    512      layers.1
256·256 + 256 (bias)  = 65,792      layers.5
256·2 (BatchNorm)     =    512      layers.6
256·256               = 65,536      layers.7.recurrent_weights
256·256               = 65,536      layers.7.recurrent_delays   <- the delays
256·20 + 20 (bias)    =  5,140      layers.10
                        -------
                        239,124    of which 65,536 are delays
```

### 3.5 Binning cache

`Datasets/SHD/duration_{time_step}/` is written once and reused. Deleting it forces
re-extraction from the `.h5` files (≈75 s per resolution).

---

## 4. PMSN networks

### 4.1 The pre-patched sources are shipped in the bundle

PMSN needs two changes and no more:

| Change | In | What breaks without it |
|---|---|---|
| `np.linspace(...)` equal-division binning → fixed `duration` binning | `SHD_PMSN.py` | Its "one frame" would not match DelRec's, so the two families cannot be compared at the same T |
| `bn=False` → `bn=True` | `SHD_PMSN.py` | Checkpoints were trained with BN; a mismatch raises in `load_state_dict` |

**Both live in `training_bundle/pmsn/SHD_PMSN.py`**, ready to copy — no patch step:

```bash
cp /content/bundle/pmsn/SHD_PMSN.py /content/PMSN/SHD/
cp /content/bundle/pmsn/PMSN.py     /content/PMSN/SHD/
cp /content/bundle/pmsn/lib.py      /content/PMSN/SHD/
```

`PMSN.py` and `lib.py` are unmodified upstream files — they are shipped alongside so
the three-file set is self-contained. The diff is in `pmsn/SHD_PMSN.diff`
(+10 / −7 lines, one file).

**The change is T-independent**: it replaces the bin edges with
`self.time_step_ms = 1400.0 / nb_steps`, evaluated at run time. So one copy of the
file serves all three resolutions.

### 4.2 Training

```bash
# --- T=100: LIF (d_state=1) 5 epochs -> 86.40%
#            multi-compartment (d_state=4) 3 epochs -> 88.30%
cd /content/PMSN/SHD
python SHD_PMSN.py --data_path /content/shd --T 100 --d_state 1 --epochs 5 \
  --ckpt_path /content/work/exp/pmsn_lif_T100_fill_5ep --device cuda:0
python SHD_PMSN.py --data_path /content/shd --T 100 --d_state 4 --epochs 3 \
  --ckpt_path /content/work/exp/pmsn_mc_T100_fill_3ep --device cuda:0
```

```bash
# --- T=50: LIF 88.96%, MC 91.21% ---
python SHD_PMSN.py --data_path /content/shd --T 50 --d_state 1 --epochs 5 \
  --ckpt_path /content/work/exp/pmsn_lif_T50_fill_5ep --device cuda:0
python SHD_PMSN.py --data_path /content/shd --T 50 --d_state 4 --epochs 3 \
  --ckpt_path /content/work/exp/pmsn_mc_T50_fill_3ep --device cuda:0
```

```bash
# --- T=140: LIF 87.10%, MC 88.96% ---
python SHD_PMSN.py --data_path /content/shd --T 140 --d_state 1 --epochs 5 \
  --ckpt_path /content/work/exp/pmsn_lif_T140_fill_5ep --device cuda:0
python SHD_PMSN.py --data_path /content/shd --T 140 --d_state 4 --epochs 3 \
  --ckpt_path /content/work/exp/pmsn_mc_T140_fill_3ep --device cuda:0
```

**`--d_state 4` is four *complex* compartments = 8 real state values.** The energy
model charges `d_state = 8` per neuron per timestep; this flag is what produces it.

**`--T` is the frame cap (`nb_steps`)**: frames beyond it are cropped.

**PMSN hyperparameters are independent of the delay network's** — `d_model=352`,
`lr=0.01`, `nlr=0.001`, `wd=0`, `dropout=0.2`. Only the delay network's settings
appear in §2.

---

## 5. Epoch budget

| Model | epochs | Flag | Checkpoint selection |
|---|---|---|---|
| **delay (h=256)** | **10** | `--epochs 10` | **best validation** |
| **PMSN LIF** | 5 | `--epochs 5` | last epoch |
| **PMSN MC** | 3 | `--epochs 3` | last epoch |

The two code bases carry different default budgets. The delay network needs the
longer schedule because the delay parameters start from a broad prior and only
begin to move after the weights settle.

**Why PMSN reports the last epoch and not the best one**: MC peaks at epoch 1
(93.11 / 89.62 / 89.13 %) and drops back at epoch 2 (91.21 / 88.30 / 88.96 %).
Taking its best checkpoint would widen its lead over LIF to 3–4 points, so the
last epoch is reported to keep the three models in one accuracy band. Measured
accuracies are in the main README.

## 6. Why whole files, not patches

Every modification is shipped as a complete file rather than as a text patch.
Two reasons:

- **`networks_shd.py` / `SHD_PMSN.py`** — a text-anchor patch silently breaks when
  the surrounding code moves; shipping the whole file removes that failure mode.
  Both diffs are still included (`patches/vendor/networks_shd.diff`,
  `bundle/pmsn/SHD_PMSN.diff`) so the change is reviewable.
- **`config_T*.py` / `loaders_T*.py`** — these genuinely differ per resolution
  (different `time_step`, different `fixed_collate_T{T}` symbol), so three concrete
  files are clearer than one file plus a parameter.

**The cost is that the files are pinned to the upstream commits they came from**
(`DelRec @ dfce0992`, PMSN `main`). That is what makes the runs reproducible; if
upstream moves, regenerate them.

### 6.1 What each modification does

**DelRec** — four changes, all present in `bundle/delrec/`:

| Change | What breaks without it |
|---|---|
| Fixed-duration binning | With `frames_number` equal division, "one frame" covers a different physical interval per sample; measured accuracy drops from **87% to 70.40%** |
| Fixed-length collate (`fixed_collate_T{T}`) | Frame counts differ within a batch, so one delay value means different physical durations |
| max → mean temporal pooling | Mismatch with the official checkpoint's convention |
| `networks_shd.py` per-synapse class | The class does not exist upstream — training cannot start |

**PMSN** — two changes, both in `bundle/pmsn/SHD_PMSN.py`:

| Change | What breaks without it |
|---|---|
| Fixed-`duration` binning | Its "one frame" would not match DelRec's, so the two families cannot be compared at the same T |
| `bn=True` | Checkpoints were trained with BN; a mismatch raises in `load_state_dict` |

---

## 7. Known pitfalls

**7.1 `--log_interval` default of 300 in `SHD_PMSN.py`**

An epoch has only 204 batches, so intermediate progress never prints. **Do not
lower it below 50** — the official print format string has an argument-count bug
that raises `TypeError: not enough arguments for format string` when it fires.

**7.2 Stale module cache in notebooks**

After editing `configs/perf_SHD.py`, `importlib.import_module('configs.perf_SHD')`
returns the **cached** module. Use `importlib.reload(m)` or restart the kernel.
Symptom: the config keeps printing an old `time_step` / `batch_size`.

**7.3 `git checkout` silently reverts the mask attributes**

`patch_delrec.py` does not add `mask_T_stepsi` / `mask_T_steps` / `masked_proba`
(§1.1). Restoring `perf_SHD.py` with `git checkout` removes them, and the next run
fails with `AttributeError`. Re-add them after any `git checkout`.

**7.4 Regex edits swallow indentation**

Editing `perf_SHD.py` with `re.sub` on `^\s*key\s*=.*$` and a replacement that does
not re-emit `\g<1>` strips the 4-space indent and produces
`IndentationError: unexpected indent`. Preserve the captured group:

```python
s = re.sub(rf'^(\s*){key}\s*=.*$', rf'\g<1>{value}', s, flags=re.M)
```
