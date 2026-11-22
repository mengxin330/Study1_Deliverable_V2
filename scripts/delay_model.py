"""A ready-to-use DelRec SHD delay network.

Loads ``best.pth`` into a callable model and hides every non-obvious step that the
verification work turned up:

  * ``config.epochs`` must be the run's own epoch count (10), because
    ``update_sigma`` computes ``sigma_decay ** (100 / config.epochs)``;
  * ``sigma`` is a plain attribute and is not in the state dict, so the annealing
    schedule has to be replayed before every evaluation;
  * SHD does not set sigma to zero at test time (unlike SSC/HAR);
  * the readout is the *mean* over time of the last layer's membrane potential,
    followed by argmax -- not max, and no softmax over time.

Quick start
-----------
    from delay_model import DelayModel

    m = DelayModel()
    m.describe()

    x = torch.zeros(100, 1, 140)     # (T, batch, channels), 140 = 700 // n_bins
    logits = m(x)                    # (batch, 20)
    print(m.predict_proba(x).shape)  # (batch, 20)
    print(m.predict(x))              # (batch,)

Command line
------------
    python delay_model.py                      # load and self-report
    python delay_model.py --npy input.npy      # classify an array
    python delay_model.py --h5-index 0         # classify SHD test sample 0
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

import os

DEFAULT_REPO = os.environ.get("DELREC_ROOT", r"D:\DelRec")
DEFAULT_RUN = os.environ.get("DELAY_RUN_DIR", str(Path(DEFAULT_REPO).parent / "work" / "exp" / "delay_b40_mean_10ep"))
TRAINED_EPOCHS = 10          # the --epochs this checkpoint was trained with
N_CLASSES = 20


@dataclass
class MeasureResult:
    """The four quantities Study 1 asks to record, from one forward pass.

    ``spikes`` / ``spikes_per_t`` / ``fanout`` / ``sop*`` are measured directly.
    ``state_rw`` and ``delay_buffer`` are derived under the assumption stated in
    their own entries, not counted.

    Energy is deliberately NOT computed here. These are raw per-inference counts;
    multiplying them by unit costs is a separate step, so the cost table can change
    without touching this module.
    """

    T: int
    batch: int
    spikes: dict = field(default_factory=dict)         # layer -> total spikes
    spikes_per_t: dict = field(default_factory=dict)   # layer -> np.ndarray(T,)
    fanout: dict = field(default_factory=dict)         # layer -> MAC per spike
    sop_by_layer: dict = field(default_factory=dict)   # layer -> spikes * fanout
    # PMSN Table 6.9 and §4.3.5: "The multiplication in synaptic operations
    # between layers employs more efficient AC operations, while the neuronal
    # updates involving analog computation induce more energy-intensive MAC
    # operations." So the two counters are NOT priced alike:
    #     sop        x E_AC   (0.9 pJ at 45 nm)
    #     sop_neuron x E_MAC  (4.6 pJ at 45 nm)
    # Charging E_MAC for both overstates the synaptic term by 5.1x.
    sop: int = 0                                   # AC-counted: spikes x fanout
    sop_neuron: int = 0                            # MAC-counted: state updates
    ops_per_neuron: float = 1.0                    # MACs one neuron spends per step
    delay_buffer: dict = field(default_factory=dict)
    state_rw: dict = field(default_factory=dict)
    n_samples: int = 0                            # >0 only for measure_loader() results
    totals: dict = field(default_factory=dict)    # raw sums behind the per-inference values


    @property
    def sop_total(self) -> int:
        """Synaptic + neuronal operations -- the accounting PMSN uses."""
        return self.sop + self.sop_neuron

    def summary(self) -> str:
        if self.n_samples:
            head = (f"T={self.T}  per inference over {self.n_samples} samples "
                    f"(Study 1 asks for per-inference values)")
        else:
            head = f"T={self.T}  batch={self.batch}"
        lines = [head]
        for k in self.spikes:
            pt = self.spikes_per_t[k]
            lines.append(
                f"  {k:<22} spikes={self.spikes[k]:>9,}  "
                f"peak_t={int(pt.argmax()):>3}  fanout={self.fanout[k]:>4}  "
                f"sop={self.sop_by_layer[k]:>12,}")
        lines.append(f"  {'TOTAL synaptic':<22} spikes={sum(self.spikes.values()):>9,}  "
                     f"{'':>17}  sop={self.sop:>12,}   [x E_AC]")
        lines.append(f"  {'neuronal ops':<22} {self.sop_neuron:>9,}  "
                     f"({self.ops_per_neuron:g} MAC/neuron/step)   [x E_MAC]")
        lines.append(f"  {'SOP total':<22} {self.sop_total:>9,}")
        if self.delay_buffer:
            d = self.delay_buffer
            lines.append(f"  delay buffer: n={d['n']} slots={d['slots']} "
                         f"bits={d['bits']:,.0f} max_d={d['max_delay']:.0f} "
                         f"sigma={d['sigma']:.4f}")
        if self.state_rw:
            s = self.state_rw
            lines.append(f"  state rw ({s['assumption']}): "
                         f"reads={s['total_reads']:,} writes={s['total_writes']:,}")
        return "\n".join(lines)



class DelayModel:
    """The trained delay SNN, wrapped so a forward call just works.

    Parameters
    ----------
    run_dir : directory holding best.pth / config.json / train_res.csv.
    repo    : DelRec checkout root (must be the patched copy).
    device  : 'cpu' or 'cuda'.
    """

    def __init__(self, run_dir: str = DEFAULT_RUN, repo: str = DEFAULT_REPO,
                 device: str = "cpu", verbose: bool = True,
                 model_cls: str = "SNN_recurrent_delays"):
        self.run_dir = Path(run_dir)
        self.repo = Path(repo)
        self.device = torch.device(device)
        self.model_cls = model_cls

        if not (self.repo / "src" / "delrec").is_dir():
            raise FileNotFoundError(f"not a DelRec checkout: {self.repo}")
        if not (self.run_dir / "best.pth").is_file():
            raise FileNotFoundError(f"missing best.pth in {self.run_dir}")

        sys.path[:0] = [str(self.repo), str(self.repo / "src")]

        from delrec.delay_layers import axonal_recdel
        from delrec.utils import reset_states

        self._axonal_recdel = axonal_recdel
        self._reset_states = reset_states

        cfg_mod = importlib.import_module("configs.perf_SHD")
        zoo = importlib.import_module("delrec.networks_shd")

        cfg = cfg_mod.Config()
        # configs/perf_SHD.py on disk reflects whatever experiment ran last; the
        # run's own config.json is the authority for architecture. Without this a
        # 352-wide checkpoint cannot be loaded after a 256-wide one was trained.
        run_cfg_path = self.run_dir / "config.json"
        if run_cfg_path.is_file():
            run_meta = json.loads(run_cfg_path.read_text(encoding="utf-8"))
            for k in ("hidden_layers", "tau", "bias", "use_batch_norm",
                      "sigma_init", "sigma_decay", "max_rec_delay",
                      "use_augmentations", "lr_w", "n_bins", "time_step",
                      "round_pos_each_epoch", "store_v_seq", "backend"):
                if k in run_meta:
                    setattr(cfg, k, run_meta[k])
        cfg.epochs = self._trained_epochs()
        cfg.results_dir = str(self.run_dir)
        self.config = cfg

        self.model = getattr(zoo, self.model_cls)(cfg).to(self.device)

        ckpt = torch.load(self.run_dir / "best.pth", map_location=self.device,
                          weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "state_dict mismatch: "
                f"missing={incompatible.missing_keys} "
                f"unexpected={incompatible.unexpected_keys}")

        self.ckpt_epoch = int(ckpt.get("epoch", 0))
        self.ckpt_val_acc = float(ckpt.get("acc", float("nan")))

        for e in range(cfg.epochs):
            for m in self.model.modules():
                if isinstance(m, axonal_recdel):
                    m.update_sigma(e)

        # DelRec paper §2.2: "round delays to the nearest integer for inference".
        # train.py's final test does this via round_pos_each_epoch; a direct
        # forward here would otherwise use the fractional delays and report a
        # non-integer ring-buffer depth.
        if hasattr(self.model, "round_pos"):
            with torch.no_grad():
                self.model.round_pos()

        self.model.eval()
        if verbose:
            self.describe()

    # -- introspection ---------------------------------------------------- #

    def _trained_epochs(self) -> int:
        csv = self.run_dir / "train_res.csv"
        if csv.is_file():
            header = csv.read_text(encoding="utf-8").splitlines()[0]
            n = len([c for c in header.split(",")[1:] if c.strip()])
            if n > 0:
                return n
        return TRAINED_EPOCHS

    @property
    def sigma(self) -> list[float]:
        return [round(float(m.sigma), 6) for m in self.model.modules()
                if isinstance(m, self._axonal_recdel)]

    @property
    def delays(self) -> np.ndarray:
        """Learned recurrent delays of the first recurrent layer."""
        for m in self.model.modules():
            if isinstance(m, self._axonal_recdel):
                return m.recurrent_delays.detach().cpu().numpy()
        raise RuntimeError("no recurrent delay layer found")

    def describe(self) -> None:
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"repo        : {self.repo}")
        print(f"run_dir     : {self.run_dir}")
        print(f"device      : {self.device}")
        print(f"params      : {n_params:,}")
        print(f"epochs      : {self.config.epochs}  (checkpoint epoch {self.ckpt_epoch}, "
              f"val {self.ckpt_val_acc:.4f})")
        print(f"arch        : hidden={self.config.hidden_layers} tau={self.config.tau} "
              f"bn={self.config.use_batch_norm} bias={self.config.bias}")
        print(f"input       : (T, batch, {self.config.input_size})  "
              f"output: {self.config.output_size} classes")
        print(f"sigma       : {self.sigma}")
        d = self.delays
        print(f"delays      : {d.size} values, mean {d.mean():.2f}, "
              f"min {d.min():.0f}, max {d.max():.0f}")

    # -- measurement (Study 1) -------------------------------------------- #

    def _layer_table(self):
        """(name, module, fanout) for every registered spike snapshot.

        The SHD model brackets each spiking stage with a ``spike_registrator``
        (``layers.3`` and ``layers.8``) that clones the spike tensor during
        forward. That is more reliable than hooking the LIF itself: the LIF inside
        ``axonal_recdel`` is driven through its single-step path, so a module-level
        forward hook on it never fires -- which is why hooking LIFNodes only saw
        the first and last layer.

        fanout = MACs one spike triggers in the next Linear layer.
        """
        from delrec.networks_shd import spike_registrator

        flat = [(n, m) for n, m in self.model.named_modules() if n]
        rows = []
        for i, (n, m) in enumerate(flat):
            if not isinstance(m, spike_registrator):
                continue
            fan = 0
            for _n2, m2 in flat[i + 1:]:
                if isinstance(m2, torch.nn.Linear):
                    fan = m2.out_features
                    break
            rows.append((n, m, fan))
        return rows

    def delay_buffer_shape(self) -> dict:
        """Ring-buffer geometry exactly as the reference implementation allocates it.

        ``delay_layers_pytorch.py:143-145``:
            # Eq. (11): the kernel has finite support, so scheduling only ever needs
            # a window of L = ceil(1 + max_j d_j + (1 + sigma)) steps, not all of T.
            L = int(torch.ceil(1.0 + torch.max(d) + s_max).item()) + 1
            y_window = torch.zeros(B, N, L - 1, ...)

        So the number of slots per neuron is ``L - 1 = ceil(1 + max(d) + 1 + sigma)``,
        one more than max(d) plus the one-step recurrent framing, plus the sigma
        half-width of the triangular kernel. Using max(d) alone (what an earlier
        version of this file did) under-reports the buffer by about 5%.
        """
        d = self.delays
        sigma = float(self.sigma[0]) if self.sigma else 0.0
        slots = int(math.ceil(1.0 + float(d.max()) + 1.0 + sigma))
        return {
            "n": int(d.size),
            "sum": float(d.sum()),
            "mean": float(d.mean()),
            "max_delay": float(d.max()),
            "sigma": sigma,
            "slots": slots,
            # 1 bit per slot: a slot only ever holds "did this neuron fire".
            "bits": float(d.size * slots),
        }

    def measure(self, x, ops_per_neuron: float = 1.0,
                reads_per_neuron: float = 1.0,
                writes_per_neuron: float = 1.0) -> MeasureResult:
        """Record spike counts, SOP, delay-buffer occupancy and state traffic.

        Parameters
        ----------
        x : (T, B, N) spike frames, N = config.input_size (140).

        ops_per_neuron : MACs one neuron spends per step updating its state.
            PROVENANCE: now from PMSN Table 6.9 (the thesis' Table 6.9 / paper's
            Supplementary). The LIF row is
                (m*n*Fr_in + n*Fr_out)*E_AC + n*E_MAC
            i.e. exactly one MAC per neuron per step, so 1.0 is the paper's value
            for LIF, not a guess. The PMSN row is
                m*n*Fr_in*E_AC + (2*n*log2(T) + 5*n)*E_MAC
            so a multi-compartment neuron costs ``2*log2(T) + 5`` MACs per step --
            at T=100 that is ~18.3x a LIF neuron. That ratio is the mechanism the
            whitepaper's Study 1 is built on.

        reads_per_neuron, writes_per_neuron : state accesses per neuron per step.
            PROVENANCE: also OUR assumption. "1 read + 1 write per neuron per
            step" is the conventional clock-driven accounting, not something
            either paper specifies. Set them to 0.0 for an event-driven model,
            where a neuron is touched only when input arrives.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"expected (T, B, N) or (T, N), got {tuple(x.shape)}")
        if x.shape[2] != self.config.input_size:
            raise ValueError(f"channel mismatch: got {x.shape[2]}, "
                             f"expected {self.config.input_size}")
        x = x.to(self.device)

        snaps = self._layer_table()
        self._reset_states(self.model)
        with torch.no_grad():
            self.model(x)

        T, B = int(x.shape[0]), int(x.shape[1])
        res = MeasureResult(T=T, batch=B)

        for name, mod, fan in snaps:
            spk = mod.spikes
            if spk is None:
                continue
            per_t = spk.sum(dim=(1, 2)).cpu().numpy()
            res.spikes[name] = int(per_t.sum())
            res.spikes_per_t[name] = per_t
            res.fanout[name] = fan
            res.sop_by_layer[name] = int(per_t.sum()) * fan
        res.sop = sum(res.sop_by_layer.values())

        # PMSN: "The total cost is computed by considering both synaptic operations
        # and neuronal operations." Counting only the synaptic term understates
        # every model, and understates multi-compartment models most, because
        # their per-neuron update costs more than one LIF step.
        res.ops_per_neuron = float(ops_per_neuron)
        n_neurons_total = sum(int(m.spikes.shape[2]) for _n, m, _f in snaps
                              if m.spikes is not None)
        n_neurons_total += int(self.config.output_size)
        # Every sample in the batch updates every neuron on every step.
        res.sop_neuron = int(round(n_neurons_total * T * B * res.ops_per_neuron))

        res.delay_buffer = self.delay_buffer_shape()

        per_layer = {}
        total = 0
        for name, mod, _fan in snaps:
            spk = mod.spikes
            if spk is None:
                continue
            n_neurons = int(spk.shape[2])
            per_layer[name] = int(T * B * n_neurons)   # every sample, every step
            total += per_layer[name]

        # The readout LIF (the final 20-wide layer) never emits -- its threshold is
        # 1e8 -- so it has no spike_registrator. It still holds membrane state on
        # every step, so account for its accesses explicitly.
        readout = int(T * B * self.config.output_size)
        per_layer["readout (no registrator)"] = readout
        total += readout

        n_access = int(round(n_neurons_total * T
                             * (reads_per_neuron + writes_per_neuron) / 2))
        res.state_rw = {
            # OUR assumption, not the paper's: the clock-driven convention that
            # every neuron is touched every step. Neither DelRec nor PMSN states
            # this anywhere in code; see measure()'s docstring.
            "assumption": (f"clock-driven, {reads_per_neuron:g} read + "
                           f"{writes_per_neuron:g} write per neuron per step "
                           f"[OUR assumption, not from the papers]"),
            "per_layer_accesses": per_layer,
            "total_neurons": total // (T * B),
            "total_reads": n_access,
            "total_writes": n_access,
            "total_accesses": 2 * n_access,
        }
        return res

    def measure_loader(self, loader, max_batches=None, batch_first=True,
                       ops_per_neuron: float = 1.0,
                       reads_per_neuron: float = 1.0,
                       writes_per_neuron: float = 1.0) -> MeasureResult:
        """Aggregate measure() across a loader, reported per inference.

        Study 1 asks for energy *per inference*, so the fields of the returned
        result are totals divided by the number of samples. ``totals`` keeps the
        raw sums: the division is lossless, and the accumulated ``spikes_per_t``
        is exactly what a later burst / buffer-contention analysis would need.

        ``batch_first`` is True for the loaders in this repo: their collate
        returns ``(B, T, N)`` (see ``training/shd.py``: "inputs of shape
        (Batch, Time, Neurons)"), which ``measure`` then permutes to ``(T, B, N)``.
        """
        spikes_tot = {}
        per_t_tot = {}
        sop_tot = {}
        rw_tot = {}
        fanout_ref = {}
        n_total = 0
        T_ref = None

        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            x = batch[0]
            if batch_first:
                x = x.permute(1, 0, 2)
            r = self.measure(x, ops_per_neuron=ops_per_neuron,
                             reads_per_neuron=reads_per_neuron,
                             writes_per_neuron=writes_per_neuron)

            if T_ref is None:
                T_ref = r.T
            elif r.T != T_ref:
                raise ValueError(f"T changed across batches: {T_ref} -> {r.T}")
            n_total += r.batch

            for k, v in r.spikes.items():
                spikes_tot[k] = spikes_tot.get(k, 0) + v
                fanout_ref[k] = r.fanout[k]
            for k, v in r.spikes_per_t.items():
                per_t_tot[k] = per_t_tot.get(k, np.zeros_like(v)) + v
            for k, v in r.sop_by_layer.items():
                sop_tot[k] = sop_tot.get(k, 0) + v
            for k, v in r.state_rw["per_layer_accesses"].items():
                rw_tot[k] = rw_tot.get(k, 0) + v

        if n_total == 0:
            raise RuntimeError("loader produced no batches")

        res = MeasureResult(T=T_ref, batch=n_total, n_samples=n_total)
        res.fanout = fanout_ref
        res.spikes = {k: v // n_total for k, v in spikes_tot.items()}
        res.spikes_per_t = {k: v / n_total for k, v in per_t_tot.items()}
        res.sop_by_layer = {k: v // n_total for k, v in sop_tot.items()}
        res.sop = sum(sop_tot.values()) // n_total

        res.ops_per_neuron = float(ops_per_neuron)
        n_neurons_total = sum(int(m.spikes.shape[2]) for _n, m, _f in self._layer_table()
                              if m.spikes is not None)
        n_neurons_total += int(self.config.output_size)
        res.sop_neuron = int(round(n_neurons_total * T_ref * res.ops_per_neuron))

        res.delay_buffer = self.delay_buffer_shape()

        rw_sum = sum(rw_tot.values())
        n_access = rw_sum // n_total
        res.state_rw = {
            # OUR assumption, not the paper's -- see measure()'s docstring.
            "assumption": (f"clock-driven, {reads_per_neuron:g} read + "
                           f"{writes_per_neuron:g} write per neuron per step "
                           f"[OUR assumption, not from the papers]"),
            "per_layer_accesses": {k: v // n_total for k, v in rw_tot.items()},
            "total_neurons": rw_sum // n_total // T_ref,
            "total_reads": int(round(n_access * reads_per_neuron)),
            "total_writes": int(round(n_access * writes_per_neuron)),
            "total_accesses": int(round(n_access * (reads_per_neuron + writes_per_neuron))),
        }

        res.totals = {
            "n_samples": n_total,
            "spikes": spikes_tot,
            "sop": sum(sop_tot.values()),
            "sop_by_layer": sop_tot,
            "state_accesses": rw_sum,
            "spikes_per_t": per_t_tot,   # arrival curve accumulated over the split
        }
        return res

    # -- inference -------------------------------------------------------- #


    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """(T, B, 140) or (T, 140) -> logits (B, 20), mean over time."""
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = x.float()
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(f"expected (T, B, N) or (T, N), got {tuple(x.shape)}")
        if x.shape[2] != self.config.input_size:
            raise ValueError(
                f"channel mismatch: got {x.shape[2]}, "
                f"expected {self.config.input_size} (= 700 // n_bins)")

        x = x.to(self.device)
        self._reset_states(self.model)
        with torch.no_grad():
            v = self.model(x)                 # (T, B, 20) membrane potential
            return v.mean(dim=0)              # the checkpoint's readout

    def predict_proba(self, x) -> torch.Tensor:
        return torch.softmax(self(x), dim=-1)

    def predict(self, x) -> torch.Tensor:
        return self(x).argmax(dim=-1)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _load_from_h5(repo: Path, split: str, index: int) -> np.ndarray:
    """One raw SHD sample as (T, 700) frames, using the repo's own preprocessing."""
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    import importlib as _il
    cfg_mod = _il.import_module("configs.perf_SHD")
    loaders = _il.import_module("delrec.datasets.loaders")

    cfg = cfg_mod.Config()
    cfg.epochs = TRAINED_EPOCHS
    # The upstream config keeps a relative path; resolve it against the repo so
    # this works from any working directory.
    cfg.datasets_path = str(Path(repo) / "Datasets" / "SHD")
    ds = loaders.BinnedSpikingHeidelbergDigits(
        cfg.datasets_path, cfg.n_bins,
        train=(split == "train"), data_type="frame", duration=cfg.time_step)
    x, y = ds[index]
    return np.asarray(x), int(y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--run-dir", default=DEFAULT_RUN)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--npy", default=None, help="classify a (T,140) or (T,B,140) array")
    ap.add_argument("--h5-index", type=int, default=None,
                    help="classify one SHD test sample by index")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    args = ap.parse_args()

    m = DelayModel(run_dir=args.run_dir, repo=args.repo, device=args.device)

    if args.npy:
        x = np.load(args.npy)
        print(f"\ninput : {args.npy}  shape {x.shape}")
        probs = m.predict_proba(x)
        cls = probs.argmax(dim=-1)
        print(f"class : {cls.tolist()}")
        print(f"conf  : {probs.max(dim=-1).values.tolist()}")
    elif args.h5_index is not None:
        x, y = _load_from_h5(Path(args.repo), args.split, args.h5_index)
        print(f"\ninput : {args.split}[{args.h5_index}]  shape {x.shape}  label {y}")
        probs = m.predict_proba(x)
        cls = int(probs.argmax(dim=-1).item())
        print(f"pred  : {cls}   correct={cls == y}   conf={probs.max().item():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
