#!/bin/bash
# TASK 4 -- delay-free reference network, cosine schedule.
#
# Identical per-synapse architecture to the delay model, with recurrent_delays
# fixed at 0 and frozen by SNN_vanilla_synaptic_recurrent.  Two settings differ
# from the delay runs and MUST be applied first, or the network will not learn
# (see README section 7.5):
#
#   1. use_batch_norm = False   -- with BN the network collapses (train 13% /
#      val 24%, i.e. an inverted train/val gap).  The first-order recurrence
#      makes each layer input autocorrelated, so 40-sample batches estimate the
#      running statistics badly.
#   2. max_grad_val = 1.5 (trainer default is 1.0).
#
# Result: 49.81% test / 50.20% best val.  Compare delay_h256_T100 = 84.30%.
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PYEOF'
import pathlib, re, sys
ROOT = pathlib.Path(".").resolve()

q = ROOT / "configs" / "perf_SHD.py"
t = q.read_text()
for k, v in [("time_step", "time_step = 14"),
             ("hidden_layers", "hidden_layers = [256, 256]"),
             ("use_batch_norm", "use_batch_norm = False"),
             ("batch_size", "batch_size = 40"),
             ("lr_w", "lr_w = 1e-3"),
             ("lr_positions", "lr_positions = 5e-2"),
             ("scheduler_weights", "scheduler_weights = 'cos'"),
             ("scheduler_pos", "scheduler_pos = 'cos'")]:
    if re.search(rf"^(\s*){k}\s*=", t, re.M):
        t = re.sub(rf"^(\s*){k}\s*=.*$", lambda m, _v=v: m.group(1) + _v, t, flags=re.M)
    else:
        t = t.replace("hidden_layers = [256, 256]\n",
                      "hidden_layers = [256, 256]\n    " + v + "\n")
if "mask_T_stepsi" not in t:
    t = t.rstrip("\n") + "\n\n    mask_T_stepsi = None\n    mask_T_steps = 0\n    masked_proba = None\n"
q.write_text(t)

p = ROOT / "src" / "delrec" / "training" / "shd.py"
s = p.read_text()
if "max_grad_val = 1.0" in s:
    p.write_text(s.replace("max_grad_val = 1.0", "max_grad_val = 1.5"))

for line in q.read_text().splitlines():
    ls = line.strip()
    if ls.startswith(("time_step", "hidden_layers", "use_batch_norm", "lr_w", "scheduler")):
        print("  " + ls)
src = (ROOT / "src" / "delrec" / "training" / "shd.py").read_text()
m = re.search(r"max_grad_val\s*=\s*([0-9.]+)", src)
print("  max_grad_val =", m.group(1) if m else "NOT FOUND")
PYEOF

python experiments/train.py --dataset shd --model SNN_vanilla_synaptic_recurrent \
  --seeds 1111 --epochs 10 --kernel v2 --synaptic-kernel v2 \
  --out "${OUT:-/content/work/exp/nodelay_cos_10ep}"
