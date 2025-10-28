# reeval_plan_from_json.py
import json, os, random, datetime, statistics, sys
try:
    import numpy as np
except ImportError:
    np = None

# 1) Load plan (config + sequence)
PLAN_JSON = "best_artifact_extracted_2.json"   # adjust path if needed
with open(PLAN_JSON, "r", encoding="utf-8") as f:
    blob = json.load(f)

config  = blob["config"]
sequence = blob["sequence"]

# 2) Import the simulator runner
from htn_v3 import run_with_sequence

def eval_once(seed: int, max_steps: int = 300):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    result = run_with_sequence(config, sequence, visualize=False)
    return {"seed": seed, **result}

# 3) Sweep seeds with progress bar
SEEDS = [10001 + i for i in range(30)]   # evaluate on 30 seeds
raw = []

try:
    from tqdm import tqdm
    iterator = tqdm(SEEDS, desc="Re-evaluating plan")
except ImportError:
    iterator = SEEDS

for i, s in enumerate(iterator, 1):
    raw.append(eval_once(s))
    # Fallback: simple progress print if tqdm not available
    if "tqdm" not in sys.modules:
        print(f"[{i}/{len(SEEDS)}] Finished seed {s}")

# 4) Aggregate summary
scores = [r["score"] for r in raw]
success = [1 if r.get("outpost_secured") else 0 for r in raw]
summary = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "n_evals": len(raw),
    "mean_score": statistics.fmean(scores),
    "std_score": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
    "success_rate": sum(success) / len(success),
    "mean_steps": statistics.fmean([r["steps_taken"] for r in raw]),
    "friendly_health_mean": statistics.fmean([r["health"] for r in raw]),
    "enemy_health_mean": statistics.fmean([r["enemy_health"] for r in raw]),
    "plan_source": os.path.abspath(PLAN_JSON),
}

# 5) Save artifacts
with open("plan_reeval_raw.json", "w", encoding="utf-8") as f:
    json.dump(raw, f, indent=2)
with open("plan_reeval_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("\n[i] Wrote plan_reeval_raw.json and plan_reeval_summary.json")
