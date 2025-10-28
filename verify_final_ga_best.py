# verify_final_best.py
import json, random, datetime, statistics, os
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None
try:
    import numpy as np
except ImportError:
    np = None

# 1) Load the GA results and extract best plan
FULL_JSON = "final_ga_results_full.json"
with open(FULL_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

best = data["result"].get("best_final_gen") or data["result"].get("best_overall")
if not best:
    raise ValueError("No best plan found in final_ga_results_full.json")

chrom = best.get("chromosome") or best
config = chrom["config"]
sequence = chrom["sequence"]

print("[verify] Best config:", config)
print("[verify] Sequence length:", len(sequence))

# 2) Runner: replay a fixed plan under a given seed
from htn_v3 import run_with_sequence

def eval_once(seed: int, max_steps: int = 500):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    result = run_with_sequence(config, sequence, visualize=False)
    return {"seed": seed, **result}

# 3) Sweep seeds
SEEDS = [20001 + i for i in range(30)]
raw = []
if tqdm is not None:
    for s in tqdm(SEEDS, desc="Verifying best plan"):
        raw.append(eval_once(s))
else:
    print("[i] tqdm not installed; running without progress bar.")
    for idx, s in enumerate(SEEDS, 1):
        raw.append(eval_once(s))
        if idx % 5 == 0 or idx == len(SEEDS):
            print(f"[{idx}/{len(SEEDS)}] completed")

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
    "plan_source": os.path.abspath(FULL_JSON),
}

# 5) Save artifacts
with open("final_ga_best_raw.json", "w", encoding="utf-8") as f:
    json.dump(raw, f, indent=2)
with open("final_ga_best_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("\n[i] Wrote final_ga_best_raw.json and final_ga_best_summary.json")

