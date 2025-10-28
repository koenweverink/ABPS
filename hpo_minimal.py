#!/usr/bin/env python3
"""
Minimal-run HPO for ga_optimizer.genetic_optimize with per-trial checkpoints.

Strategy (tiny budget by default):
- Stage A (low fidelity): 2 quick trials (pop=12, gens=6, seeds=1)
- Stage B (confirm): re-run top 1 at higher fidelity (pop=24, gens=12, seeds=3)
- Metrics:
  * mode="single": maximize best_final_gen["fitness"].
  * mode="nsga2": maximize hypervolume over (adjusted_raw ↑, cost ↓).

Features:
- Per-trial checkpoints (ckpt_*.pkl) and optional best files (best_*.pkl)
- Auto-resume trials if their checkpoint exists (--resume-trials)
- No external HPO libraries; pure Python
"""

import argparse
from ast import arg
import json
import math
import os
import random
import time
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

from ga_optimizer import (
    genetic_optimize,                 # main API (assumed available)
    FRIENDLY_TEMPLATES,               # for min/max friendly counts
    _build_enemy_units,               # to derive default enemy names
)

# -------------------- Helpers --------------------

def _cfg_for_log(cfg: dict, seed: int) -> dict:
    c = dict(cfg)
    c.pop("rng", None)          # remove non-serializable Random
    c["seed"] = seed            # keep a reproducible handle
    return c

def enemy_names_default():
    """Derive enemy names from the module's default builder."""
    return [e.name for e in _build_enemy_units()]

def default_bounds(max_count):
    """Return per-template min/max count dictionaries."""
    mins = {t.name: 0 for t in FRIENDLY_TEMPLATES}
    maxs = {t.name: int(max_count) for t in FRIENDLY_TEMPLATES}
    return mins, maxs

def sample_params(rng, args):
    """
    Sample a compact set of tunables within sensible ranges.
    Defaults keep elites fixed at 2 unless you widen via CLI.
    """
    # Ensure mutate_p_start >= mutate_p_end (anneal down)
    mp_start = rng.uniform(args.mutate_p_start_min, args.mutate_p_start_max)
    mp_end_hi = min(mp_start, args.mutate_p_end_max)
    mp_end_lo = min(mp_end_hi, args.mutate_p_end_min)
    mp_end = rng.uniform(mp_end_lo, mp_end_hi)

    params = {
        "mutate_p_start": mp_start,
        "mutate_p_end":   mp_end,
        "block_shift_mut_prob": rng.uniform(args.block_shift_min, args.block_shift_max),
        "elites_per_gen": rng.randint(args.elites_min, args.elites_max),
        "random_frac":    rng.uniform(args.random_frac_min, args.random_frac_max),
        "robust_lambda":  rng.uniform(args.robust_lambda_min, args.robust_lambda_max),
        # single-objective scalarization weight (unused in nsga2 mode)
        "cost_weight":    rng.uniform(args.cost_weight_min, args.cost_weight_max),
    }
    return params

def hv_2d(points, ref):
    """
    Hypervolume in 2D for objectives:
      s = adjusted_raw (maximize), c = cost (minimize).

    We compute the area of the dominated region relative to ref=(s_ref, c_ref)
    by sweeping costs from c_ref downwards, tracking the best s frontier.

    Only points with s>ref_s and c<ref_c contribute.
    """
    if not points:
        return 0.0
    s_ref, c_ref = ref
    # Filter strictly improving points w.r.t ref
    pts = [(s, c) for (s, c) in points if (s > s_ref and c < c_ref)]
    if not pts:
        return 0.0
    # Sort by cost ascending (better cost first)
    pts.sort(key=lambda x: x[1])
    area = 0.0
    best_s = s_ref
    prev_c = c_ref
    for s, c in pts:
        # width in cost dimension
        width = max(0.0, prev_c - c)
        best_s = max(best_s, s)
        height = max(0.0, best_s - s_ref)
        area += width * height
        prev_c = c
    return max(0.0, area)

def nsga2_metric(result):
    """
    Metric for nsga2 mode: hypervolume over (adjusted_raw, cost).
    Falls back to best_final_gen fitness if no pareto_front is present.
    """
    front = result.get("pareto_front", []) or []
    pts = []
    for d in front:
        ev = d.get("eval", {})
        s = float(ev.get("adjusted_raw", 0.0))
        c = float(ev.get("cost", 0.0))
        pts.append((s, c))
    if not pts:
        bf = result.get("best_final_gen")
        return float("-inf") if not bf else float(bf.get("fitness", float("-inf")))
    # Reference point: slightly worse than current worst set
    s_min = min(p[0] for p in pts)
    c_max = max(p[1] for p in pts)
    ref = (s_min - 1.0, c_max + 1.0)
    return hv_2d(pts, ref)

def single_metric(result):
    """Metric for single-objective mode: best scalar fitness of final generation."""
    bf = result.get("best_final_gen")
    return float("-inf") if not bf else float(bf.get("fitness", float("-inf")))

def run_once(cfg, args, fidelity, base_seed, trial_tag, enemy_names, min_counts, max_counts):
    """
    Run genetic_optimize for a given fidelity level and return (metric, duration, result, cfg_used).
    Includes per-trial checkpointing/best files and optional resume.
    """
    # Fidelity presets
    if fidelity == "low":
        generations = args.low_generations
        pop_size    = args.low_pop
        seeds       = 1
    elif fidelity == "high":
        generations = args.high_generations
        pop_size    = args.high_pop
        seeds       = args.high_seeds
    else:
        raise ValueError("fidelity must be 'low' or 'high'")

    # Per-trial paths
    ckpt_path = os.path.join(args.ckpt_dir, f"ckpt_{trial_tag}.pkl")
    best_path = os.path.join(args.ckpt_dir, f"best_{trial_tag}.pkl") if args.save_best else None
    resume_from = ckpt_path if (args.ckpt_every and args.resume_trials and os.path.exists(ckpt_path)) else None
    if resume_from:
        print(f"[{fidelity.upper()} RESUME] {trial_tag} from {resume_from}")

    log_dir, log_base = os.path.split(args.worker_log)
    name, ext = os.path.splitext(log_base)
    worker_log_file = os.path.join(log_dir, f"{name}_{trial_tag}{ext}")

    run_cfg = {
        "enemy_names": enemy_names,
        "pop_size": pop_size,
        "generations": generations,
        "min_size": args.min_team_size,
        "min_counts": min_counts,
        "max_counts": max_counts,
        "mode": args.mode,  # "nsga2" or "single"
        "cost_weight": cfg["cost_weight"],
        "elites_per_gen": cfg["elites_per_gen"],
        "random_frac": cfg["random_frac"],
        "block_shift_mut_prob": cfg["block_shift_mut_prob"],
        "mutate_p_start": cfg["mutate_p_start"],
        "mutate_p_end": cfg["mutate_p_end"],
        "num_eval_seeds": seeds,
        "robust_lambda": cfg["robust_lambda"],
        "time_weight": args.time_weight,
        "workers": args.workers,
        "max_steps": args.max_steps,
        "bias_team": True,
        "quiet_workers": True,
        "worker_log_file": worker_log_file,
        "op_log_sample_rate": 50,
        "cache_capacity": 2000,
        # Checkpointing & resume
        "checkpoint_every": args.ckpt_every,
        "resume_from": resume_from,
        "best_path": best_path,
        # Keep GA's own checkpoints contained to this trial
        "checkpoint_path": ckpt_path,
        # Re-eval off inside HPO (do it outside if needed)
        "reeval_best": False,
        # Reproducibility: base RNG for this trial
        "rng": random.Random(base_seed),
    }

    t0 = time.time()
    result = genetic_optimize(**run_cfg)
    dur = time.time() - t0

    metric = nsga2_metric(result) if args.mode == "nsga2" else single_metric(result)
    return metric, dur, result, run_cfg

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nsga2", "single"], default="nsga2")

    # Ultra-lean defaults (2 low + 1 high)
    ap.add_argument("--low-trials", type=int, default=2)
    ap.add_argument("--confirm", type=int, default=1)

    # Fidelity settings
    ap.add_argument("--low-generations", type=int, default=6)
    ap.add_argument("--low-pop", type=int, default=12)
    ap.add_argument("--high-generations", type=int, default=10)
    ap.add_argument("--high-pop", type=int, default=20)
    ap.add_argument("--high-seeds", type=int, default=3)

    # GA scenario & runtime
    ap.add_argument("--workers", type=int, default=None, help="GA workers per trial")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--min-team-size", type=int, default=2)
    ap.add_argument("--max-count", type=int, default=2, help="per-template upper bound")
    ap.add_argument("--worker-log", default="eval_runs.log")
    ap.add_argument("--max-workers", type=int, default=None, help="parallel trials for Stage A (ProcessPoolExecutor)")

    # Search ranges (tight; elites fixed at 2 by default)
    ap.add_argument("--mutate-p-start-min", type=float, default=0.3)
    ap.add_argument("--mutate-p-start-max", type=float, default=0.7)
    ap.add_argument("--mutate-p-end-min",   type=float, default=0.02)
    ap.add_argument("--mutate-p-end-max",   type=float, default=0.25)
    ap.add_argument("--block-shift-min", type=float, default=0.2)
    ap.add_argument("--block-shift-max", type=float, default=0.8)
    ap.add_argument("--elites-min", type=int, default=2)
    ap.add_argument("--elites-max", type=int, default=2)
    ap.add_argument("--random-frac-min", type=float, default=0.0)
    ap.add_argument("--random-frac-max", type=float, default=0.3)
    ap.add_argument("--robust-lambda-min", type=float, default=0.0)
    ap.add_argument("--robust-lambda-max", type=float, default=2.0)
    ap.add_argument("--cost-weight-min", type=float, default=0.0)
    ap.add_argument("--cost-weight-max", type=float, default=3.0)
    ap.add_argument("--time-weight", type=float, default=0.5)

    # Repro
    ap.add_argument("--seed", type=int, default=1234)

    # Checkpointing controls
    ap.add_argument("--ckpt-every", type=int, default=0, help="checkpoint every N generations (0=off)")
    ap.add_argument("--ckpt-dir", default="hpo_ckpts", help="directory for per-trial checkpoints")
    ap.add_argument("--save-best", action="store_true", help="save per-trial best files")
    ap.add_argument("--resume-trials", action="store_true", help="auto-resume trials if checkpoint exists")

    args = ap.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)

    cpu_count = os.cpu_count() or 1
    if args.workers is None and args.max_workers is None:
        args.workers = max(1, cpu_count // 2)
        args.max_workers = max(1, cpu_count // args.workers)
    elif args.workers is None:
        args.workers = max(1, cpu_count // args.max_workers)
    elif args.max_workers is None:
        args.max_workers = max(1, cpu_count // args.workers)

    rng = random.Random(args.seed)
    enemies = enemy_names_default()  # default scenario from module
    min_counts, max_counts = default_bounds(args.max_count)

    # ---- Stage A: low-fidelity search ----
    trials = []
    futures = {}
    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        for i in range(args.low_trials):
            p = sample_params(rng, args)
            base_seed = args.seed + 10_000 * (i + 1)  # seed ladder
            trial_tag = f"low_t{i}_seed{base_seed}"
            fut = ex.submit(
                run_once, p, args, "low", base_seed, trial_tag, enemies, min_counts, max_counts
            )
            futures[fut] = (i, p, base_seed, trial_tag)

        for fut in as_completed(futures):
            i, p, base_seed, trial_tag = futures[fut]
            metric, dur, result, cfg_used = fut.result()
            cfg_log = _cfg_for_log(cfg_used, base_seed)
            trials.append({
                "stage": "low",
                "trial": i,
                "metric": float(metric),
                "duration_sec": float(dur),
                "params": p,
                "cfg_used": cfg_log,             # <-- use sanitized cfg
            })
            print(f"[LOW {i+1}/{args.low_trials}] metric={metric:.4f} time={dur/60:.1f}m")

    # ---- Pick top-K for confirmation ----
    K = max(1, args.confirm)
    topK = sorted(trials, key=lambda t: t["metric"], reverse=True)[:K]

    # ---- Stage B: confirm at higher fidelity ----
    confirms = []
    for j, rec in enumerate(topK):
        p = rec["params"]
        base_seed = args.seed + 99_973 * (j + 1)  # separate ladder
        trial_tag = f"high_from{rec['trial']}_seed{base_seed}"
        metric, dur, result, cfg_used = run_once(
            p, args, "high", base_seed, trial_tag, enemies, min_counts, max_counts
        )
        cfg_log = _cfg_for_log(cfg_used, base_seed)
        confirms.append({
            "stage": "high",
            "candidate_from": rec["trial"],
            "metric": float(metric),
            "duration_sec": float(dur),
            "params": p,
            "cfg_used": cfg_log,             # <-- use sanitized cfg
            "result_head": {
                "has_pareto": bool(result.get("pareto_front")),
                "best_final_gen_fitness": (
                    None if not result.get("best_final_gen")
                    else float(result["best_final_gen"].get("fitness"))
                ),
            },
        })
        print(f"[HIGH {j+1}/{len(topK)}] from trial={rec['trial']} metric={metric:.4f} time={dur/60:.1f}m")

    # ---- Final pick ----
    best = (max(confirms, key=lambda t: t["metric"]) if confirms
            else max(trials, key=lambda t: t["metric"]))
    print("\n=== FINAL PICK ===")
    print(json.dumps(best, indent=2, default=str))

    # Save everything for reproducibility
    out = {
        "args": vars(args),
        "low_trials": trials,
        "confirms": confirms,
        "final_pick": best,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open("hpo_minimal_results_2.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: hpo_minimal_results_2.json")

if __name__ == "__main__":
    main()
