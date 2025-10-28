# final_ga_run.py
import os, json, datetime, random

def pick_workers():
    try:
        import multiprocessing as mp
        cpu = max(1, mp.cpu_count())
    except Exception:
        cpu = 4
    # use all logical cores minus one for headroom, but at least 2
    return max(2, min(cpu - 1, cpu))

def main():
    from ga_optimizer import genetic_optimize  # import inside main on Windows

    enemy_names = [
        "EnemyInfantryGroup1",
        "EnemyTankGroup1",
        "EnemyAntiTankGroup1",
        "EnemyArtilleryGroup1",
    ]

    min_counts = {
        "FriendlyTankGroup": 0,
        "FriendlyInfantryGroup": 0,
        "FriendlyArtilleryGroup": 0,
        "FriendlyAntiTankGroup": 0,
    }
    max_counts = {
        "FriendlyTankGroup": 3,
        "FriendlyInfantryGroup": 3,
        "FriendlyArtilleryGroup": 3,
        "FriendlyAntiTankGroup": 3,
    }

    workers = int(os.getenv("WORKERS_OVERRIDE", pick_workers()))
    print(f"[final_ga_run] using workers={workers}")

    cfg = {
        "enemy_names": enemy_names,
        "min_size": 2,
        "min_counts": min_counts,
        "max_counts": max_counts,

        "mode": "nsga2",
        "pop_size": 30,
        "generations": 20,

        "elites_per_gen": 2,
        "random_frac": 0.08754045570898923,
        "block_shift_mut_prob": 0.28530503865890067,
        "mutate_p_start": 0.6621594609137995,
        "mutate_p_end": 0.041335849117053874,
        "robust_lambda": 0.8612751748175633,
        "time_weight": 0.5,

        "num_eval_seeds": 1,       # use 1 during GA search (per your plan)
        "workers": workers,
        "max_steps": 300,
        "bias_team": True,
        "quiet_workers": True,
        "op_log_sample_rate": 50,
        "cache_capacity": 2000,

        "checkpoint_every": 2,
        "resume_from": None,
        "best_path": "final_ga_best.pkl",
        "checkpoint_path": "final_ga_checkpoint",
        "reeval_best": True,       # re-evaluates the best with num_eval_seeds
        "rng": random.Random(999),           # <-- int seed, not a Random() object
    }

    res = genetic_optimize(**cfg)

    summary = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "workers_used": workers,
        "pop_size": cfg["pop_size"],
        "generations": cfg["generations"],
        "num_eval_seeds": cfg["num_eval_seeds"],
        "best_final_gen_fitness": (
            None if not res.get("best_final_gen") else res["best_final_gen"].get("fitness")
        ),
        "pareto_size": (0 if not res.get("pareto_front") else len(res["pareto_front"])),
        "hypervolume": res.get("hypervolume") or (res.get("metrics", {}) or {}).get("hypervolume"),
    }

    with open("final_ga_results_full.json", "w", encoding="utf-8") as f:
        json.dump({"cfg_used": cfg, "result": res}, f, indent=2, default=str)
    with open("final_ga_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    # Windows-safe multiprocessing bootstrap
    import multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method("spawn")   # Windows default; explicit for clarity
    except RuntimeError:
        pass  # start method already set
    main()
