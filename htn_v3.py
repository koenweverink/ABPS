import argparse, os, datetime, json
import matplotlib.pyplot as plt

from units.enemy_units import *
from units.friendly_units import *
from state_templates import (
    tank_state_template,
    infantry_state_template,
    artillery_state_template,
    anti_tank_state_template,
    enemy_anti_tank_state_template,
    enemy_infantry_state_template,
    enemy_artillery_state_template,
    enemy_tank_state_template
)
from domains import enemy_domain, secure_outpost_domain
from simulation import Simulation
from unit_factory import make_enemy, make_friendly
from log import logger

# Import GA entry‐point and run helper
from ga_optimizer import genetic_optimize, build_friendly_units, _build_enemy_units, COST_MAP
from gantt import generate_gantt_chart

def _build_units(config):
    enemy_units = _build_enemy_units()
    friendly_units = build_friendly_units(config)
    return friendly_units, enemy_units

def run_with_sequence(config, sequence, visualize=False):
    """Run a sim with a pre-built set of friendly & enemy units, according to sequence."""
    friendly_units, enemy_units = _build_units(config)
    sim = Simulation(friendly_units, enemy_units, visualize=visualize, plan_name="HTN_V3_Sim")
    sim.attack_sequence = sequence
    for u in friendly_units:
        u.sim = sim
    print(f"Running final simulation with attack sequence: {sequence}")
    return sim.run(max_steps=500)

def test_run():
    """Execute a single predefined simulation used for quick testing."""
    config = {
        "FriendlyTankGroup":      1,
        "FriendlyInfantryGroup":  2,
        "FriendlyArtilleryGroup": 1,
        "FriendlyAntiTankGroup":  2,    
    }

    sequence = [
        # 1. Same friendly unit reused for multiple attacks (sequential reuse)
        ("EnemyTankGroup1", "ConsolidateAttack", ["FriendlyTankGroup_0", "FriendlyArtilleryGroup_0"]),
        ("EnemyInfantryGroup1", "FlankAttack", ["FriendlyTankGroup_0", "FriendlyArtilleryGroup_0"]),
    ]

    sequence2 = [
    # 2. Large attack using all available friendly units (resource saturation)
    ("EnemyTankGroup1", "ConsolidateAttack", [
        "FriendlyTankGroup_0",
        "FriendlyAntiTankGroup_0",
        "FriendlyInfantryGroup_0",
        "FriendlyArtilleryGroup_0"
    ])]

    sequence3 = [
        # 3. Overlapping assignments requiring synchronization
        ("EnemyArtilleryGroup1", "FlankAttack", ["FriendlyTankGroup_0", "FriendlyArtilleryGroup_0"]),
        ("EnemyAntiTankGroup1", "ConsolidateAttack", ["FriendlyTankGroup_0", "FriendlyAntiTankGroup_0"]),
    ]

    sequence4 = [
        # 4. Minimal allocation with a single specialized unit
        ("EnemyInfantryGroup1", "ConsolidateAttack", ["FriendlyInfantryGroup_1"]),
        ("EnemyInfantryGroup2", "ConsolidateAttack", ["FriendlyInfantryGroup_1"]),
        ("EnemyInfantryGroup3", "ConsolidateAttack", ["FriendlyInfantryGroup_1"]),
    ]

    test_seq = [
        ('EnemyTankGroup1', 'ConsolidateAttack', ['FriendlyArtilleryGroup_0', 'FriendlyAntiTankGroup_1', 'FriendlyAntiTankGroup_0']), 
        ('EnemyTankGroup2', 'FlankAttack', ['FriendlyInfantryGroup_1', 'FriendlyTankGroup_0', 'FriendlyAntiTankGroup_0', 'FriendlyArtilleryGroup_0', 'FriendlyInfantryGroup_0']), 
        ('EnemyAntiTankGroup1', 'FlankAttack', ['FriendlyInfantryGroup_0', 'FriendlyAntiTankGroup_0', 'FriendlyInfantryGroup_1']), 
        ('EnemyArtilleryGroup1', 'ConsolidateAttack', ['FriendlyInfantryGroup_1', 'FriendlyAntiTankGroup_0']), 
        ('EnemyInfantryGroup1', 'FlankAttack', ['FriendlyTankGroup_1', 'FriendlyInfantryGroup_1', 'FriendlyInfantryGroup_0', 'FriendlyTankGroup_0'])
    ]

    result = run_with_sequence(config, test_seq, visualize=True)
    logger.info("=== Test Simulation Metrics ===")
    logger.info(f"Score: {result['score']:.1f}")
    logger.info(f"Friendly health remaining: {result['health']}")
    logger.info(f"Enemy health remaining: {result['enemy_health']}")
    logger.info(f"Outpost secured: {result['outpost_secured']}")
    logger.info(f"Steps taken: {result['steps_taken']}")

    plt.ioff()
    plt.show()

def _jsonable(obj):
    """Convert tuples -> lists so we can dump JSON without issues."""
    if isinstance(obj, tuple):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def _pick_best(result_dict):
    """Prefer best_overall; fallback to best_final_gen."""
    return result_dict.get("best_overall") or result_dict.get("best_final_gen")


def main():
    # --- Paths that match your repo style ---
    ROOT = os.getcwd()
    OUTDIR_DEFAULT = os.path.join(ROOT, "results")
    LOGDIR_DEFAULT = os.path.join(ROOT, "logs")
    CKPTDIR_DEFAULT = os.path.join(ROOT, "checkpoints")
    CKPT_PATH_DEFAULT = os.path.join(CKPTDIR_DEFAULT, "ga_checkpoint.pkl")
    BEST_PATH_DEFAULT = os.path.join(CKPTDIR_DEFAULT, "ga_best.pkl")
    WORKER_LOG_DEFAULT = os.path.join(LOGDIR_DEFAULT, "eval_runs.log")

    parser = argparse.ArgumentParser(description="HTN v3 + GA optimizer (rich results + checkpointing)")
    parser.add_argument("--test", action="store_true", help="run a predefined test simulation and exit")

    # GA knobs (defaults aligned with what you’ve been using)
    parser.add_argument("--mode", choices=["nsga2", "single"], default="nsga2")
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-seeds", type=int, default=1, help="num eval seeds per chromosome")
    parser.add_argument("--cost-weight", type=float, default=1.0)
    parser.add_argument("--elites", type=int, default=2)
    parser.add_argument("--random-frac", type=float, default=0.10)
    parser.add_argument("--mut-start", type=float, default=0.40)
    parser.add_argument("--mut-end", type=float, default=0.10)
    parser.add_argument("--robust-lambda", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=300)

    # Checkpoint/resume (folders reflect your repo layout)
    parser.add_argument("--checkpoint-every", type=int, default=1, help="0 disables periodic checkpoints")
    parser.add_argument("--ckpt-path", default=CKPT_PATH_DEFAULT)
    parser.add_argument("--best-path", default=BEST_PATH_DEFAULT)
    parser.add_argument("--fresh", action="store_true", help="ignore checkpoint and start fresh")

    # Output
    parser.add_argument("--outdir", default=OUTDIR_DEFAULT, help="folder for JSON + artifacts")
    parser.add_argument("--worker-log", default=WORKER_LOG_DEFAULT, help="per-simulation prints from workers")

    args = parser.parse_args()

    # Optional local test hook
    if args.test:
        try:
            test_run()  # noqa: F821 - assumed to exist in your file
            return
        except NameError:
            print("[warn] test_run() not defined here; skipping `--test`.")

    # Ensure folders exist
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.best_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.worker_log), exist_ok=True)

    # 1) Enemies for GA
    enemy_units = _build_enemy_units()
    enemy_names = [e.name for e in enemy_units]

    # 2) Friendly caps (your 0..2 pattern)
    min_counts = {
        "FriendlyTankGroup":      0,
        "FriendlyInfantryGroup":  0,
        "FriendlyArtilleryGroup": 0,
        "FriendlyAntiTankGroup":  0,
    }
    max_counts = {
        "FriendlyTankGroup":      2,
        "FriendlyInfantryGroup":  2,
        "FriendlyArtilleryGroup": 2,
        "FriendlyAntiTankGroup":  2,
    }

    # 3) Auto-resume if a checkpoint exists and --fresh not given
    resume_from = None
    if not args.fresh and os.path.exists(args.ckpt_path):
        resume_from = args.ckpt_path
        logger.info(f"[MAIN] Resuming from checkpoint: {resume_from}")

    # 4) Run GA
    result = genetic_optimize(
        enemy_names,
        pop_size=args.pop_size,
        generations=args.generations,
        min_size=2,
        min_counts=min_counts,
        max_counts=max_counts,
        mode=args.mode,
        cost_weight=args.cost_weight,
        elites_per_gen=args.elites,
        random_frac=args.random_frac,
        mutate_p_start=args.mut_start,
        mutate_p_end=args.mut_end,
        num_eval_seeds=args.num_seeds,
        robust_lambda=args.robust_lambda,
        workers=args.workers,
        max_steps=args.max_steps,
        bias_team=True,
        quiet_workers=True,
        worker_log_file=args.worker_log,
        # checkpointing
        checkpoint_every=args.checkpoint_every,
        checkpoint_path=args.ckpt_path,
        best_path=args.best_path,
        resume_from=resume_from,
        # clean re-eval of the best final plan
        reeval_best=True,
        reeval_top_k=1,
        reeval_num_eval_seeds=max(3, args.num_seeds),
        reeval_robust_lambda=0.0,
    )

    # 5) Persist rich results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best = _pick_best(result)
    if not best:
        print("No best solution returned — aborting.")
        return

    # JSON artifacts
    best_json_path = os.path.join(args.outdir, f"best_result_{timestamp}.json")
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(best), f, indent=2)
    with open(os.path.join(args.outdir, f"population_final_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(result.get("population_final", [])), f, indent=2)
    with open(os.path.join(args.outdir, f"pareto_front_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(result.get("pareto_front", [])), f, indent=2)
    with open(os.path.join(args.outdir, f"history_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonable(result.get("history", [])), f, indent=2)

    # Compatibility artifact (plain text) — keep name you used before
    best_config = best["chromosome"]["config"]
    best_seq = best["chromosome"]["sequence"]
    with open(os.path.join(args.outdir, "best_config.txt"), "w", encoding="utf-8") as f:
        f.write(str(best_config) + "\n")
        f.write(str(best_seq) + "\n")
    # Also drop a copy in repo root (matches your earlier workflow)
    with open("best_config.txt", "w", encoding="utf-8") as f:
        f.write(str(best_config) + "\n")
        f.write(str(best_seq) + "\n")

    # 6) Console summary
    total_cost = sum(best_config[n] * COST_MAP[n] for n in best_config)
    print("=== GA Optimization Results ===")
    fit = best.get("fitness", None)
    ev = best.get("eval", {})
    if fit is not None:
        print(f"Best fitness: {fit:.2f}")
    print(f"Adj score: {ev.get('adjusted_raw', 0):.2f} | mean: {ev.get('mean_raw', 0):.2f} | std: {ev.get('std_raw', 0):.2f}")
    print(f"Total deployment cost: {total_cost:.1f}")
    print(f"Friendly config: {best_config}")
    print(f"Attack sequence (N={len(best_seq)}): {best_seq}")
    if "seeds" in ev:
        print(f"Eval seeds: {ev['seeds']} (seed_base={ev.get('seed_base')})")
    print(f"[SAVE] Best JSON → {best_json_path}")
    print(f"[SAVE] Worker logs → {args.worker_log}")

    # 7) Replay the best plan and make a Gantt
    friendly_units = build_friendly_units(best_config)
    enemy_units = _build_enemy_units()   # fresh copy
    sim = None
    try:
        sim = Simulation(enemy_units=enemy_units, friendly_units=friendly_units, visualize=False, plan_name="BestRun")
        sim.attack_sequence = best_seq
        for u in friendly_units:
            u.sim = sim
        sim.run(max_steps=args.max_steps)
    except Exception as e:
        logger.error(f"[MAIN] Replay sim failed: {e}")

    gantt_base = f"best_run_gantt_{timestamp}.png"
    gantt_path = os.path.join(args.outdir, gantt_base)
    try:
        generate_gantt_chart(friendly_units, filename=gantt_path)  # noqa: F821
        print(f"Saved Gantt chart to {gantt_path}")
        # also drop a simple name in repo root, like before
        try:
            generate_gantt_chart(friendly_units, filename="best_run_gantt.png")  # noqa: F821
            print("Saved Gantt chart to best_run_gantt.png")
        except Exception:
            pass
    except NameError:
        print("Note: generate_gantt_chart(...) not found/imported here; skipping Gantt export.")


if __name__ == "__main__":
    main()
