import matplotlib.pyplot as plt

from units.enemy_units import *
from units.friendly_units import *
from state_templates import *
from domains import enemy_domain, secure_outpost_domain
from simulation import Simulation
from unit_factory import make_enemy, make_friendly
from log import logger

# Import GA entry‐point and run helper
from ga_optimizer import genetic_optimize, build_friendly_units, _build_enemy_units, COST_MAP

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
    return sim.run(max_steps=300)

def main():
    # 1) Define your enemy list
    enemy_units = _build_enemy_units()
    enemy_names = [e.name for e in enemy_units]

    # 2) Define min/max counts for each friendly type
    min_counts = {
        "FriendlyTankGroup":      0,
        "FriendlyInfantryGroup":  0,
        "FriendlyArtilleryGroup": 0,
        "FriendlyAntiTankGroup":  0,
    }
    max_counts = {
        "FriendlyTankGroup":      3,
        "FriendlyInfantryGroup":  3,
        "FriendlyArtilleryGroup": 3,
        "FriendlyAntiTankGroup":  3,
    }

    # 3) Run the GA optimizer
    best_score, (best_config, best_seq) = genetic_optimize(
        enemy_names,
        pop_size=10,
        generations=2,
        min_size=2,
        min_counts=min_counts,
        max_counts=max_counts,
        cost_weight=1.0,
        retain_frac=0.3,
        random_frac=0.1,
        mutate_p=0.2,
        tournament_size=5
    )

    # 4) Log out your optimal force mix & plan
    total_cost = sum(best_config[n] * COST_MAP[n] for n in best_config)
    logger.info("=== GA Optimization Results ===")
    logger.info(f"Best (score - cost): {best_score:.1f}")
    logger.info(f"Total deployment cost: {total_cost}")
    logger.info(f"Friendly config: {best_config}")
    logger.info(f"Attack sequence: {best_seq}")

    # 5) Visualize the best‐found sequence
    final = run_with_sequence(best_config, best_seq, visualize=True)
    logger.info("=== Final Simulation Metrics ===")
    logger.info(f"Score: {final['score']:.1f}")
    logger.info(f"Friendly health remaining: {final['health']}")
    logger.info(f"Enemy health remaining: {final['enemy_health']}")
    logger.info(f"Outpost secured: {final['outpost_secured']}")
    logger.info(f"Steps taken: {final['steps_taken']}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
