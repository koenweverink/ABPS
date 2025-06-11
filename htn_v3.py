import matplotlib.pyplot as plt

from units.enemy_units import *
from units.friendly_units import *
from state_templates import *
from domains import enemy_domain, secure_outpost_domain
from simulation import Simulation
from unit_factory import make_enemy, make_friendly

import itertools
from log import logger


def generate_attack_sequences(friendly_names, enemy_names, group_size=2):
    """Yield all possible attack sequences.

    Each sequence is a list of ``(enemy_name, [friendly_names])`` tuples. The
    order of enemies in the list represents the order in which they will be
    attacked.  For every enemy, a combination of ``group_size`` friendlies is
    chosen.  Friendlies may be reused across different enemies.
    """
    pairs = list(itertools.combinations(friendly_names, group_size))
    for enemy_order in itertools.permutations(enemy_names):
        for combo in itertools.product(pairs, repeat=len(enemy_names)):
            yield [
                (enemy, list(pairs_assigned))
                for enemy, pairs_assigned in zip(enemy_order, combo)
            ]




###############################
# Main Simulation Setup
###############################

def _build_units():
    """Create fresh friendly and enemy unit lists."""
    enemy_units = [
        make_enemy("EnemyTankGroup1", EnemyTank, enemy_tank_state_template, position=(63, 15), domain=enemy_domain),
        make_enemy("EnemyTankGroup2", EnemyTank, enemy_tank_state_template, position=(17, 5), domain=enemy_domain),
        make_enemy("EnemyInfantryGroup1", EnemyInfantry, enemy_infantry_state_template, domain=enemy_domain),
        make_enemy("EnemyAntiTankGroup1", EnemyAntiTank, enemy_anti_tank_state_template, domain=enemy_domain),
        make_enemy("EnemyArtilleryGroup1", EnemyArtillery, enemy_artillery_state_template, domain=enemy_domain),
    ]

    friendly_units = [
        make_friendly("FriendlyTankGroup", FriendlyTank, tank_state_template, secure_outpost_domain, enemy_tank_state_template),
        make_friendly("FriendlyInfantryGroup", FriendlyInfantry, infantry_state_template, secure_outpost_domain, enemy_tank_state_template),
        make_friendly("FriendlyArtilleryGroup", FriendlyArtillery, artillery_state_template, secure_outpost_domain, enemy_tank_state_template),
        make_friendly("FriendlyAntiTankGroup", FriendlyAntiTank, anti_tank_state_template, secure_outpost_domain, enemy_tank_state_template),
    ]

    return friendly_units, enemy_units


def run_with_sequence(sequence, visualize=False):
    """Run a single simulation using the provided attack sequence."""
    friendly_units, enemy_units = _build_units()
    sim = Simulation(friendly_units, enemy_units, visualize=visualize, plan_name="HTN_V3_Sim")
    sim.attack_sequence = sequence

    for unit in friendly_units:
        unit.sim = sim

    return sim.run(max_steps=300)


def main():
    friendly_units, enemy_units = _build_units()

    friendly_names = [u.name for u in friendly_units]
    enemy_names = [e.name for e in enemy_units]

    best = None
    best_seq = None

    # Limit combinations to keep runtime manageable; adjust as needed
    MAX_COMBOS = 10
    for i, seq in enumerate(generate_attack_sequences(friendly_names, enemy_names)):
        if i >= MAX_COMBOS:
            break
        result = run_with_sequence(seq, visualize=False)
        score = result['score']
        if best is None or score > best['score']:
            best = result
            best_seq = seq

    # Run the best sequence again with visualization so the user can observe it
    final_result = run_with_sequence(best_seq, visualize=True)

    logger.info("\n=== Best Sequence Evaluation ===")
    logger.info(f"Attack sequence: {best_seq}")
    logger.info(f"Score: {final_result['score']:.1f}")
    logger.info(f"Total Friendly Health Remaining: {final_result['health']}")
    logger.info(f"Enemy Health Remaining: {final_result['enemy_health']}")
    logger.info(f"Outpost Secured: {final_result['outpost_secured']}")
    logger.info(f"Steps Taken: {final_result['steps_taken']}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
