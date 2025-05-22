import matplotlib.pyplot as plt

from units.enemy_units import *
from units.friendly_units import *
from state_templates import *
from domains import enemy_domain, secure_outpost_domain
from simulation import Simulation
from unit_factory import make_enemy, make_friendly
from log import logger


###############################
# Main Simulation Setup
###############################

def main():
    """
    Initializes enemy and friendly units, sets up the simulation,
    and runs the scenario until completion. Logs evaluation metrics.
    """
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

    sim = Simulation(friendly_units, enemy_units, visualize=True, plan_name="Mode1_Test_Grouped_Dynamic_Partial")

    for unit in friendly_units:
        unit.sim = sim

    result = sim.run(max_steps=300)
    logger.info("\n=== Plan Evaluation ===")
    logger.info(f"Score: {result['score']:.1f}")
    logger.info(f"Total Friendly Health Remaining: {result['health']}")
    logger.info(f"Enemy Health Remaining: {result['enemy_health']}")
    logger.info(f"Outpost Secured: {result['outpost_secured']}")
    logger.info(f"Steps Taken: {result['steps_taken']}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
