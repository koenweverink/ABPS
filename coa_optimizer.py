import random
from simulation import Simulation
from units.friendly_units import FriendlyTank, FriendlyInfantry, FriendlyArtillery, FriendlyAntiTank
from units.enemy_units import EnemyTank, EnemyInfantry, EnemyArtillery, EnemyAntiTank
from state_templates import (
    tank_state_template, infantry_state_template, artillery_state_template, anti_tank_state_template,
    enemy_tank_state_template, enemy_infantry_state_template, enemy_artillery_state_template, enemy_anti_tank_state_template
)
from domains import secure_outpost_domain, enemy_domain
from unit_factory import make_enemy, make_friendly

# Define available friendly unit types and their templates
FRIENDLY_TEMPLATES = [
    (FriendlyTank, tank_state_template),
    (FriendlyInfantry, infantry_state_template),
    (FriendlyArtillery, artillery_state_template),
    (FriendlyAntiTank, anti_tank_state_template),
]

MAX_UNITS = 6  # constraint on total friendly units per COA

def generate_random_coa(enemy_units):
    """
    Randomly assign friendly units to enemy units.
    Returns a list of (enemy_name, FriendlyUnitClass, template, count) tuples.
    """
    coa = []
    unit_budget = MAX_UNITS
    for enemy in enemy_units:
        num_units = random.randint(1, min(3, unit_budget))
        unit_type = random.choice(FRIENDLY_TEMPLATES)
        coa.append((enemy.state['name'], unit_type[0], unit_type[1], num_units))
        unit_budget -= num_units
        if unit_budget <= 0:
            break
    return coa

def build_friendly_units_from_coa(coa):
    """Construct a friendly unit list based on the COA plan."""
    friendly_units = []
    idx = 0
    for enemy_name, cls, template, count in coa:
        for _ in range(count):
            name = f"{cls.__name__}_{idx}"
            unit = make_friendly(name, cls, template, secure_outpost_domain, tank_state_template)  # assume tank as default enemy
            friendly_units.append(unit)
            idx += 1
    return friendly_units

def generate_fixed_enemy_units():
    """Generate a fixed set of enemies for consistency across COAs."""
    return [
        make_enemy("EnemyTankGroup1", EnemyTank, enemy_tank_state_template, position=(63, 15), domain=enemy_domain),
        make_enemy("EnemyInfantryGroup1", EnemyInfantry, enemy_infantry_state_template, position=(45, 25), domain=enemy_domain),
        make_enemy("EnemyArtilleryGroup1", EnemyArtillery, enemy_artillery_state_template, position=(50, 40), domain=enemy_domain),
    ]

def run_coa_simulation(coa, enemy_units):
    """Build friendly units from COA, run simulation, and return results."""
    friendly_units = build_friendly_units_from_coa(coa)
    sim = Simulation(friendly_units, enemy_units, visualize=False)
    result = sim.run(max_steps=300)
    return result

def search_best_coa(num_trials=100):
    """
    Perform random COA search over multiple trials.
    Returns sorted list of (COA, result) tuples.
    """
    results = []
    for trial in range(num_trials):
        print(f"\n[Trial {trial+1}/{num_trials}] Generating and evaluating COA...")
        try:
            enemy_units = generate_fixed_enemy_units()
            coa = generate_random_coa(enemy_units)
            result = run_coa_simulation(coa, enemy_units)

            print(f"  Score: {result['score']:.2f}, Health: {result['health']}, "
                  f"Enemy Health: {result['enemy_health']}, Steps: {result['steps_taken']}, "
                  f"Outpost Secured: {result['outpost_secured']}")
            for entry in coa:
                print(f"    Assigned {entry[3]} x {entry[1].__name__} to {entry[0]}")
            results.append((coa, result))
        except Exception as e:
            print(f"  Simulation failed: {e}")
    if not results:
        return []
    return sorted(results, key=lambda x: -x[1]['score'])

if __name__ == "__main__":
    top_coas = search_best_coa(10)
    if not top_coas:
        print("No valid COAs to display.")
    else:
        for i, (coa, res) in enumerate(top_coas[:5]):
            print(f"\n=== COA {i+1} ===")
            for entry in coa:
                print(f"Assign {entry[3]} x {entry[1].__name__} to {entry[0]}")
            print(f"Score: {res['score']:.2f}, Health: {res['health']}, Enemies Remaining: {res['enemy_health']}, Steps: {res['steps_taken']}, Outpost Secured: {res['outpost_secured']}")
