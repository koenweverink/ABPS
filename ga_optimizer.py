import random
import itertools
from tqdm import tqdm
from simulation import Simulation
from unit_factory import make_enemy, make_friendly
from units.enemy_units import *
from units.friendly_units import *
from state_templates import *
from domains import enemy_domain, secure_outpost_domain
from log import logger

# Define friendly templates and cost per unit type
FRIENDLY_TEMPLATES = [
    ("FriendlyTankGroup", FriendlyTank, tank_state_template),
    ("FriendlyInfantryGroup", FriendlyInfantry, infantry_state_template),
    ("FriendlyArtilleryGroup", FriendlyArtillery, artillery_state_template),
    ("FriendlyAntiTankGroup", FriendlyAntiTank, anti_tank_state_template),
]
COST_MAP = {
    "FriendlyTankGroup": 5,
    "FriendlyInfantryGroup": 2,
    "FriendlyArtilleryGroup": 4,
    "FriendlyAntiTankGroup": 3,
}

# Build fixed enemy units
def _build_enemy_units():
    return [
        make_enemy("EnemyTankGroup1", EnemyTank, enemy_tank_state_template, position=(63, 15), domain=enemy_domain),
        make_enemy("EnemyTankGroup2", EnemyTank, enemy_tank_state_template, position=(17, 5),  domain=enemy_domain),
        make_enemy("EnemyInfantryGroup1", EnemyInfantry, enemy_infantry_state_template, domain=enemy_domain),
        make_enemy("EnemyAntiTankGroup1", EnemyAntiTank, enemy_anti_tank_state_template, domain=enemy_domain),
        make_enemy("EnemyArtilleryGroup1", EnemyArtillery, enemy_artillery_state_template, domain=enemy_domain),
    ]

# Build friendly units from config dict {type_name: count}
def build_friendly_units(config):
    units = []
    for name, cls, template in FRIENDLY_TEMPLATES:
        count = config.get(name, 0)
        for i in range(count):
            unit_name = f"{name}_{i}"
            units.append(make_friendly(unit_name, cls, template, secure_outpost_domain, enemy_tank_state_template))
    return units

# Generate random config within bounds
def random_config(min_counts, max_counts):
    return {name: random.randint(min_counts.get(name, 0), max_counts.get(name, 0))
            for name, _, _ in FRIENDLY_TEMPLATES}

# One random chromosome: (config, sequence)
def random_chromosome(enemy_names, min_size, min_counts, max_counts):
    config = random_config(min_counts, max_counts)
    friendly_units = build_friendly_units(config)
    friendly_names = [u.name for u in friendly_units]
    order = enemy_names[:]
    random.shuffle(order)
    seq = []
    for enemy in order:
        if friendly_names:
            max_group = len(friendly_names)
            k = max_group if max_group < min_size else random.randint(min_size, max_group)
        else:
            k = 0
        team = random.sample(friendly_names, k) if k > 0 else []
        seq.append((enemy, team))
    return (config, seq)

# Evaluate: returns (fitness, raw_score, cost)
def evaluate(chromosome, cost_weight=1.0):
    config, seq = chromosome
    friendly_units = build_friendly_units(config)
    enemy_units = _build_enemy_units()
    sim = Simulation(friendly_units, enemy_units, visualize=False, plan_name="HTN_V3_Sim")
    sim.attack_sequence = seq
    for u in friendly_units:
        u.sim = sim
    result = sim.run(max_steps=300)
    raw_score = result["score"]
    cost = sum(config[name] * COST_MAP.get(name, 0) for name in config)
    fitness = raw_score - cost_weight * cost
    return fitness, raw_score, cost

# Order-based crossover for (config, seq)
def crossover(a, b):
    cfg_a, seq_a = a
    cfg_b, seq_b = b
    child_cfg = {name: random.choice([cfg_a.get(name, 0), cfg_b.get(name, 0)])
                 for name, _, _ in FRIENDLY_TEMPLATES}
    # seq crossover
    n = len(seq_a)
    cut = random.randrange(1, n) if n > 1 else 1
    front = seq_a[:cut]
    front_enemies = [e for e, _ in front]
    seq_map_b = {e: team for e, team in seq_b}
    child_seq = front.copy()
    for e, _ in seq_b:
        if e not in front_enemies:
            child_seq.append((e, seq_map_b[e]))
    return (child_cfg, child_seq)

# Mutation: adjust config counts and sequence assignments
def mutate(chromosome, min_counts, max_counts, min_size=2, p_mut=0.2):
    cfg, seq = chromosome
    for name in cfg:
        if random.random() < p_mut:
            low, high = min_counts.get(name, 0), max_counts.get(name, 0)
            cfg[name] = random.randint(low, high)
    friendly_units = build_friendly_units(cfg)
    friendly_names = [u.name for u in friendly_units]
    seq = [(e, team[:]) for e, team in seq]
    if random.random() < p_mut and len(seq) > 1:
        i, j = random.sample(range(len(seq)), 2)
        seq[i], seq[j] = seq[j], seq[i]
    for idx, (e, _) in enumerate(seq):
        if random.random() < p_mut and friendly_names:
            max_group = len(friendly_names)
            k = max_group if max_group < min_size else random.randint(min_size, max_group)
            seq[idx] = (e, random.sample(friendly_names, k))
    return (cfg, seq)

# Tournament selection

def tournament_selection(scored_population, tournament_size, select_count):
    selected = []
    for _ in range(select_count):
        contestants = random.sample(scored_population, tournament_size)
        winner = max(contestants, key=lambda x: x[0])[1]
        selected.append(winner)
    return selected

# GA optimizer with progress bar showing raw and fitness scores
def genetic_optimize(enemy_names,
                     pop_size=50, generations=20,
                     min_size=2, min_counts=None, max_counts=None,
                     cost_weight=1.0,
                     retain_frac=0.2, random_frac=0.1,
                     mutate_p=0.2, tournament_size=5):
    if min_counts is None:
        min_counts = {name: 0 for name,_,_ in FRIENDLY_TEMPLATES}
    if max_counts is None:
        max_counts = {name: 3 for name,_,_ in FRIENDLY_TEMPLATES}

    population = [random_chromosome(enemy_names, min_size, min_counts, max_counts)
                  for _ in range(pop_size)]
    best_overall_fitness = float('-inf')
    best_overall_raw = 0.0
    best_chrom = None

    gen_bar = tqdm(range(generations), desc="GA generations", position=0, leave=True)
    for gen in gen_bar:
        scored = []
        for chrom in tqdm(population, desc=f"Gen {gen} fitness", position=1, leave=False):
            fit, raw, cost = evaluate(chrom, cost_weight)
            scored.append((fit, raw, cost, chrom))
        scored.sort(key=lambda x: x[0], reverse=True)
        gen_best_fit, gen_best_raw, _, gen_best_chrom = scored[0]
        if gen_best_fit > best_overall_fitness:
            best_overall_fitness = gen_best_fit
            best_overall_raw = gen_best_raw
            best_chrom = gen_best_chrom
        gen_bar.set_postfix(best_fit=f"{best_overall_fitness:.1f}", best_raw=f"{best_overall_raw:.1f}")

        # Prepare for selection
        retain_len = int(pop_size * retain_frac)
        scored_for_sel = [(fit, chrom) for fit, raw, cost, chrom in scored]
        parents = tournament_selection(scored_for_sel, tournament_size, retain_len)
        for _ in range(int(pop_size * random_frac)):
            parents.append(random_chromosome(enemy_names, min_size, min_counts, max_counts))

        # Crossover & mutation
        children = []
        while len(children) < pop_size - len(parents):
            a, b = random.sample(parents, 2)
            child = crossover(a, b)
            child = mutate(child, min_counts, max_counts, min_size, p_mut=mutate_p)
            children.append(child)
        population = parents + children

    # Final evaluation (fitness only)
    final_scored = []
    for chrom in tqdm(population, desc="Final eval", position=0, leave=True):
        fit, raw, cost = evaluate(chrom, cost_weight)
        final_scored.append((fit, raw, cost, chrom))
    final_scored.sort(key=lambda x: x[0], reverse=True)
    best_fit, _, _, best_chrom = final_scored[0]
    return best_fit, best_chrom

if __name__ == "__main__":
    enemies = [e.name for e in _build_enemy_units()]
    fit, (config, seq) = genetic_optimize(
        enemies,
        pop_size=50,
        generations=20,
        min_size=2,
        min_counts={name:0 for name,_,_ in FRIENDLY_TEMPLATES},
        max_counts={name:3 for name,_,_ in FRIENDLY_TEMPLATES},
        cost_weight=0.5,
        retain_frac=0.2,
        random_frac=0.1,
        mutate_p=0.2,
        tournament_size=5
    )
    print("Best GA fitness:", fit)
    print("Config:", config)
    print("Sequence:", seq)
