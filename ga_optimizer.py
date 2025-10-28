# ga_optimizer.py
# GA + NSGA-II with correctness, robustness, and performance:
# - Persistent process pool (Windows-safe)
# - Memoization (LRU) + pre-dedup + early clone rejection
# - Explicit elitism, adaptive mutation, smarter operators, type-aware teams
# - Worker stdout/stderr redirected to file; console shows clean tqdm bars
# - RICH RESULTS: best_overall, best_final_gen, pareto_front, history
# - CHECKPOINTING: save population + RNG every few generations; resume support

import random
import statistics
from tqdm import tqdm
import concurrent.futures
import contextlib
import sys
import os
import pickle
from datetime import datetime
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple, Optional

from simulation import Simulation
from unit_factory import make_enemy, make_friendly
from units.enemy_units import EnemyAntiTank, EnemyArtillery, EnemyInfantry, EnemyTank
from units.friendly_units import (
    FriendlyTank,
    FriendlyInfantry,
    FriendlyArtillery,
    FriendlyAntiTank,
)
from state_templates import (
    enemy_anti_tank_state_template,
    enemy_artillery_state_template,
    tank_state_template,
    infantry_state_template,
    artillery_state_template,
    anti_tank_state_template,
    enemy_tank_state_template,
    enemy_infantry_state_template,
)
from domains import enemy_domain, secure_outpost_domain
from log import logger
from config import GRID_WIDTH


# ===== Friendly templates and costs =====

@dataclass(frozen=True)
class FriendlyTemplate:
    """Meta-data for a friendly unit template."""

    name: str
    cls: type
    template: dict
    cost: int


FRIENDLY_TEMPLATES: List[FriendlyTemplate] = [
    FriendlyTemplate("FriendlyTankGroup", FriendlyTank, tank_state_template, 5),
    FriendlyTemplate("FriendlyInfantryGroup", FriendlyInfantry, infantry_state_template, 2),
    FriendlyTemplate("FriendlyArtilleryGroup", FriendlyArtillery, artillery_state_template, 4),
    FriendlyTemplate("FriendlyAntiTankGroup", FriendlyAntiTank, anti_tank_state_template, 3),
]
FRIENDLY_TEMPLATE_MAP: Dict[str, FriendlyTemplate] = {t.name: t for t in FRIENDLY_TEMPLATES}
# Map of friendly template names to their unit costs

COST_MAP: Dict[str, int] = {t.name: t.cost for t in FRIENDLY_TEMPLATES}


@dataclass
class Chromosome:
    """Configuration of friendly units and attack sequence."""

    config: Dict[str, int]
    sequence: List[Tuple[str, str, List[str]]]

    def __iter__(self):  # allow unpacking like a tuple
        yield self.config
        yield self.sequence


@dataclass
class Evaluation:
    """Result of evaluating a chromosome."""

    mean_raw: float
    std_raw: float
    adjusted_raw: float
    cost: float
    seed_base: int
    seeds: List[int]

    def __getitem__(self, item):  # for backward compatibility
        return getattr(self, item)


# ===== Logging helpers (throttle noisy ops) =====

def _log_sampled_debug(msg: str, sample_rate: int, rng=None):
    """Log at DEBUG roughly once every `sample_rate` calls (>=1)."""
    rng = rng or random
    if sample_rate <= 1 or rng.randint(1, sample_rate) == 1:
        logger.debug(msg)


# ===== Enemy setup =====

def _build_enemy_units():
    units = [
        make_enemy(
            "EnemyInfantryGroup1",
            EnemyInfantry,
            enemy_infantry_state_template,
            position=(GRID_WIDTH - 1, 1),
            domain=enemy_domain,
        ),
        make_enemy(
            "EnemyInfantryGroup2",
            EnemyInfantry,
            enemy_infantry_state_template,
            position=(GRID_WIDTH - 1, 1),
            domain=enemy_domain,
        ),
        make_enemy(
            "EnemyAntiTankGroup1",
            EnemyAntiTank,
            enemy_anti_tank_state_template,
            position=(GRID_WIDTH - 1, 1),
            domain=enemy_domain,
        ),
        make_enemy(
            "EnemyArtilleryGroup1",
            EnemyArtillery,
            enemy_artillery_state_template,
            position=(GRID_WIDTH - 1, 1),
            domain=enemy_domain,
        ),
    ]
    
    for u in units:
        u.state["mission"] = "DelayMission"
    return units


# ===== Utilities =====

def build_friendly_units(config: Dict[str, int], enemy_template: Optional[dict] = None):
    """Instantiate friendly units from a config mapping.

    Parameters
    ----------
    config : dict
        Mapping of unit-template names to counts.
    enemy_template : dict or None
        Template describing a default enemy.  If ``None``,
        ``enemy_tank_state_template`` is used.
    """
    enemy_template = enemy_template or enemy_tank_state_template
    units = []
    for tmpl in FRIENDLY_TEMPLATES:
        count = int(config.get(tmpl.name, 0))
        for i in range(count):
            unit_name = f"{tmpl.name}_{i}"
            units.append(
                make_friendly(
                    unit_name,
                    tmpl.cls,
                    tmpl.template,
                    secure_outpost_domain,
                    enemy_template,
                )
            )
    return units

def clamp_counts(cfg: Dict[str, int], min_counts: Dict[str, int], max_counts: Dict[str, int]) -> Dict[str, int]:
    """Clamp configuration counts to be within min/max bounds."""
    for tmpl in FRIENDLY_TEMPLATES:
        lo = int(min_counts.get(tmpl.name, 0))
        hi = int(max_counts.get(tmpl.name, 0))
        cfg[tmpl.name] = max(lo, min(hi, int(cfg.get(tmpl.name, 0))))
    return cfg

def friendly_names_from_config(cfg: Dict[str, int]) -> List[str]:
    """Return ordered friendly unit names from configuration counts."""
    names = []
    for tmpl in FRIENDLY_TEMPLATES:
        count = int(cfg.get(tmpl.name, 0))
        names.extend(f"{tmpl.name}_{i}" for i in range(count))
    return names

def friendly_name_set_from_config(cfg: Dict[str, int]) -> set:
    return set(friendly_names_from_config(cfg))

def enemy_type_hint(enemy_name: str) -> str:
    n = enemy_name.lower()
    if "tank" in n: return "Tank"
    if "artillery" in n: return "Artillery"
    if "antitank" in n or "anti_tank" in n or "anti-tank" in n: return "AntiTank"
    if "infantry" in n: return "Infantry"
    return "Unknown"

def unit_type_from_member(member_name: str) -> str:
    return member_name.split("_")[0] if "_" in member_name else member_name

def weighted_sample_without_replacement(items: Iterable, weights: Iterable[float], k: int, rng=None):
    """Pick ``k`` items without replacement using the provided ``weights``.

    This is a deliberately simple implementation that prioritizes readability
    over raw speed.  Each draw recalculates the cumulative weights and selects
    a single item.  All weights must be non-negative.
    """
    rng = rng or random
    item_list = list(items)
    weight_list = list(weights)
    if len(item_list) != len(weight_list):
        raise ValueError("items and weights must be the same length")
    if any(w < 0 for w in weight_list):
        raise ValueError("weights must be non-negative")

    picks = []
    draws = min(k, len(item_list))
    for _ in range(draws):
        total_weight = sum(weight_list)
        if total_weight <= 0:
            index = rng.randrange(len(item_list))
        else:
            target = rng.random() * total_weight
            cumulative = 0.0
            index = 0
            for i, weight in enumerate(weight_list):
                cumulative += weight
                if cumulative >= target:
                    index = i
                    break
        picks.append(item_list.pop(index))
        weight_list.pop(index)
    return picks

def biased_team_sample(valid_names: Iterable[str], min_size: int, enemy_name: str, rng=None) -> List[str]:
    """Sample a team of friendly units biased by enemy type."""
    rng = rng or random
    valid = sorted(valid_names)
    if not valid:
        return []
    k = len(valid) if len(valid) < min_size else rng.randint(min_size, len(valid))
    etype = enemy_type_hint(enemy_name)
    weights: List[float] = []
    for m in valid:
        utype = unit_type_from_member(m)
        w = 1.0
        if etype == "Tank" and utype in ("FriendlyAntiTankGroup", "FriendlyTankGroup"):
            w *= 3.0
        elif etype == "Infantry" and utype in ("FriendlyInfantryGroup", "FriendlyArtilleryGroup"):
            w *= 2.0
        elif etype == "Artillery" and utype in ("FriendlyTankGroup", "FriendlyInfantryGroup"):
            w *= 1.5
        weights.append(w)
    return weighted_sample_without_replacement(valid, weights, k, rng=rng)


# ===== Chromosome + normalization (for caching & dedup) =====

def random_config(min_counts: Dict[str, int], max_counts: Dict[str, int], rng=None) -> Dict[str, int]:
    """Generate a random friendly unit configuration.

    Uses a straightforward loop with descriptive variable names to keep the
    code easy to read for newcomers.
    """
    rng = rng or random
    config: Dict[str, int] = {}
    for template in FRIENDLY_TEMPLATES:
        name = template.name
        minimum = int(min_counts.get(name, 0))
        maximum = int(max_counts.get(name, 0))
        config[name] = rng.randint(minimum, maximum)
    return config

def random_chromosome(enemy_names: List[str], min_size: int, min_counts: Dict[str, int], max_counts: Dict[str, int], bias_team: bool = True, rng=None) -> Chromosome:
    """Create a random chromosome with a casual, easy-to-follow style."""
    rng = rng or random

    # Randomly decide how many of each friendly unit we have available
    configuration = random_config(min_counts, max_counts, rng=rng)
    friendly_names = friendly_names_from_config(configuration)

    # Shuffle the enemy order so we get different attack sequences
    enemy_order = list(enemy_names)
    rng.shuffle(enemy_order)

    sequence: List[Tuple[str, str, List[str]]] = []
    for enemy in enemy_order:
        if not friendly_names:
            team_members: List[str] = []
        else:
            if bias_team:
                team_members = biased_team_sample(friendly_names, min_size, enemy, rng=rng)
            else:
                max_group_size = len(friendly_names)
                group_size = max_group_size if max_group_size < min_size else rng.randint(min_size, max_group_size)
                team_members = rng.sample(friendly_names, group_size)
        attack_type = rng.choice(["ConsolidateAttack", "FlankAttack"])
        sequence.append((enemy, attack_type, team_members))

    return Chromosome(configuration, sequence)

def normalize_key(chromosome):
    """Canonical, hashable key for memoization & dedup."""
    cfg, seq = chromosome
    cfg_key = tuple((tmpl.name, int(cfg.get(tmpl.name, 0))) for tmpl in FRIENDLY_TEMPLATES)  # fixed order
    seq_key = tuple((e, t, tuple(sorted(team))) for (e, t, team) in seq)
    return (cfg_key, seq_key)


# ===== Repair / Validate =====

def repair(
    chromosome: Chromosome,
    min_counts: Dict[str, int],
    max_counts: Dict[str, int],
    min_size: int,
    rng=None,
) -> Chromosome:
    """
    Make a chromosome feasible by:
      1) clamping per-template counts into [min_counts, max_counts],
      2) dropping team members not present in the (clamped) roster,
      3) if possible, topping up undersized teams to at least `min_size`.
    """
    rng = rng or random

    # Unpack and clamp counts to the allowed range.
    cfg, seq = chromosome
    cfg = clamp_counts(dict(cfg), min_counts, max_counts)

    # Compute the valid roster from the clamped config.
    valid = friendly_name_set_from_config(cfg)
    sorted_valid = sorted(valid)  # stable ordering used for deterministic fills

    fixed_seq = []
    for enemy, t, team in seq:
        # Keep only members that exist in the valid roster.
        team = [m for m in team if m in valid]

        # If we have a non-empty roster and the team is too small, top it up.
        if valid and len(team) < min_size:
            k = min(min_size, len(valid))  # cannot exceed roster size
            # If k equals the roster size, take all (deterministic).
            # Otherwise, sample k distinct members (non-deterministic).
            team = (
                sorted_valid[:k]
                if k == len(valid)
                else rng.sample(sorted_valid, k)
            )

        fixed_seq.append((enemy, t, team))

    return Chromosome(cfg, fixed_seq)


def validate(
    chromosome: Chromosome,
    enemy_names: List[str],
    min_counts: Dict[str, int],
    max_counts: Dict[str, int],
    min_size: int,
) -> bool:
    """
    Validate a chromosome against:
      1) count bounds per friendly template,
      2) enemy sequence (must be a permutation of `enemy_names` with no extras/dupes),
      3) team membership (all members must be in roster),
      4) minimum team size, *but only if* the roster has at least `min_size` members.

    Returns
    -------
    bool
        True if valid; False and logs an error otherwise.
    """
    cfg, seq = chromosome

    # Enforce a minimum total roster size (global)
    total_units = sum(int(cfg.get(t.name, 0)) for t in FRIENDLY_TEMPLATES)
    if total_units < min_size:
        logger.error(f"[VALIDATE] Total roster {total_units} < min_size={min_size}")
        return False

    # (1) Per-template count bounds.
    for template in FRIENDLY_TEMPLATES:
        name = template.name
        low = int(min_counts.get(name, 0))
        high = int(max_counts.get(name, 0))
        count = int(cfg.get(name, 0))
        if not (low <= count <= high):
            logger.error(f"[VALIDATE] Count out of bounds for {name}: {count} not in [{low},{high}]")
            return False

    # (2) Enemy sequence must match exactly (same set and same cardinality).
    seen_enemies = [enemy for enemy, _, _ in seq]
    if set(seen_enemies) != set(enemy_names) or len(seen_enemies) != len(enemy_names):
        logger.error(f"[VALIDATE] Enemy sequence invalid. Seen={seen_enemies}, Expected={list(enemy_names)}")
        return False

    # Build roster once from the (already-checked) config.
    roster = friendly_name_set_from_config(cfg)

    # (3) Membership + (4) min team size, conditionally enforced.
    for enemy, _, team in seq:
        # Every member must exist in the roster.
        for m in team:
            if m not in roster:
                logger.error(f"[VALIDATE] Team member {m} not in roster")
                return False

        # Enforce min team size only when it is actually feasible.
        if roster and len(roster) >= min_size and len(team) < min_size:
            logger.error(f"[VALIDATE] Team for {enemy} smaller than min_size={min_size}")
            return False

    return True


"""New scoring pipeline utilities."""

# ===== Evaluation (stochastic robustness) =====

def compute_raw_score(result: dict, wd: float = 0.5, ws: float = 0.5) -> float:
    """
    Returns raw in [0,1]. damage_frac = 1 - enemy_hp_final/enemy_hp_initial;
    survival_frac = friendly_hp_final/friendly_hp_initial; clamp to [0,1].
    raw = wd*damage_frac + ws*survival_frac (renormalize wd,ws to sum 1; handle missing keys by falling back to normalized proxies or result.get('score',0.0) clipped to [0,1]).
    No time/cost here.
    """
    wd = max(0.0, wd)
    ws = max(0.0, ws)

    fracs = []
    weights = []

    enemy_initial_health = result.get("enemy_hp_initial")
    enemy_final_health = result.get("enemy_hp_final")

    if enemy_initial_health and enemy_final_health is not None and enemy_initial_health > 0:
        dmg = 1.0 - float(enemy_final_health) / float(enemy_initial_health)
        fracs.append(max(0.0, min(1.0, dmg)))
        weights.append(wd)

    friendly_initial_health = result.get("friendly_hp_initial")
    friendly_final_health = result.get("friendly_hp_final")

    if friendly_initial_health and friendly_final_health is not None and friendly_initial_health > 0:
        surv = float(friendly_final_health) / float(friendly_initial_health)
        fracs.append(max(0.0, min(1.0, surv)))
        weights.append(ws)

    if fracs:
        total = sum(weights) or 1.0
        weights = [w / total for w in weights]
        raw = sum(w * f for w, f in zip(weights, fracs))
        return float(max(0.0, min(1.0, raw)))

    proxy = result.get("score", 0.0)
    try:
        proxy = float(proxy)
    except Exception:
        proxy = 0.0
    return float(max(0.0, min(1.0, proxy)))


def _single_run_metrics(config: Dict[str, int], seq, max_steps: int = 300, seed: Optional[int] = None, rng=None):
    """Run a single simulation returning (raw, success, steps)."""
    rng = random.Random(seed) if seed is not None else (rng or random)
    friendly_units = build_friendly_units(config, enemy_template=None)
    enemy_units = _build_enemy_units()
    sim = Simulation(enemy_units=enemy_units, friendly_units=friendly_units, visualize=False, plan_name="HTN_V3_Sim")
    if seed is not None:
        try:
            if hasattr(sim, "set_seed") and callable(sim.set_seed):
                sim.set_seed(seed)
            elif hasattr(sim, "seed"):
                sim.seed = seed
            if hasattr(sim, "rng"):
                sim.rng = rng
        except Exception:
            pass
    sim.attack_sequence = seq
    for u in friendly_units:
        u.sim = sim

    friendly_initial = sum(u.state.get("max_health", 0) for u in friendly_units)
    enemy_initial = sum(e.state.get("max_health", 0) for e in enemy_units)

    result = sim.run(max_steps=max_steps)

    friendly_final = result.get("health")
    if friendly_final is None:
        friendly_final = sum(u.state.get("health", 0) for u in sim.friendly_units)
    enemy_final = result.get("enemy_health")
    if enemy_final is None:
        enemy_final = sum(e.state.get("health", 0) for e in sim.enemy_units)

    enriched = dict(result)
    enriched.setdefault("friendly_hp_initial", friendly_initial)
    enriched.setdefault("enemy_hp_initial", enemy_initial)
    enriched.setdefault("friendly_hp_final", friendly_final)
    enriched.setdefault("enemy_hp_final", enemy_final)

    steps = result.get("steps") or result.get("steps_taken") or result.get("t") or result.get("time") or result.get("num_steps")
    if steps is None and hasattr(sim, "step_count"):
        steps = sim.step_count
    enriched.setdefault("steps", steps)

    raw = compute_raw_score(enriched)
    success = result.get("success", None)
    steps_int = int(steps) if steps is not None else None
    return float(raw), (bool(success) if isinstance(success, bool) else None), steps_int


def evaluate_multi(
    chromosome: Chromosome,
    num_eval_seeds: int = 3,
    robust_lambda: float = 0.0,
    seed_base: int = 0,
    max_steps: int = 300,
    time_weight: float = 0.0,
    max_counts: Optional[Dict[str, int]] = None,
    rng=None,
) -> Evaluation:
    """Evaluate a chromosome across multiple seeds.

    Raw∈[0,1], Adj∈[0,1], FinalScore∈[0,100)."""
    cfg, seq = chromosome
    raws = []
    shapeds = []
    seeds: List[int] = []
    for j in range(num_eval_seeds):
        seed = seed_base + 9973 * j
        seeds.append(seed)
        raw, _success, steps = _single_run_metrics(cfg, seq, max_steps=max_steps, seed=seed, rng=rng)
        raws.append(raw)
        step_ratio = min(1.0, (steps or max_steps) / max_steps) if max_steps > 0 else 0.0
        shapeds.append(raw - time_weight * step_ratio)

    mean_raw = statistics.fmean(raws) if raws else 0.0
    std_raw = statistics.pstdev(raws) if len(raws) > 1 else 0.0
    mean_shaped = statistics.fmean(shapeds) if shapeds else 0.0
    std_shaped = statistics.pstdev(shapeds) if len(shapeds) > 1 else 0.0
    adj = mean_shaped - robust_lambda * std_shaped
    adj = max(0.0, min(1.0, adj))

    cost = sum(int(cfg.get(tmpl.name, 0)) * tmpl.cost for tmpl in FRIENDLY_TEMPLATES)
    if max_counts is None:
        max_counts = {}
    cost_max = max(1, sum(int(max_counts.get(tmpl.name, 0)) * tmpl.cost for tmpl in FRIENDLY_TEMPLATES))
    cost_norm = min(1.0, cost / cost_max)
    final_score = 100.0 * adj * (1.0 - cost_norm)

    ev = Evaluation(mean_raw, std_raw, adj, float(cost), int(seed_base), seeds)
    ev.final_score = final_score
    return ev


def fitness_single_objective(ev: Evaluation, cost_weight: float = 1.0) -> float:
    """Return scalar fitness for single-objective mode (FinalScore0_100)."""
    return getattr(ev, "final_score", 100.0 * ev.adjusted_raw)


# ===== Genetic operators (smarter, throttled logs) =====

def crossover_order_then_uniform_teams(a: Chromosome, b: Chromosome, log_sample_rate: int = 20, rng=None) -> Chromosome:
    """Mix two parents to create a child chromosome.

    The process is intentionally written step by step so the logic is easy to
    follow.  Configuration values are chosen per-template from either parent.
    The attack sequence begins with a slice from parent ``a`` and then appends
    any enemies not yet seen from parent ``b``.  For each enemy we randomly
    choose which parent's team/attack details to keep.
    """

    rng = rng or random

    config_a, sequence_a = a
    config_b, sequence_b = b

    # Pick the unit counts for the child one template at a time
    child_config: Dict[str, int] = {}
    for template in FRIENDLY_TEMPLATES:
        name = template.name
        count_a = int(config_a.get(name, 0))
        count_b = int(config_b.get(name, 0))
        child_config[name] = rng.choice([count_a, count_b])

    # Decide how much of parent A's order we keep at the front
    length_a = len(sequence_a)
    cut_point = rng.randrange(1, length_a) if length_a > 1 else 1
    child_sequence: List[Tuple[str, str, List[str]]] = []

    # Start with a slice from parent A
    for idx in range(cut_point):
        enemy, attack, team = sequence_a[idx]
        child_sequence.append((enemy, attack, list(team)))

    # Record which enemies we've already added so far
    existing_enemies = {enemy for enemy, _, _ in child_sequence}

    # Append remaining enemies from parent B
    for enemy, attack, team in sequence_b:
        if enemy not in existing_enemies:
            child_sequence.append((enemy, attack, list(team)))
            existing_enemies.add(enemy)

    # For each enemy, randomly pick team/attack info from either parent
    final_sequence: List[Tuple[str, str, List[str]]] = []
    for enemy, _, _ in child_sequence:
        # Look up the enemy's entry in both parents
        entry_a = next((x for x in sequence_a if x[0] == enemy), None)
        entry_b = next((x for x in sequence_b if x[0] == enemy), None)

        # Default values in case neither parent has information
        attack_choice = "ConsolidateAttack"
        team_choice: List[str] = []

        if rng.random() < 0.5 and entry_a is not None:
            attack_choice = entry_a[1]
            team_choice = list(entry_a[2])
        elif entry_b is not None:
            attack_choice = entry_b[1]
            team_choice = list(entry_b[2])

        final_sequence.append((enemy, attack_choice, team_choice))

    _log_sampled_debug(
        f"[CROSSOVER] Cut={cut_point}, ChildCfg={child_config}",
        log_sample_rate,
        rng=rng,
    )
    return Chromosome(child_config, final_sequence)


def mutate_block_shift(sequence: List[Tuple[str, str, List[str]]], rng=None) -> List[Tuple[str, str, List[str]]]:
    """Move a random contiguous block of the sequence to a new position."""

    rng = rng or random

    # Need at least three elements to meaningfully move a block
    if len(sequence) <= 2:
        return sequence

    # Choose the start and end of the block to move
    indices = list(range(len(sequence)))
    start_idx, end_idx = sorted(rng.sample(indices, 2))

    if start_idx == end_idx:
        return sequence

    # Extract the block
    block: List[Tuple[str, str, List[str]]] = []
    for i in range(start_idx, end_idx + 1):
        block.append(sequence[i])

    # Remaining elements after removing the block
    remainder: List[Tuple[str, str, List[str]]] = []
    for i, item in enumerate(sequence):
        if i < start_idx or i > end_idx:
            remainder.append(item)

    # Insert the block at a new random position within the remainder
    insert_pos = rng.randrange(len(remainder) + 1)
    new_sequence = remainder[:insert_pos] + block + remainder[insert_pos:]
    return new_sequence


def mutate(
    chromosome: Chromosome,
    min_counts: Dict[str, int],
    max_counts: Dict[str, int],
    min_size: int = 2,
    p_mut: float = 0.2,
    block_shift_prob: float = 0.5,
    bias_team: bool = True,
    log_sample_rate: int = 20,
    rng=None,
) -> Chromosome:
    """Randomly tweak a chromosome using plain, easy-to-read steps."""

    rng = rng or random

    # Work on copies so the original chromosome is unchanged
    config, sequence = chromosome
    config = dict(config)
    sequence = [(e, t, list(team)) for e, t, team in sequence]

    # === Mutate friendly unit counts ===
    for name in config:
        if rng.random() < p_mut:
            low = int(min_counts.get(name, 0))
            high = int(max_counts.get(name, 0))
            config[name] = rng.randint(low, high)

    # === Mutate the order of enemy attacks ===
    if len(sequence) > 1 and rng.random() < p_mut:
        if rng.random() < block_shift_prob:
            sequence = mutate_block_shift(sequence, rng=rng)
        else:
            i, j = rng.sample(range(len(sequence)), 2)
            sequence[i], sequence[j] = sequence[j], sequence[i]

    # === Mutate team compositions and attack types ===
    friendly_names = friendly_names_from_config(config)
    for idx, (enemy, attack, team) in enumerate(sequence):
        # Possibly swap out the team members attacking this enemy
        if rng.random() < p_mut and friendly_names:
            if bias_team:
                new_team = biased_team_sample(friendly_names, min_size, enemy, rng=rng)
            else:
                max_group = len(friendly_names)
                team_size = max_group if max_group < min_size else rng.randint(min_size, max_group)
                new_team = rng.sample(friendly_names, team_size)
            sequence[idx] = (enemy, attack, new_team)

        # Possibly flip the attack type
        if rng.random() < p_mut:
            new_attack = rng.choice(["ConsolidateAttack", "FlankAttack"])
            sequence[idx] = (enemy, new_attack, sequence[idx][2])

    _log_sampled_debug(
        f"[MUTATE] After counts: {config}",
        log_sample_rate,
        rng=rng,
    )
    return Chromosome(config, sequence)


# ===== Selection helpers =====

def tournament_selection_single(scored_population, tournament_size, select_count, rng=None):
    rng = rng or random
    k = max(1, min(tournament_size, len(scored_population)))
    out = []
    for _ in range(select_count):
        contestants = rng.sample(scored_population, k)
        winner = max(contestants, key=lambda x: x[0])[1]
        out.append(winner)
    return out


# --- NSGA-II machinery ---

def dominates(candidate_a, candidate_b):
    """True if `candidate_a` is no worse on both goals and better on at least one."""

    adj_a, cost_a = candidate_a
    adj_b, cost_b = candidate_b
    not_worse = (adj_a >= adj_b) and (cost_a <= cost_b)
    strictly_better = (adj_a > adj_b) or (cost_a < cost_b)
    return not_worse and strictly_better


def fast_non_dominated_sort(objectives):
    """Split objective pairs into Pareto fronts (front[0] is the best)."""

    dominates_map = [set() for _ in objectives]
    domination_counts = [0 for _ in objectives]
    fronts = [[]]
    for p_idx in range(len(objectives)):
        for q_idx in range(len(objectives)):
            if p_idx == q_idx:
                continue
            if dominates(objectives[p_idx], objectives[q_idx]):
                dominates_map[p_idx].add(q_idx)
            elif dominates(objectives[q_idx], objectives[p_idx]):
                domination_counts[p_idx] += 1
        if domination_counts[p_idx] == 0:
            fronts[0].append(p_idx)
    i = 0
    while fronts[i]:
        next_front = []
        for p_idx in fronts[i]:
            for q_idx in dominates_map[p_idx]:
                domination_counts[q_idx] -= 1
                if domination_counts[q_idx] == 0:
                    next_front.append(q_idx)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts


def crowding_distance(front_indices, objectives):
    """Estimate how isolated each solution is within a single front."""

    if not front_indices:
        return {}
    distance = {idx: 0.0 for idx in front_indices}

    # adjusted_raw (maximize)
    ar_sorted = sorted(front_indices, key=lambda idx: objectives[idx][0], reverse=True)
    ar_values = [objectives[idx][0] for idx in ar_sorted]
    ar_min, ar_max = min(ar_values), max(ar_values)
    ar_range = (ar_max - ar_min) if ar_max != ar_min else 1.0
    distance[ar_sorted[0]] = float("inf")
    distance[ar_sorted[-1]] = float("inf")
    for k in range(1, len(ar_sorted) - 1):
        prev_val = objectives[ar_sorted[k - 1]][0]
        next_val = objectives[ar_sorted[k + 1]][0]
        distance[ar_sorted[k]] += (next_val - prev_val) / ar_range

    # cost (minimize)
    cost_sorted = sorted(front_indices, key=lambda idx: objectives[idx][1])
    cost_values = [objectives[idx][1] for idx in cost_sorted]
    cost_min, cost_max = min(cost_values), max(cost_values)
    cost_range = (cost_max - cost_min) if cost_max != cost_min else 1.0
    distance[cost_sorted[0]] = float("inf")
    distance[cost_sorted[-1]] = float("inf")
    for k in range(1, len(cost_sorted) - 1):
        prev_val = objectives[cost_sorted[k - 1]][1]
        next_val = objectives[cost_sorted[k + 1]][1]
        distance[cost_sorted[k]] += (prev_val - next_val) / cost_range
    return distance


def crowded_tournament_pick(pool_indices, ranks, crowding, k=2, rng=None):
    """Pick one index via crowded tournament selection."""

    rng = rng or random
    contenders = rng.sample(pool_indices, k)

    def sort_key(idx):
        return (ranks[idx], -crowding.get(idx, 0.0))

    return min(contenders, key=sort_key)


def nsga2_environmental_selection(population, objectives, pop_size):
    """Select the next population according to NSGA-II rules."""

    fronts = fast_non_dominated_sort(objectives)
    next_indices = []
    rank_map = {}
    for rank, front in enumerate(fronts):
        if len(next_indices) + len(front) <= pop_size:
            for idx in front:
                rank_map[idx] = rank
            next_indices.extend(front)
        else:
            cd = crowding_distance(front, objectives)
            front_sorted = sorted(front, key=lambda idx: cd[idx], reverse=True)
            remaining = pop_size - len(next_indices)
            for idx in front_sorted[:remaining]:
                rank_map[idx] = rank
            next_indices.extend(front_sorted[:remaining])
            break
    crowd_map = {}
    for rank, front in enumerate(fronts):
        intersect = [idx for idx in front if idx in next_indices]
        if intersect:
            crowd_map.update(crowding_distance(intersect, objectives))
    selected_pop = [population[i] for i in next_indices]
    selected_objs = [objectives[i] for i in next_indices]
    rank_subset = {i: rank_map[i] for i in next_indices}
    return selected_pop, selected_objs, rank_subset, crowd_map, fronts


# ===== IO redirection (workers → file) =====

@contextlib.contextmanager
def _suppress_io():
    with open(os.devnull, 'w') as _null:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _null, _null
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err

@contextlib.contextmanager
def _redirect_io(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = f, f
    try:
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        sys.stdout, sys.stderr = old_out, old_err
        f.close()


# ===== LRU memo cache =====

class _LRUCache:
    def __init__(self, capacity: int = 5000):
        self.capacity = max(1, capacity)
        self._od = OrderedDict()

    def get(self, key):
        if key in self._od:
            val = self._od.pop(key)
            self._od[key] = val
            return val
        return None

    def put(self, key, value):
        if key in self._od:
            self._od.pop(key)
        self._od[key] = value
        if len(self._od) > self.capacity:
            self._od.popitem(last=False)


# ===== Windows-safe top-level worker =====

def _eval_multi_star(args):
    """Top-level worker (pickleable on Windows)."""
    (
        chrom,
        num_eval_seeds,
        robust_lambda,
        time_weight,
        seed_base,
        max_steps,
        max_counts,
        quiet_workers,
        worker_log_file,
        rng_seed,
    ) = args
    rng = random.Random(rng_seed)
    if quiet_workers:
        context = _redirect_io(worker_log_file) if worker_log_file else _suppress_io()
        with context:
            print(f"\n==== Worker eval start {datetime.now().isoformat(timespec='seconds')} ====")
            print(f"seed_base={seed_base}, num_eval_seeds={num_eval_seeds}, max_steps={max_steps}")
            res = evaluate_multi(
                chrom,
                num_eval_seeds=num_eval_seeds,
                robust_lambda=robust_lambda,
                seed_base=seed_base,
                max_steps=max_steps,
                time_weight=time_weight,
                max_counts=max_counts,
                rng=rng,
            )
            print(
                f"---- Eval summary: adjusted_raw={res.adjusted_raw:.3f}, mean={res.mean_raw:.3f}, "
                f"std={res.std_raw:.3f}, cost={res.cost:.1f} ----"
            )
            print(f"==== Worker eval end {datetime.now().isoformat(timespec='seconds')} ====\n")
            return res
    else:
        return evaluate_multi(
            chrom,
            num_eval_seeds=num_eval_seeds,
            robust_lambda=robust_lambda,
            seed_base=seed_base,
            max_steps=max_steps,
            time_weight=time_weight,
            max_counts=max_counts,
            rng=rng,
        )


# ===== Evaluation driver with PERSISTENT POOL + MEMOIZATION =====

def _eval_population_with_cache(pop, ex, workers, num_eval_seeds, robust_lambda, time_weight, max_steps,
                                max_counts, quiet_workers, worker_log_file, cache: _LRUCache,
                                desc="Eval", leave=False, position=0, rng=None):
    """
    Evaluate a list of chromosomes using:
      - persistent executor `ex` (or sequential if workers<=1)
      - LRU cache to reuse prior results
      - pre-dedup within this batch to avoid duplicate dispatch
    Returns list of eval dicts aligned to `pop`.
    """
    # Build keys and check cache
    keys = [normalize_key(ch) for ch in pop]
    results = [None] * len(pop)

    # First fill from cache
    missing_indices = []
    for i, k in enumerate(keys):
        cached = cache.get(k)
        if cached is not None:
            results[i] = cached
        else:
            missing_indices.append(i)

    if not missing_indices:
        return results  # all cached

    # Pre-dedup within the missing set (avoid dispatching same key twice)
    seen_local = {}
    unique_jobs = []      # [(key, chrom, idx_rep)]
    index_map = {}        # key -> list of indices in `missing_indices`
    for idx in missing_indices:
        k = keys[idx]
        if k in seen_local:
            index_map[k].append(idx)
        else:
            seen_local[k] = True
            index_map[k] = [idx]
            unique_jobs.append((k, pop[idx], idx))

    # Prepare args for unique jobs
    rng = rng or random
    args_list = [
        (
            chrom,
            num_eval_seeds,
            robust_lambda,
            time_weight,
            rng.randrange(10_000_000),  # seed_base
            max_steps,
            max_counts,
            quiet_workers,
            worker_log_file,
            rng.randrange(10_000_000),  # rng seed
        )
        for _, chrom, _ in unique_jobs
    ]

    # Dispatch via persistent executor (or sequential)
    evals_unique = []
    if workers and workers > 1 and ex is not None:
        for res in tqdm(ex.map(_eval_multi_star, args_list), total=len(args_list),
                        desc=desc, leave=leave, position=position,
                        mininterval=0.02, smoothing=0.0, dynamic_ncols=True):
            evals_unique.append(res)
    else:
        for args in tqdm(args_list, total=len(args_list), desc=desc, leave=leave, position=position,
                         mininterval=0.02, smoothing=0.0, dynamic_ncols=True):
            evals_unique.append(_eval_multi_star(args))

    # Write unique results back to cache and to all corresponding indices
    for (k, _ch, _), ev in zip(unique_jobs, evals_unique):
        cache.put(k, ev)
        for idx in index_map[k]:
            results[idx] = ev

    return results


# ===== Checkpointing helpers =====

def _save_checkpoint(path, state: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(state, f)

def _load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _pack_best(gen, chrom, ev, fit, key):
    if chrom is None:
        return None
    cfg, seq = chrom
    return {
        "generation": gen,
        "fitness": fit,
        "eval": {**asdict(ev), "final_score": getattr(ev, "final_score", None)},
        "chromosome": {"config": cfg, "sequence": seq},
        "key": key,
    }


def _initialize_population(enemy_names, pop_size, min_size, min_counts, max_counts, bias_team, rng):
    """Create an initial population of chromosomes."""
    population: List[Chromosome] = []
    for _ in range(pop_size):
        chrom = random_chromosome(
            enemy_names, min_size, min_counts, max_counts, bias_team=bias_team, rng=rng
        )
        chrom = repair(chrom, min_counts, max_counts, min_size, rng=rng)
        if not validate(chrom, enemy_names, min_counts, max_counts, min_size):
            chrom = repair(
                random_chromosome(
                    enemy_names,
                    min_size,
                    min_counts,
                    max_counts,
                    bias_team=bias_team,
                    rng=rng,
                ),
                min_counts,
                max_counts,
                min_size,
                rng=rng,
            )
        population.append(chrom)
    return population


def _resume_from_checkpoint(resume_from, rng):
    ckpt = _load_checkpoint(resume_from)
    rng.setstate(ckpt["random_state"])
    logger.info(f"[RESUME] Resumed from {resume_from} at generation {ckpt['generation'] + 1}")
    return ckpt


def _maybe_checkpoint(gen, population, evals, cache, best_overall, history, rng, checkpoint_every, checkpoint_path, mode):
    if checkpoint_every and ((gen + 1) % checkpoint_every == 0):
        state = {
            "generation": gen,
            "population": population,
            "evals": evals,
            "cache": cache,
            "best_overall": best_overall,
            "history": history,
            "random_state": rng.getstate(),
            "mode": mode,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        _save_checkpoint(checkpoint_path, state)
        logger.info(f"[CKPT] Saved checkpoint at gen {gen} → {checkpoint_path}")


def _nsga2_generation_step(
    population,
    evals,
    gen,
    enemy_names,
    pop_size,
    elites_per_gen,
    random_frac,
    min_counts,
    max_counts,
    min_size,
    mutate_p,
    block_shift_mut_prob,
    bias_team,
    op_log_sample_rate,
    cost_weight,
    ex,
    workers,
    num_eval_seeds,
    robust_lambda,
    time_weight,
    max_steps,
    quiet_workers,
    worker_log_file,
    cache,
    global_best_scalar_fit,
    best_overall,
    best_path,
    rng,
): 
    # Prepare objective pairs and compute ranks/crowding for the full population
    objective_pairs = [(ev.adjusted_raw, ev.cost) for ev in evals]
    _, _, full_ranks, full_crowding, _ = nsga2_environmental_selection(
        population, objective_pairs, len(population)
    )
    parent_pool = list(range(len(population)))

    # Identify elites from the current Pareto front
    fronts = fast_non_dominated_sort(objective_pairs)
    elite_candidates = fronts[0] if fronts else []
    if elite_candidates:
        elite_crowding = crowding_distance(elite_candidates, objective_pairs)
        elite_indices = sorted(
            elite_candidates,
            key=lambda idx: elite_crowding[idx],
            reverse=True,
        )[: min(elites_per_gen, len(elite_candidates))]
        elites = [population[idx] for idx in elite_indices]
    else:
        elites = []

    required_children = pop_size - len(elites)
    children: List[Chromosome] = []
    existing_keys = {normalize_key(ch) for ch in elites}
    existing_keys.update(normalize_key(ch) for ch in population)
    max_attempts = required_children * 10 if required_children > 0 else 0
    attempts = 0

    # Generate children via crossover/mutation until we meet the quota
    while len(children) < required_children and attempts < max_attempts:
        attempts += 1
        parent_a_idx = crowded_tournament_pick(parent_pool, full_ranks, full_crowding, rng=rng)
        parent_b_idx = crowded_tournament_pick(parent_pool, full_ranks, full_crowding, rng=rng)
        if parent_a_idx == parent_b_idx:
            continue
        parent_a, parent_b = population[parent_a_idx], population[parent_b_idx]
        child = crossover_order_then_uniform_teams(parent_a, parent_b, log_sample_rate=op_log_sample_rate, rng=rng)
        child = repair(child, min_counts, max_counts, min_size, rng=rng)
        child = mutate(
            child,
            min_counts,
            max_counts,
            min_size,
            p_mut=mutate_p,
            block_shift_prob=block_shift_mut_prob,
            bias_team=bias_team,
            log_sample_rate=op_log_sample_rate,
            rng=rng,
        )
        child = repair(child, min_counts, max_counts, min_size, rng=rng)
        if not validate(child, enemy_names, min_counts, max_counts, min_size):
            continue
        if rng.random() < random_frac:
            child = repair(
                random_chromosome(enemy_names, min_size, min_counts, max_counts, bias_team=bias_team, rng=rng),
                min_counts,
                max_counts,
                min_size,
                rng=rng,
            )
            if not validate(child, enemy_names, min_counts, max_counts, min_size):
                continue
        child_key = normalize_key(child)
        if child_key in existing_keys:
            continue
        existing_keys.add(child_key)
        children.append(child)

    # If we still need children, fill the rest with random valid ones
    if len(children) < required_children:
        while len(children) < required_children:
            random_child = repair(
                random_chromosome(enemy_names, min_size, min_counts, max_counts, bias_team=bias_team, rng=rng),
                min_counts,
                max_counts,
                min_size,
                rng=rng,
            )
            if not validate(random_child, enemy_names, min_counts, max_counts, min_size):
                continue
            child_key = normalize_key(random_child)
            if child_key in existing_keys:
                continue
            existing_keys.add(child_key)
            children.append(random_child)

    # Evaluate new children and merge with existing population
    child_evals = _eval_population_with_cache(
        children,
        ex,
        workers,
        num_eval_seeds,
        robust_lambda,
        time_weight,
        max_steps,
        max_counts,
        quiet_workers,
        worker_log_file,
        cache,
        desc=f"Gen {gen} eval (children)",
        leave=False,
        rng=rng,
    )
    combined = population + children
    combined_evals = evals + child_evals
    combined_objectives = [(ev.adjusted_raw, ev.cost) for ev in combined_evals]
    population, _, ranks, crowd, fronts_combined = nsga2_environmental_selection(
        combined, combined_objectives, pop_size
    )
    eval_map = {id(ch): ev for ch, ev in zip(combined, combined_evals)}
    evals = [eval_map[id(ch)] for ch in population]

    front0 = fronts_combined[0] if fronts_combined else []
    front0_adj_raw = [combined_objectives[i][0] for i in front0]
    front0_costs = [combined_objectives[i][1] for i in front0]
    logger.info(
        f"[GEN {gen}] NSGA-II F0={len(front0)} ; adj_raw=({min(front0_adj_raw, default=0):.2f},{max(front0_adj_raw, default=0):.2f}) ; "
        f"cost=({min(front0_costs, default=0):.2f},{max(front0_costs, default=0):.2f})"
    )

    scalarized = [(fitness_single_objective(e, cost_weight), idx) for idx, e in enumerate(evals)]
    scalarized.sort(key=lambda x: x[0], reverse=True)
    best_idx = scalarized[0][1] if scalarized else None
    best_fit = scalarized[0][0] if scalarized else None
    gen_summary = {
        "gen": gen,
        "mutate_p": mutate_p,
        "front0_size": len(front0),
        "best_scalar_fitness": best_fit,
        "avg_adjusted_raw": statistics.fmean([e.adjusted_raw for e in evals]) if evals else 0.0,
        "avg_cost": statistics.fmean([e.cost for e in evals]) if evals else 0.0,
    }
    if best_idx is not None and best_fit is not None and best_fit > global_best_scalar_fit:
        global_best_scalar_fit = best_fit
        key = normalize_key(population[best_idx])
        best_overall = _pack_best(gen, population[best_idx], evals[best_idx], best_fit, key)
        _save_checkpoint(
            best_path,
            {
                "generation": gen,
                "best_overall": best_overall,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )
    return population, evals, gen_summary, global_best_scalar_fit, best_overall


def _single_generation_step(
    population,
    evals,
    gen,
    enemy_names,
    pop_size,
    elites_per_gen,
    random_frac,
    min_counts,
    max_counts,
    min_size,
    mutate_p,
    block_shift_mut_prob,
    bias_team,
    op_log_sample_rate,
    cost_weight,
    ex,
    workers,
    num_eval_seeds,
    robust_lambda,
    time_weight,
    max_steps,
    quiet_workers,
    worker_log_file,
    cache,
    best_overall_fit_single,
    best_overall,
    best_path,
    rng,
):
    scored = [(fitness_single_objective(ev, cost_weight), idx) for idx, ev in enumerate(evals)]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_idx = scored[0][1] if scored else None
    best_fit = scored[0][0] if scored else None
    logger.info(
        f"[GEN {gen}] BestFit={best_fit:.2f}, AvgFit={statistics.fmean([s[0] for s in scored]):.2f}" if scored else f"[GEN {gen}] (no scores)"
    )
    elites_count = min(elites_per_gen, len(population))
    elites = [population[i] for _, i in scored[:elites_count]] if elites_count else []
    retain_len = pop_size - len(elites)
    parents = tournament_selection_single(
        [(s, population[i]) for s, i in scored],
        tournament_size=5,
        select_count=max(0, retain_len),
        rng=rng,
    )
    existing_keys = {normalize_key(ch) for ch in elites}
    existing_keys.update(normalize_key(ch) for ch in parents)
    children: List[Chromosome] = []
    max_attempts = retain_len * 10 if retain_len > 0 else 0
    attempts = 0
    while len(children) < retain_len and attempts < max_attempts:
        attempts += 1
        a, b = (
            rng.sample(parents, 2) if len(parents) >= 2 else (rng.choice(parents), rng.choice(parents))
        )
        child = crossover_order_then_uniform_teams(a, b, log_sample_rate=op_log_sample_rate, rng=rng)
        child = repair(child, min_counts, max_counts, min_size, rng=rng)
        child = mutate(
            child,
            min_counts,
            max_counts,
            min_size,
            p_mut=mutate_p,
            block_shift_prob=block_shift_mut_prob,
            bias_team=bias_team,
            log_sample_rate=op_log_sample_rate,
            rng=rng,
        )
        child = repair(child, min_counts, max_counts, min_size, rng=rng)
        if not validate(child, enemy_names, min_counts, max_counts, min_size):
            continue
        if rng.random() < random_frac:
            child = repair(
                random_chromosome(enemy_names, min_size, min_counts, max_counts, bias_team=bias_team, rng=rng),
                min_counts,
                max_counts,
                min_size,
                rng=rng,
            )
            if not validate(child, enemy_names, min_counts, max_counts, min_size):
                continue
        k = normalize_key(child)
        if k in existing_keys:
            continue
        existing_keys.add(k)
        children.append(child)
    population = elites + children
    evals = _eval_population_with_cache(
        population,
        ex,
        workers,
        num_eval_seeds,
        robust_lambda,
        time_weight,
        max_steps,
        max_counts,
        quiet_workers,
        worker_log_file,
        cache,
        desc=f"Gen {gen} eval",
        leave=False,
        rng=rng,
    )
    gen_summary = {
        "gen": gen,
        "mutate_p": mutate_p,
        "best_fitness": best_fit,
        "avg_fitness": statistics.fmean([fitness_single_objective(ev, cost_weight) for ev in evals]) if evals else None,
    }
    if best_idx is not None and best_fit is not None and best_fit > best_overall_fit_single:
        best_overall_fit_single = best_fit
        key = normalize_key(population[best_idx])
        best_overall = _pack_best(gen, population[best_idx], evals[best_idx], best_fit, key)
        _save_checkpoint(
            best_path,
            {
                "generation": gen,
                "best_overall": best_overall,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )
    return population, evals, gen_summary, best_overall_fit_single, best_overall


# ===== GA optimize (NSGA-II + Single Objective) =====

def genetic_optimize(enemy_names,
                     pop_size=10, generations=1,
                     min_size=2, min_counts=None, max_counts=None,
                     mode="nsga2",                 # "nsga2" | "single"
                     cost_weight=1.0,              # single-mode + scalarization for reporting
                     elites_per_gen=2,
                     random_frac=0.1,
                     block_shift_mut_prob=0.5,
                     # Adaptive mutation:
                     mutate_p_start=0.4,
                     mutate_p_end=0.1,
                     # Robust eval:
                     num_eval_seeds=3,
                     robust_lambda=0.0,
                     time_weight=0.0,
                     # Parallelism:
                     workers=None,            # None == sequential
                     max_steps=300,
                     bias_team=True,
                     # Worker IO control:
                     quiet_workers=True,
                     worker_log_file=None,
                     # Logging throttle for ops:
                     op_log_sample_rate=20,
                     # Memoization cache size:
                     cache_capacity=5000,
                     # CHECKPOINTING
                     checkpoint_every=0,           # 0 disables periodic checkpoints
                     checkpoint_path="ga_checkpoint.pkl",
                     best_path="ga_best.pkl",
                     resume_from=None,             # path to checkpoint.pkl to resume
                     # OPTIONAL: final re-eval for clean scores
                     reeval_best=False,
                     reeval_top_k=1,
                     reeval_num_eval_seeds=5,
                     reeval_robust_lambda=0.0,
                     rng=None):
    """
    Returns a rich dict with:
      - best_overall: {generation, fitness, eval, chromosome{config, sequence}, key}
      - best_final_gen: idem but for last generation
      - pareto_front: list of {fitness, eval, chromosome{...}, key} (empty in single mode)
      - history: list per generation with summaries ({gen, mutate_p, ...})
      - population_final: list of {fitness, eval, chromosome{...}, key} for the last generation
    """
    if min_counts is None:
        min_counts = {tmpl.name: 0 for tmpl in FRIENDLY_TEMPLATES}
    if max_counts is None:
        max_counts = {tmpl.name: 3 for tmpl in FRIENDLY_TEMPLATES}

    rng = rng or random

    if workers is None:
        workers = os.cpu_count() or 1

    # Persistent pool + LRU cache
    cache = _LRUCache(capacity=cache_capacity)
    ex = None
    if workers and workers > 1:
        ex = concurrent.futures.ProcessPoolExecutor(max_workers=workers)

    # RNG state & resume
    start_gen = 0
    population = []
    evals = []
    history = []
    best_overall = None
    global_best_scalar_fit = float("-inf")  # used in NSGA-II to track best across gens
    best_overall_fit_single = float("-inf") # used in single mode

    try:
        if resume_from:
            ckpt = _resume_from_checkpoint(resume_from, rng)
            start_gen = ckpt["generation"] + 1
            population = ckpt["population"]
            evals = ckpt["evals"]
            cache = ckpt.get("cache", cache)
            best_overall = ckpt.get("best_overall", None)
            history = ckpt.get("history", [])
            mode_ck = ckpt.get("mode", mode)
            if mode_ck != mode:
                logger.warning(f"[RESUME] Checkpoint mode={mode_ck} differs from requested mode={mode}; continuing anyway.")
        else:
            logger.info("[INFO] Creating initial population...")
            population = _initialize_population(enemy_names, pop_size, min_size, min_counts, max_counts, bias_team, rng)
            evals = _eval_population_with_cache(
                population,
                ex,
                workers,
                num_eval_seeds,
                robust_lambda,
                time_weight,
                max_steps,
                max_counts,
                quiet_workers,
                worker_log_file,
                cache,
                desc="Init eval (P0)",
                leave=False,
                rng=rng,
            )

        scored = [
            (fitness_single_objective(ev, cost_weight), ch, ev, normalize_key(ch))
            for ch, ev in zip(population, evals)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            if mode == "nsga2":
                global_best_scalar_fit = scored[0][0]
            else:
                best_overall_fit_single = scored[0][0]
            best_overall = _pack_best(
                start_gen - 1 if resume_from else -1,
                scored[0][1],
                scored[0][2],
                scored[0][0],
                scored[0][3],
            )

        for gen in range(start_gen, generations):
            t = (gen - start_gen) / (generations - start_gen - 1) if generations - start_gen > 1 else 1.0
            mutate_p = mutate_p_start + (mutate_p_end - mutate_p_start) * t
            if mode == "nsga2":
                population, evals, gen_summary, global_best_scalar_fit, best_overall = _nsga2_generation_step(
                    population,
                    evals,
                    gen,
                    enemy_names,
                    pop_size,
                    elites_per_gen,
                    random_frac,
                    min_counts,
                    max_counts,
                    min_size,
                    mutate_p,
                    block_shift_mut_prob,
                    bias_team,
                    op_log_sample_rate,
                    cost_weight,
                    ex,
                    workers,
                    num_eval_seeds,
                    robust_lambda,
                    time_weight,
                    max_steps,
                    quiet_workers,
                    worker_log_file,
                    cache,
                    global_best_scalar_fit,
                    best_overall,
                    best_path,
                    rng,
                )
            else:
                population, evals, gen_summary, best_overall_fit_single, best_overall = _single_generation_step(
                    population,
                    evals,
                    gen,
                    enemy_names,
                    pop_size,
                    elites_per_gen,
                    random_frac,
                    min_counts,
                    max_counts,
                    min_size,
                    mutate_p,
                    block_shift_mut_prob,
                    bias_team,
                    op_log_sample_rate,
                    cost_weight,
                    ex,
                    workers,
                    num_eval_seeds,
                    robust_lambda,
                    time_weight,
                    max_steps,
                    quiet_workers,
                    worker_log_file,
                    cache,
                    best_overall_fit_single,
                    best_overall,
                    best_path,
                    rng,
                )
            history.append(gen_summary)
            _maybe_checkpoint(
                gen,
                population,
                evals,
                cache,
                best_overall,
                history,
                rng,
                checkpoint_every,
                checkpoint_path,
                mode,
            )

        if checkpoint_every:
            _maybe_checkpoint(
                generations - 1,
                population,
                evals,
                cache,
                best_overall,
                history,
                rng,
                1,
                checkpoint_path,
                mode,
            )

        # ===== Final results (+ optional re-eval) =====
        # Final per-individual packaging
        def _package_pop(chs, evs, cost_weight):
            out = []
            for ch, ev in zip(chs, evs):
                fit = fitness_single_objective(ev, cost_weight)
                out.append(
                    {
                        "fitness": fit,
                        "eval": {**asdict(ev), "final_score": getattr(ev, "final_score", None)},
                        "chromosome": {"config": ch.config, "sequence": ch.sequence},
                        "key": normalize_key(ch),
                    }
                )
            return out

        population_final = _package_pop(population, evals, cost_weight)

        if mode == "nsga2":
            # Derive Pareto fronts for the final population
            final_objectives = [(ev.adjusted_raw, ev.cost) for ev in evals]
            pareto_fronts = fast_non_dominated_sort(final_objectives)
            front0 = pareto_fronts[0] if pareto_fronts else []
            pareto_front = [population_final[i] for i in front0]

            # Best individual in final generation (scalarized)
            best_final = max(population_final, key=lambda d: d["fitness"]) if population_final else None

            out = {
                "best_overall": best_overall,
                "best_final_gen": best_final,
                "pareto_front": pareto_front,
                "history": history,
                "population_final": population_final,
            }

        else:
            # Single-objective: best in the last generation
            best_final = max(population_final, key=lambda d: d["fitness"]) if population_final else None
            out = {
                "best_overall": best_overall,
                "best_final_gen": best_final,
                "pareto_front": [],  # not used in single mode
                "history": history,
                "population_final": population_final,
            }

        if reeval_best and population_final:
            top_k = max(1, min(len(population_final), int(reeval_top_k)))
            top = sorted(population_final, key=lambda d: d["fitness"], reverse=True)[:top_k]
            chroms = [
                Chromosome(d["chromosome"]["config"], d["chromosome"]["sequence"])
                for d in top
            ]
            reevals = _eval_population_with_cache(
                chroms,
                ex,
                workers,
                reeval_num_eval_seeds,
                reeval_robust_lambda,
                time_weight,
                max_steps,
                max_counts,
                quiet_workers,
                worker_log_file,
                _LRUCache(0),
                desc="Re-eval top",
                leave=False,
                rng=rng,
            )
            out["reevaluated_top"] = [
                {
                    "fitness": fitness_single_objective(ev, cost_weight),
                    "eval": asdict(ev),
                    "chromosome": top[i]["chromosome"],
                    "key": top[i]["key"],
                }
                for i, ev in enumerate(reevals)
            ]
            if out["reevaluated_top"]:
                out["best_final_gen_reeval"] = out["reevaluated_top"][0]
        return out

    finally:
        if ex is not None:
            ex.shutdown()


# ===== Script entry point (example) =====

if __name__ == "__main__":
    enemies = [e.name for e in _build_enemy_units()]
    result = genetic_optimize(
        enemies,
        pop_size=20,
        generations=3,
        min_size=2,
        min_counts={tmpl.name: 0 for tmpl in FRIENDLY_TEMPLATES},
        max_counts={tmpl.name: 2 for tmpl in FRIENDLY_TEMPLATES},
        mode="nsga2",
        cost_weight=1.0,
        elites_per_gen=2,
        random_frac=0.1,
        mutate_p_start=0.4,
        mutate_p_end=0.1,
        num_eval_seeds=1,
        robust_lambda=0.0,
        workers=4,
        max_steps=300,
        bias_team=True,
        quiet_workers=True,
        worker_log_file="eval_runs.log",
        op_log_sample_rate=20,
        cache_capacity=5000,
        # checkpointing:
        checkpoint_every=1,
        checkpoint_path="ga_checkpoint.pkl",
        best_path="ga_best.pkl",
        resume_from=None,               # set to "ga_checkpoint.pkl" to resume
        # re-eval option:
        reeval_best=True,
        reeval_top_k=1,
        reeval_num_eval_seeds=5,
        reeval_robust_lambda=0.0
    )
    # Demo printout
    bo = result["best_overall"]
    bf = result["best_final_gen"]
    print("[Summary] Best overall:", None if not bo else f"gen={bo['generation']} fit={bo['fitness']:.2f}")
    print("[Summary] Best final gen:", None if not bf else f"fit={bf['fitness']:.2f}")
    print("[Summary] Pareto size:", len(result["pareto_front"]))
