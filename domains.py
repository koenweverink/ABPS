from utils import manhattan, units_spotted_by_vision, compute_staging_position, compute_flanking_position, astar, _is_path_safe_from_other_enemies
from log import logger


def condition_outpost_secured(s):
    """Check if any friendly unit has already secured the outpost."""
    return isinstance(s, dict) and any(
        u.state.get("outpost_secured", False)
        for u in getattr(s.get("sim", object()), "friendly_units", [])
    )


def condition_all_enemies_defeated(s):
    """Return True if no known enemy is still alive."""
    return isinstance(s, dict) and not any(
        e.state.get("enemy_alive", False)
        for e in getattr(s.get("sim", object()), "enemy_units_dict", {}).values()
    )


def condition_spotted_enemies_exist(s):
    """Return True if the agent has any spotted enemies that are still alive."""
    return isinstance(s, dict) and any(
        name in getattr(s.get("sim", object()), "enemy_units_dict", {}) and
        s["sim"].enemy_units_dict[name].state.get("enemy_alive", False)
        for name in s.get("spotted_enemies", [])
    )

def condition_spotted_friendlies_exist(s):
    """Return True if any friendlies are spotted (via drone or LOS)."""
    return isinstance(s, dict) and any(
        name in getattr(s.get("sim", object()), "friendly_units_dict", {})
        and s["sim"].friendly_units_dict[name].state.get("health", 0) > 0
        for name in s.get("spotted_enemies", [])
    )

def condition_can_consolidate_attack(s):
    """Return True if multiple friendlies are alive and enemies are spotted."""
    if not isinstance(s, dict):
        return False
    sim = s.get("sim")
    if not sim:
        return False
    alive_friendlies = [u for u in sim.friendly_units if u.state.get("current_group_size", 0) > 0]
    return len(alive_friendlies) >= 2 and condition_spotted_enemies_exist(s)


def _has_attack_order(s, order_type):
    if not isinstance(s, dict):
        return False
    seq = getattr(s.get("sim"), "attack_sequence", [])
    for item in seq:
        if len(item) == 3:
            enemy, atype, assigned = item
            if (
                atype == order_type
                and s["unit"].name in assigned
                and enemy in s["sim"].enemy_units_dict
                and s["sim"].enemy_units_dict[enemy].state.get("enemy_alive", False)
            ):
                return True
    return False


def condition_has_flank_order(s):
    return condition_can_consolidate_attack(s) and _has_attack_order(s, "FlankAttack")


def condition_has_consolidate_order(s):
    return condition_can_consolidate_attack(s) and _has_attack_order(s, "ConsolidateAttack")

def condition_has_any_attack_order(s):
    """Return True if this unit has any remaining attack orders in the sequence."""
    if not isinstance(s, dict):
        return False
    sim = s.get("sim")
    unit = s.get("unit")
    if not sim or not unit:
        return False
    seq = getattr(sim, "attack_sequence", [])
    for enemy, _, assigned in seq:
        if (
            unit.name in assigned
            and enemy in sim.enemy_units_dict
            and sim.enemy_units_dict[enemy].state.get("enemy_alive", False)
        ):
            return True
    return False


def condition_spotted_enemies_with_order(s):
    """True if enemies are spotted and this unit still has attack orders."""
    return condition_spotted_enemies_exist(s) and condition_has_any_attack_order(s)


def expand_attack_or_move(s):
    """
    Always schedule an AttackEnemy step for each spotted, alive enemy.
    Runtime will swap to Move if needed.
    """
    enemies = [
        name for name in s.get("spotted_enemies", [])
        if name in s["sim"].enemy_units_dict
        and s["sim"].enemy_units_dict[name].state.get("enemy_alive", False)
    ]
    # sort by current distance for a deterministic order
    enemies.sort(key=lambda n: manhattan(
        s["unit"].state["position"],
        s["sim"].enemy_units_dict[n].state["position"]
    ))
    return [("AttackEnemy", name) for name in enemies]


def expand_consolidate_attack(s):
#     """Return a consolidated (frontal) attack plan for this unit if it is in the attack sequence."""
    return multi_enemy_plan_expander(s["sim"], s["unit"])
#     seq = getattr(s.get("sim"), "attack_sequence", None)
#     plan = []

#     if not seq:
#         return plan

#     for enemy_name, attack_type, assigned in seq:
#         if attack_type != "ConsolidateAttack":
#             continue
#         if s["unit"].name not in assigned:
#             continue

#         enemy_unit = s["sim"].enemy_units_dict.get(enemy_name)
#         if not enemy_unit or not enemy_unit.state.get("enemy_alive", False):
#             continue

#         sim = s["sim"]
#         unit = s["unit"]
#         unit_type = unit.state.get("type")

#         # Compute and cache the staging position
#         staging_pos = compute_staging_position(sim, enemy_name, unit_type)
#         if staging_pos:
#             unit.state["staging_position"] = staging_pos
#             unit.state["staging_type"] = "front"
#             unit.state["flank_position"] = None  # Not used here
#             plan += [("MoveToStaging", enemy_name), "WaitForGroup", ("AttackEnemy", enemy_name)]
#         else:
#             logger.warning(f"[plan] {unit.name}: No valid staging for Consolidate vs {enemy_name}, skipping")

#     return plan


def expand_flank_attack(s):
#     """
#     Generate a flanking attack plan for this unit:
#     - Primary plan: flank or stage depending on role
#     - Fallback: consolidate (frontal staging)
#     - Fallback #2: skip the attack step entirely
#     """
    return multi_enemy_plan_expander(s["sim"], s["unit"])

#     seq = getattr(s.get("sim"), "attack_sequence", None)
#     plan = []

#     if not seq:
#         return plan

#     sim = s["sim"]
#     unit = s["unit"]
#     unit_type = unit.state["type"]
#     unit_name = unit.name

#     for enemy_name, attack_type, assigned in seq:
#         if attack_type != "FlankAttack":
#             continue
#         if unit_name not in assigned:
#             continue

#         enemy_unit = sim.enemy_units_dict.get(enemy_name)
#         if not enemy_unit or not enemy_unit.state.get("enemy_alive", False):
#             continue

#         # Determine which friendly unit has the strongest frontal armor
#         units = [sim.friendly_units_dict[n] for n in assigned if n in sim.friendly_units_dict]
#         front_unit = max(units, key=lambda u: u.state.get("armor_front", 0)) if units else None

#         if unit is front_unit:
#             # Frontal attacker: only stage
#             staging_pos = compute_staging_position(sim, enemy_name, unit_type)
#             if staging_pos:
#                 unit.state["staging_position"] = staging_pos
#                 unit.state["flank_position"] = None
#                 unit.state["staging_type"] = "front"
#                 plan += [("MoveToStaging", enemy_name), "WaitForGroup", ("AttackEnemy", enemy_name)]
#             else:
#                 logger.warning(f"[plan] {unit_name}: No valid staging for {enemy_name}, skipping")
#         else:
#             # Flanker: try flank, fallback to consolidate
#             flank_pos, staging_pos = compute_flanking_position(sim, enemy_name, unit_type)

#             if flank_pos and flank_pos != sim.enemy_units_dict[enemy_name].state.get("position"):
#                 unit.state["flank_position"] = flank_pos
#                 unit.state["staging_position"] = staging_pos
#                 unit.state["staging_type"] = "flank"
#                 plan += [("MoveToFlank", enemy_name), "WaitForGroup", ("AttackEnemy", enemy_name)]
#             elif staging_pos:
#                 logger.warning(f"[plan] {unit_name}: Flank failed — falling back to frontal attack")
#                 unit.state["flank_position"] = None
#                 unit.state["staging_position"] = staging_pos
#                 unit.state["staging_type"] = "front"
#                 plan += [("MoveToStaging", enemy_name), "WaitForGroup", ("AttackEnemy", enemy_name)]
#             else:
#                 logger.warning(f"[plan] {unit_name}: No flank or staging for {enemy_name} — skipping this attack")

#     return plan

def multi_enemy_plan_expander(sim, unit):
    """
    Generate a full attack plan for a friendly unit based on multi-enemy attack_sequence.
    Supports FlankAttack and ConsolidateAttack types, including fallback logic.
    Units begin moving to staging/flank immediately; attack waits are handled in _execute_wait_for_group.
    """
    plan = []
    unit_name = unit.name
    unit_type = unit.state["type"]
    seq = getattr(sim, "attack_sequence", [])

    flank_cache = {}
    # Use a shared cache on the simulation to keep staging locations
    # consistent across all friendly units planning the same phase.
    group_staging_cache = getattr(sim, "group_staging_cache", {})

    # Store per-phase attack info so execution knows correct staging/flank
    unit.state["phase_attack_info"] = {}
    phase_map = unit.state["phase_attack_info"]
    first_phase_set = False

    for phase_index, (enemy_name, attack_type, assigned_units) in enumerate(seq):
        if unit_name not in assigned_units:
            continue

        enemy_unit = sim.enemy_units_dict.get(enemy_name)
        if not enemy_unit or not enemy_unit.state.get("enemy_alive", False):
            continue

        if (enemy_name not in sim.friendly_drone.last_known and enemy_name not in unit.state.get("spotted_enemies", [])):
            logger.warning(f"[plan] {unit_name}: Enemy {enemy_name} not spotted yet — skipping phase for now")
            continue

        units = [sim.friendly_units_dict[n] for n in assigned_units if n in sim.friendly_units_dict]
        front_unit = max(units, key=lambda u: u.state.get("armor_front", 0)) if units else None
        
        group_key = (enemy_name, phase_index)
        if group_key not in group_staging_cache and front_unit:
            group_staging_cache[group_key] = compute_staging_position(
                sim, enemy_name, front_unit, front_unit
            )
        staging_common = group_staging_cache.get(group_key)

        # Info describing how this unit should approach this enemy in this phase
        phase_info = {"enemy_position": enemy_unit.state.get("position")}

        try:
            if attack_type == "FlankAttack":
                if unit is front_unit:
                    staging = staging_common

                    if staging:
                        phase_info.update({"staging": staging, "staging_type": "front", "flank": None})
                        plan += [("MoveToStaging", enemy_name), "WaitForGroup", ("AttackEnemy", enemy_name)]
                    else:
                        logger.warning(f"[plan] {unit_name} (front): ❌ No valid staging for {enemy_name}, skipping phase")
                        continue
                else:
                    key = (enemy_name, unit_name, "flank")
                    if key not in flank_cache:
                        flank_cache[key] = compute_flanking_position(sim, enemy_name, front_unit, unit)
                    flank, staging_hint = flank_cache[key]

                    if flank and flank != sim.enemy_units_dict[enemy_name].state.get("position"):
                        phase_info.update({"staging": staging_hint, "staging_type": "flank", "flank": flank})
                        plan += [("MoveToFlank", enemy_name), "WaitForGroup", ("AttackEnemy", enemy_name)]
                    else:
                        staging = staging_common

                        if staging:
                            logger.warning(f"[plan] {unit_name}: Flank failed — fallback to ConsolidateAttack for {enemy_name}")
                            phase_info.update({"staging": staging, "staging_type": "front", "flank": None})
                            plan += [("MoveToStaging", enemy_name), "WaitForGroup", ("AttackEnemy", enemy_name)]
                        else:
                            logger.warning(f"[plan] {unit_name}: ❌ No valid flank or fallback staging for {enemy_name}, skipping phase")
                            continue

            elif attack_type == "ConsolidateAttack":
                staging = staging_common

                if staging:
                    phase_info.update({"staging": staging, "staging_type": "front", "flank": None})
                    plan += [("MoveToStaging", enemy_name), "WaitForGroup", ("AttackEnemy", enemy_name)]
                else:
                    logger.warning(f"[plan] {unit_name}: ❌ No valid staging for ConsolidateAttack on {enemy_name}, skipping phase")
                    continue

        except Exception as e:
            logger.exception(f"[plan] Exception during expansion for {unit_name} vs {enemy_name}: {e}")
            continue

        # Record attack info for this phase so execution knows where to go
        if phase_info:
            phase_map[group_key] = phase_info
            if not first_phase_set:
                first_phase_set = True
                unit.state["staging_position"] = phase_info.get("staging")
                unit.state["flank_position"] = phase_info.get("flank")
                unit.state["staging_type"] = phase_info.get("staging_type")
                unit.state.setdefault("targeting_enemy", enemy_name)

    return plan


def condition_default(s):
    """Catch-all condition to match any state dictionary."""
    return isinstance(s, dict)


# Friendly HTN domain
secure_outpost_domain = {
    "SecureOutpostMission": [
        # 1) If the outpost is already secured, do nothing
        (condition_outpost_secured, []),

        # 2) If all enemies are defeated, transition to the SecureOutpost task
        (condition_all_enemies_defeated, [("SecureOutpost", None)]),

        # 3) If we have orders to flank attack, execute that
        (condition_has_flank_order, [("FlankAttack", None)]),

        # 4) Otherwise, if we can consolidate (stage) for an attack, do that
        (condition_has_consolidate_order, [("ConsolidateAttack", None)]),

        # 5) If we have spotted enemies and standing orders, attack or move toward them
        (condition_spotted_enemies_with_order, expand_attack_or_move),

        # 6) Default: hold position
        (condition_default, ["Hold"]),
    ],

    "DefeatEnemies": [
        (
            # If there are spotted enemies, for each do AttackEnemy if in range, else Move
            lambda s: isinstance(s, dict) and bool(s.get("spotted_enemies", [])),
            lambda s: [
                ("AttackEnemy", name)
                if s["unit"].can_attack(s["sim"].enemy_units_dict[name])
                else ("Move", name)
                for name in s["spotted_enemies"]
                if name in s["sim"].enemy_units_dict
                   and s["sim"].enemy_units_dict[name].state.get("enemy_alive", False)
            ]
        ),
        (condition_default, ["Hold"]),
    ],

    "ConsolidateAttack": [
        # ConsolidateAttack is handled by your expand_consolidate_attack (staging + per-target loops)
        (condition_default, expand_consolidate_attack),
    ],

    "FlankAttack": [
        # FlankAttack uses expand_flank_attack to approach from side/rear
        (condition_default, expand_flank_attack),
    ],

    "SecureOutpost": [
        (
            # If not yet at the outpost, move there
            lambda s: isinstance(s, dict) and s.get("position") != s.get("outpost_position"),
            [("Move", "outpost")]
        ),
        (
            # Once at the outpost, execute SecureOutpostNoArg (e.g. ‘dig in’)
            lambda s: isinstance(s, dict) and s.get("position") == s.get("outpost_position"),
            ["SecureOutpostNoArg"]
        ),
    ],
}


# Enemy HTN domain logic

def face_and_attack(s):
    """Face assigned enemy using vision or drone memory and attack if in range."""
    unit = s["unit"]
    visible = units_spotted_by_vision(unit, s.get("friendly_units", []))
    
    if not visible:
        return ["Hold"]

    # Pick closest
    target = min(visible, key=lambda u: manhattan(unit.state["position"], u.state["position"]))

    # Determine if already facing
    dx = target.state["position"][0] - unit.state["position"][0]
    dy = target.state["position"][1] - unit.state["position"][1]
    if dx != 0:
        dx = int(dx / abs(dx))
    if dy != 0:
        dy = int(dy / abs(dy))
    desired_facing = (dx, dy)
    current_facing = unit.state.get("facing", (0, 1))

    tasks = []

    last_task = s.get("last_task")
    allow_face = last_task in ("Hold", "BattlePosition")

    if current_facing != desired_facing and allow_face:
        tasks.append(("FaceEnemy", target.name))

    if unit.can_attack(target):
        tasks.append(("AttackEnemy", target.name))
    elif allow_face:
        tasks.append("Hold")  # Hold only when intentionally facing
    return tasks


def condition_needs_retreat(s):
    """Determine if a unit should retreat based on health."""
    unit = s["unit"]
    health = unit.state.get("health", 0)
    max_health = unit.state.get("max_health", 1)
    # Only retreat when the unit has taken damage and falls below the
    # 25% health threshold. This prevents premature retreats at mission
    # start when units are still at full health.
    low = health < max_health and health <= 0.25 * max_health
    return low and not unit.state.get("has_retreated", False)

def condition_can_see_friendly(s):
    """Return True if any friendlies are actually visible (LOS + range + stealth)."""
    return bool(units_spotted_by_vision(s["unit"], s.get("friendly_units", [])))

def condition_should_delay(s):
    """Trigger a fallback at mission start or when health drops."""
    unit = s["unit"]
    if "delay_bridges" not in unit.state:
        return True
    if unit.state.get("force_delay", False):
        return True

    thresholds = unit.state.get("delay_health_thresholds", [0.75, 0.5])
    stage = unit.state.get("delay_stage", 0)
    order = unit.state.get("delay_bridge_order", [])
    pos = unit.state.get("delay_order_pos", 0)
    health_ratio = unit.state.get("health", 0) / max(1, unit.state.get("max_health", 1))

    if not order:
        return health_ratio < thresholds[0] and not unit.state.get("delay_retreating", False)

    return (
        stage < len(thresholds)
        and health_ratio < thresholds[stage]
        and pos < len(order) - 1
        and not unit.state.get("delay_retreating", False)
    )

defend_area_conditions = [
    # 1. Retreat if low health
    (condition_needs_retreat, [("Retreat", None)]),

    # 2. Pick a defend position
    (lambda s: not s["unit"].state.get("picked_position", False), ["PickPosition"]),

    # 3. Move to that position
    (lambda s: s["unit"].state.get("position") != s["unit"].state.get("defend_position", s["unit"].state.get("position")), ["MoveToPosition"]),

    # 4. Enter BattlePosition if not already entrenched
    (lambda s: not s["unit"].state.get("in_battle_position", False), ["BattlePosition"]),

    # 5. Face/attack enemies **only if visible**
    (condition_can_see_friendly, face_and_attack),

    # 6. Hold if nothing else applies (entrenched, no enemies visible)
    (lambda s: True, ["Hold"]),
]

delay_conditions = [
    # 1. Retreat if low health
    (condition_needs_retreat, [("Retreat", None)]),

    # 2. Start mission or fall back when health drops
    (condition_should_delay, [("Delay", None)]),

    # 3. Pick a defend position
    (lambda s: not s["unit"].state.get("picked_position", False), ["PickPosition"]),

    # 4. Move to that position
    (lambda s: s["unit"].state.get("position") != s["unit"].state.get("defend_position", s["unit"].state.get("position")), ["MoveToPosition"]),

    # 5. Enter BattlePosition if not already entrenched
    (lambda s: not s["unit"].state.get("in_battle_position", False), ["BattlePosition"]),

    # 6. Face/attack enemies if visible
    (condition_can_see_friendly, face_and_attack),

    # 7. Hold otherwise
    (lambda s: True, ["Hold"]),
]

# Enemy domain
enemy_domain = {
    "DefendAreaMission": defend_area_conditions,
    "DelayMission": delay_conditions,
}
