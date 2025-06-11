from utils import manhattan, has_line_of_sight, compute_staging_position


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


def condition_can_consolidate_attack(s):
    """Return True if multiple friendlies are alive and enemies are spotted."""
    if not isinstance(s, dict):
        return False
    sim = s.get("sim")
    if not sim:
        return False
    alive_friendlies = [u for u in sim.friendly_units if u.state.get("current_group_size", 0) > 0]
    return len(alive_friendlies) >= 2 and condition_spotted_enemies_exist(s)


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
    """
    For each spotted, alive enemy (in order of proximity), 
    insert: MoveToStaging → WaitForGroup → AttackEnemy(enemy_name)
    """
    # 1) Gather and sort all spotted, alive enemies
    enemies = [
        name for name in s["spotted_enemies"]
        if name in s["sim"].enemy_units_dict
        and s["sim"].enemy_units_dict[name].state.get("enemy_alive", False)
    ]
    enemies.sort(key=lambda n: manhattan(
        s["unit"].state["position"],
        s["sim"].enemy_units_dict[n].state["position"]
    ))

    plan = []
    for name in enemies:
        # each time we’re about to switch to a new enemy…
        plan.append(("MoveToStaging", None))
        plan.append("WaitForGroup")
        plan.append(("AttackEnemy", name))

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

        # 3) Otherwise, if we can consolidate (stage) for an attack, do that
        (condition_can_consolidate_attack, [("ConsolidateAttack", None)]),

        # 4) If we have any spotted enemies, attack or move toward them
        (condition_spotted_enemies_exist, expand_attack_or_move),

        # 5) Default: hold position
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

defend_area_conditions = [
    # (
    #     lambda s: bool(s["spotted_enemies"]),
    #     lambda s: [
    #         ("AttackEnemy", name) if (
    #             manhattan(s["unit"].state["position"], s["sim"].friendly_units_dict[name].state["position"]) <= s["unit"].state["attack_range"]
    #             and has_line_of_sight(s["unit"].state["position"], s["sim"].friendly_units_dict[name].state["position"])
    #         ) else ("Move", name)
    #         for name in s["spotted_enemies"]
    #         if s["sim"].friendly_units_dict[name].state["current_group_size"] > 0
    #     ]
    # ),
    (lambda s: True, ["BattlePosition"]),
]

# Enemy domain
enemy_domain = {
    "DefendAreaMission": defend_area_conditions
}
