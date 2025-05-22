from utils import manhattan, has_line_of_sight


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


def expand_attack_or_move(s):
    """Generate a plan to either attack or move toward each spotted enemy."""
    return [
        ("AttackEnemy", name) if s.get("unit", {}).can_attack(s["sim"].enemy_units_dict[name])
        else ("Move", name)
        for name in sorted(
            [
                name for name in s.get("spotted_enemies", [])
                if name in s["sim"].enemy_units_dict and s["sim"].enemy_units_dict[name].state.get("enemy_alive", False)
            ],
            key=lambda n: manhattan(
                s.get("unit", {}).state.get("position", (0, 0)),
                s["sim"].enemy_units_dict[n].state.get("position", (0, 0))
            )
        )
    ]


def condition_default(s):
    """Catch-all condition to match any state dictionary."""
    return isinstance(s, dict)


# Friendly HTN domain
secure_outpost_domain = {
    "SecureOutpostMission": [
        (condition_outpost_secured, []),
        (condition_all_enemies_defeated, [("SecureOutpost", None)]),
        (condition_spotted_enemies_exist, expand_attack_or_move),
        (condition_default, ["Hold"]),
    ],

    "DefeatEnemies": [
        (
            lambda s: isinstance(s, dict) and bool(s.get("spotted_enemies", [])),
            lambda s: [
                ("AttackEnemy", name) if s["unit"].can_attack(s["sim"].enemy_units_dict.get(name)) else ("Move", name)
                for name in s.get("spotted_enemies", [])
                if name in s["sim"].enemy_units_dict and s["sim"].enemy_units_dict[name].state.get("enemy_alive", False)
            ]
        ),
        (condition_default, ["Hold"]),
    ],

    "SecureOutpost": [
        (
            lambda s: isinstance(s, dict) and s.get("position") != s.get("outpost_position"),
            [("Move", "outpost")]
        ),
        (
            lambda s: isinstance(s, dict) and s.get("position") == s.get("outpost_position"),
            ["SecureOutpostNoArg"]
        ),
    ],
}


# Enemy HTN domain logic

defend_area_conditions = [
    (
        lambda s: bool(s["spotted_enemies"]),
        lambda s: [
            ("AttackEnemy", name) if (
                manhattan(s["unit"].state["position"], s["sim"].friendly_units_dict[name].state["position"]) <= s["unit"].state["attack_range"]
                and has_line_of_sight(s["unit"].state["position"], s["sim"].friendly_units_dict[name].state["position"])
            ) else ("Move", name)
            for name in s["spotted_enemies"]
            if s["sim"].friendly_units_dict[name].state["current_group_size"] > 0
        ]
    ),
    (lambda s: True, ["BattlePosition"]),
]

# Enemy domain
enemy_domain = {
    "DefendAreaMission": defend_area_conditions
}
