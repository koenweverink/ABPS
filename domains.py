from utils import manhattan, has_line_of_sight

secure_outpost_domain = {
    "SecureOutpostMission": [
        # Condition 1: Outpost already secured
        (lambda s: isinstance(s, dict) and any(u.state.get("outpost_secured", False)
                    for u in getattr(s.get("sim", object()), "friendly_units", [])), []),

        # Condition 2: All enemies defeated, proceed to secure outpost
        (lambda s: isinstance(s, dict) and not any(
            e.state.get("enemy_alive", False)
            for e in getattr(s.get("sim", object()), "enemy_units_dict", {}).values()
        ), [("SecureOutpost", None)]),

        # Condition 3: Spotted enemies exist, defeat them
        (lambda s: isinstance(s, dict) and any(
            name in getattr(s.get("sim", object()), "enemy_units_dict", {}) and
            s["sim"].enemy_units_dict[name].state.get("enemy_alive", False)
            for name in s.get("spotted_enemies", [])
        ), lambda s: [
            sub
            for name in sorted(
                [
                    name for name in s.get("spotted_enemies", [])
                    if name in getattr(s.get("sim", object()), "enemy_units_dict", {}) and
                    s["sim"].enemy_units_dict[name].state.get("enemy_alive", False)
                ],
                key=lambda n: manhattan(
                    s.get("unit", {}).state.get("position", (0, 0)),
                    getattr(s.get("sim", object()), "enemy_units_dict", {}).get(n, {}).state.get("position", (0, 0))
                )
            )
            for sub in [
                ("AttackEnemy", name) if s.get("unit", {}).can_attack(s["sim"].enemy_units_dict[name])
                else ("Move", name)
            ]
        ]),

        # Condition 4: Default to Hold (waiting for drone to spot enemies)
        (lambda s: isinstance(s, dict), ["Hold"]),
    ],

    "DefeatEnemies": [
        # Condition 1: Spotted enemies exist
        (
            lambda s: isinstance(s, dict) and bool(s.get("spotted_enemies", [])),
            lambda s: [
                ("AttackEnemy", name) if s.get("unit", {}).can_attack(getattr(s.get("sim", object()), "enemy_units_dict", {}).get(name))
                else ("Move", name)
                for name in s.get("spotted_enemies", [])
                if name in getattr(s.get("sim", object()), "enemy_units_dict", {}) and
                s["sim"].enemy_units_dict[name].state.get("enemy_alive", False)
            ]
        ),
        # Condition 2: Default to Hold
        (
            lambda s: isinstance(s, dict),
            ["Hold"]
        ),
    ],

    "SecureOutpost": [
        (lambda s: isinstance(s, dict) and s.get("position") != s.get("outpost_position"), [("Move", "outpost")]),
        (lambda s: isinstance(s, dict) and s.get("position") == s.get("outpost_position"), ["SecureOutpostNoArg"]),
    ],
}

enemy_domain = {
    "DefendAreaMission": [

        # 1) As long as there's *any* name in state["spotted_enemies"],
        #    stay in Attack mode
        (
            lambda s: bool(s["spotted_enemies"]),
            lambda s: [
                # For each spotted enemy name, choose Move vs Attack
                ("AttackEnemy", name)
                if manhattan(
                    s["unit"].state["position"],
                    s["sim"].friendly_units_dict[name].state["position"]
                ) <= s["unit"].state["attack_range"] 
                and has_line_of_sight(
                    s["unit"].state["position"],
                    s["sim"].friendly_units_dict[name].state["position"]
                )
                else ("Move", name)
                for name in s["spotted_enemies"]
                # filter out any that have since died
                if s["sim"].friendly_units_dict[name].state["current_group_size"] > 0
            ]
        ),

        # 2) Otherwise—no more spotted contacts—dig in
        (
            lambda s: True,
            ["BattlePosition"]
        ),
    ]
}