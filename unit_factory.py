import copy

# Enemy unit factory
def make_enemy(name, cls, base_template, position=None, domain=None):
    state = copy.deepcopy(base_template)
    state["name"] = name

    if position is not None:
        state["position"] = position

    # Store base armor values for defensive state transitions
    state["base_armor_front"] = state["armor_front"]
    state["base_armor_side"]  = state["armor_side"]
    state["base_armor_rear"]  = state["armor_rear"]

    return cls(name, state, domain)


# Friendly unit factory
def make_friendly(name, cls, base_template, domain, enemy_template):
    state = copy.deepcopy(base_template)
    state["name"] = name
    state["enemy"] = enemy_template
    state["target_enemy"] = enemy_template
    state["outpost_position"] = enemy_template["outpost_position"]
    state["visible_enemies"] = []

    return cls(name, state, domain)
