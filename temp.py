    enemy_domain = {
            "DefendAreaMission": [
                (lambda state: any(manhattan(state["enemy_position"], u.state["position"]) <= state["enemy_attack_range"] and 
                                    has_line_of_sight(state["enemy_position"], u.state["position"]) 
                                    for u in state["friendly_units"]), ["AttackTarget"]),
                (lambda state: any(manhattan(state["enemy_position"], u.state["position"]) <= state["vision_range"] and 
                                    has_line_of_sight(state["enemy_position"], u.state["position"]) 
                                    for u in state["friendly_units"]), ["ChaseTarget"]),
                (lambda state: True, ["Patrol"])
            ]
        }



def enemy_in_range_with_los(state):
    return (state["enemy"]["enemy_alive"] and
            manhattan(state["position"], state["enemy"]["enemy_position"]) <= state["friendly_attack_range"] and
            has_line_of_sight(state["position"], state["enemy"]["enemy_position"]))
