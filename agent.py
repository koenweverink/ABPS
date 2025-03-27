# agent.py

class Agent:
    def __init__(self, name, x, y, fuel, attack_range=1, 
                 v_max=6.0, s_eff=1.0, c_cover=1.0, n_noise=1.0, e_elev=1.0,
                 health=100, max_health=100,
                 max_speed=None, front_armor=None, turret_armor=None, side_armor=None,
                 rear_armor=None, durability=None, vision_range=None, night_vision=None,
                 stealth_modifier=None, has_smoke_grenades=None, survivability=None,
                 main_weapon=None, secondary_weapon=None):
        self.name = name
        self.x = x
        self.y = y
        self.fuel = fuel
        self.attack_range = attack_range
        self.detected_by_enemy = False
        self.detects_enemy = False
        self.v_max = v_max
        self.s_eff = s_eff
        self.c_cover = c_cover
        self.n_noise = n_noise
        self.e_elev = e_elev
        self.health = health
        self.max_health = max_health
        self.max_speed = max_speed
        self.front_armor = front_armor
        self.turret_armor = turret_armor
        self.side_armor = side_armor
        self.rear_armor = rear_armor
        self.durability = durability
        self.vision_range = vision_range
        self.night_vision = night_vision
        self.stealth_modifier = stealth_modifier
        self.has_smoke_grenades = has_smoke_grenades
        self.survivability = survivability
        self.main_weapon = main_weapon if main_weapon is not None else {}
        self.secondary_weapon = secondary_weapon if secondary_weapon is not None else {}
        # New attributes for combat mechanics
        self.suppression = 0.0  # Suppression level (0 to 1)
        self.moved_last_turn = False  # Track if the agent moved in the last turn