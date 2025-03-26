# agent.py

class Agent:
    def __init__(self, name, x, y, fuel, attack_strength, attack_range=1, 
                 v_max=6.0, s_eff=1.0, c_cover=1.0, n_noise=1.0, e_elev=1.0,
                 max_speed=None, front_armor=None, turret_armor=None, side_armor=None,
                 rear_armor=None, durability=None, main_ammo=None, secondary_ammo=None,
                 vision_range=None, night_vision=None, stealth_modifier=None,
                 has_smoke_grenades=None, main_gun_range=None, main_gun_accuracy=None,
                 main_gun_stabilized=None, main_gun_reaction=None, main_gun_rate_of_fire=None,
                 apfsds_damage=None, apfsds_suppression=None, apfsds_penetration=None,
                 he_frag_damage=None, he_frag_suppression=None, he_frag_suppression_radius=None,
                 secondary_gun_range=None, secondary_gun_accuracy=None, secondary_gun_stabilized=None,
                 secondary_gun_reaction=None, secondary_gun_rate_of_fire=None,
                 ball_ammo_damage=None, ball_ammo_suppression=None, ball_ammo_penetration=None,
                 health=100, max_health=100):
        self.name = name
        self.x = x
        self.y = y
        self.fuel = fuel
        self.attack_strength = attack_strength
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
        # New attributes
        self.max_speed = max_speed
        self.front_armor = front_armor
        self.turret_armor = turret_armor
        self.side_armor = side_armor
        self.rear_armor = rear_armor
        self.durability = durability
        self.main_ammo = main_ammo
        self.secondary_ammo = secondary_ammo
        self.vision_range = vision_range
        self.night_vision = night_vision
        self.stealth_modifier = stealth_modifier
        self.has_smoke_grenades = has_smoke_grenades
        self.main_gun_range = main_gun_range
        self.main_gun_accuracy = main_gun_accuracy
        self.main_gun_stabilized = main_gun_stabilized
        self.main_gun_reaction = main_gun_reaction
        self.main_gun_rate_of_fire = main_gun_rate_of_fire
        self.apfsds_damage = apfsds_damage
        self.apfsds_suppression = apfsds_suppression
        self.apfsds_penetration = apfsds_penetration
        self.he_frag_damage = he_frag_damage
        self.he_frag_suppression = he_frag_suppression
        self.he_frag_suppression_radius = he_frag_suppression_radius
        self.secondary_gun_range = secondary_gun_range
        self.secondary_gun_accuracy = secondary_gun_accuracy
        self.secondary_gun_stabilized = secondary_gun_stabilized
        self.secondary_gun_reaction = secondary_gun_reaction
        self.secondary_gun_rate_of_fire = secondary_gun_rate_of_fire
        self.ball_ammo_damage = ball_ammo_damage
        self.ball_ammo_suppression = ball_ammo_suppression
        self.ball_ammo_penetration = ball_ammo_penetration