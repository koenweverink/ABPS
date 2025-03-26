# state.py
from agent import Agent

class State:
    def __init__(self):
        self.data = {
            "agents": {
                "scouts": Agent("scouts", 0, 0, 800, 0, attack_range=5, v_max=6.0, health=50, max_health=50),
                "infantry": Agent("infantry", 0, 0, 800, 15, attack_range=2, v_max=4.0, health=75, max_health=75),
                "tanks": Agent(
                    "tanks", 0, 0, 1700, 30, attack_range=3, v_max=3.0, health=150, max_health=150,
                    max_speed="75 km/h",
                    front_armor=17,
                    turret_armor="18 / 19",
                    side_armor=4,
                    rear_armor=3,
                    durability=20,
                    main_ammo="120mm Gun",
                    secondary_ammo="7.62mm Machine Gun",
                    vision_range="2000 m",
                    night_vision="Thermal",
                    stealth_modifier=0,
                    has_smoke_grenades="Yes",
                    main_gun_range="2400 meters",
                    main_gun_accuracy="75%",
                    main_gun_stabilized="60%",
                    main_gun_reaction="2s",
                    main_gun_rate_of_fire="4.9 rpm",
                    apfsds_damage=9,
                    apfsds_suppression=12,
                    apfsds_penetration=18,
                    he_frag_damage=2.5,
                    he_frag_suppression=5,
                    he_frag_suppression_radius="25 meters",
                    secondary_gun_range="1200 meters",
                    secondary_gun_accuracy="30%",
                    secondary_gun_stabilized="18%",
                    secondary_gun_reaction="1s",
                    secondary_gun_rate_of_fire="197 rpm",
                    ball_ammo_damage=0.8,
                    ball_ammo_suppression=1,
                    ball_ammo_penetration=1
                )
            },
            "enemy": Agent("enemy", 9, 9, float('inf'), 10, attack_range=2, v_max=5.0, health=100, max_health=100),
            "mission_complete": False
        }