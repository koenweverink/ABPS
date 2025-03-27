# state.py
from agent import Agent

class State:
    def __init__(self):
        self.data = {
            "agents": {
                "scouts": Agent(
                    "scouts", 0, 0, 800, attack_range=5, v_max=6.0, health=50, max_health=50,
                    max_speed="90 km/h",
                    front_armor="3 / 3",
                    turret_armor="---",
                    side_armor=2,
                    rear_armor=2,
                    durability=18,
                    vision_range="2600 m",
                    night_vision="Advanced",
                    stealth_modifier=225,
                    has_smoke_grenades="Yes",
                    main_weapon={
                        "name": "25mm Autocannon",
                        "range": "1800 meters",
                        "accuracy": "35%",
                        "stabilized": "21%",
                        "reaction": "2s",
                        "rate_of_fire": "115 rpm",
                        "damage": 2.4,
                        "suppression": 5,
                        "penetration": 5,
                        "he_damage": 2,
                        "he_suppression": 5,
                        "he_penetration": 1
                    },
                    secondary_weapon={}
                ),
                "infantry": Agent(
                    "infantry", 0, 0, 800, attack_range=2, v_max=4.0, health=75, max_health=75,
                    survivability="100%",
                    main_weapon={
                        "name": "Assault Rifle",
                        "range": "800 meters",
                        "accuracy": "10%",
                        "stabilized": "8%",
                        "reaction": "1s",
                        "rate_of_fire": "400 rpm",
                        "damage": 0.5,
                        "suppression": 0.5,
                        "penetration": 0
                    },
                    secondary_weapon={
                        "name": "M72 LAW",
                        "range": "800 meters",
                        "accuracy": "55%",
                        "stabilized": "55%",
                        "reaction": "1s",
                        "rate_of_fire": "6.0 rpm",
                        "damage": 10,
                        "suppression": 14,
                        "penetration": 15
                    }
                ),
                "tanks": Agent(
                    "tanks", 0, 0, 1700, attack_range=3, v_max=3.0, health=150, max_health=150,
                    max_speed="75 km/h",
                    front_armor=17,
                    turret_armor="18 / 19",
                    side_armor=4,
                    rear_armor=3,
                    durability=20,
                    vision_range="2000 m",
                    night_vision="Thermal",
                    stealth_modifier=0,
                    has_smoke_grenades="Yes",
                    main_weapon={
                        "name": "120mm Gun",
                        "range": "2400 meters",
                        "accuracy": "75%",
                        "stabilized": "60%",
                        "reaction": "2s",
                        "rate_of_fire": "4.9 rpm",
                        "damage": 9,
                        "suppression": 12,
                        "penetration": 18
                    },
                    secondary_weapon={
                        "name": "7.62mm Machine Gun",
                        "range": "1200 meters",
                        "accuracy": "30%",
                        "stabilized": "18%",
                        "reaction": "1s",
                        "rate_of_fire": "197 rpm",
                        "damage": 0.8,
                        "suppression": 1,
                        "penetration": 1
                    }
                )
            },
            "enemy": Agent(
                "enemy", 9, 9, float('inf'), attack_range=3, v_max=3.0, health=150, max_health=150,
                max_speed="75 km/h",
                front_armor=17,
                turret_armor="18 / 19",
                side_armor=4,
                rear_armor=3,
                durability=20,
                vision_range="2000 m",
                night_vision="Thermal",
                stealth_modifier=0,
                has_smoke_grenades="Yes",
                main_weapon={
                    "name": "120mm Gun",
                    "range": "2400 meters",
                    "accuracy": "75%",
                    "stabilized": "60%",
                    "reaction": "2s",
                    "rate_of_fire": "4.9 rpm",
                    "damage": 9,
                    "suppression": 12,
                    "penetration": 18
                },
                secondary_weapon={
                    "name": "7.62mm Machine Gun",
                    "range": "1200 meters",
                    "accuracy": "30%",
                    "stabilized": "18%",
                    "reaction": "1s",
                    "rate_of_fire": "197 rpm",
                    "damage": 0.8,
                    "suppression": 1,
                    "penetration": 1
                }
            ),
            "mission_complete": False
        }