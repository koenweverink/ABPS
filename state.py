# state.py (updated)
class Terrain:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_obstacle(self, x, y):
        return (x, y) in self.obstacles

class Agent:
    def __init__(self, x, y, fuel, health, attack_range, accuracy, damage, suppression_per_hit, weapon):
        self.x = x
        self.y = y
        self.fuel = fuel
        self.health = health
        self.attack_range = attack_range
        self.accuracy = accuracy
        self.damage = damage
        self.suppression_per_hit = suppression_per_hit
        self.weapon = weapon

class Enemy:
    def __init__(self, x, y, health, accuracy, damage, suppression, weapon, attack_range=3):  # Added attack_range
        self.x = x
        self.y = y
        self.health = health
        self.accuracy = accuracy
        self.damage = damage
        self.suppression = suppression
        self.weapon = weapon
        self.attack_range = attack_range  # Added attack_range attribute

class State:
    def __init__(self, terrain, agents, enemy):
        self.terrain = terrain
        self.data = {
            "agents": agents,
            "enemy": enemy,
            "mission_complete": False
        }

    def get_enemy_position(self):
        return self.data["enemy"].x, self.data["enemy"].y

    def has_line_of_sight(self, x1, y1, x2, y2):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1
        err = dx - dy

        while True:
            if self.terrain.is_obstacle(x, y):
                return False
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return True

def initialize_state():
    terrain = Terrain(10, 10, [(0, 3), (1, 1), (1, 3), (1, 6), (2, 8), (3, 0), (3, 1), (3, 4), (3, 6), (4, 3), (4, 7), (5, 1), (5, 8), (6, 3), (6, 4), (6, 7), (7, 9), (8, 1), (8, 5), (8, 7), (9, 5), (9, 7)])
    agents = {
        "infantry": Agent(x=0, y=0, fuel=800, health=75, attack_range=2, accuracy=60, damage=10, suppression_per_hit=0.14, weapon="M72 LAW"),
        "tanks": Agent(x=0, y=0, fuel=1700, health=150, attack_range=3, accuracy=80, damage=9, suppression_per_hit=0.12, weapon="120mm Gun")
    }
    enemy = Enemy(x=9, y=9, health=150, accuracy=75, damage=9, suppression=0.0, weapon="120mm Gun", attack_range=3)
    return State(terrain, agents, enemy)