import random

from terrain_utils import neighbors, get_line, in_bounds
from config import GRID_WIDTH, GRID_HEIGHT

river = set()
for x in range(0, GRID_WIDTH):
    for y in range(0, GRID_HEIGHT):
        if (x < 7 and y < 25):
            river.add((x, y))
        elif (7 <= x < 25 and 20 <= y < 25):
            river.add((x, y))
        elif (x >= 50 and 35 <= y < 40):
            river.add((x, y))
        for i in range(0, 5):
            for pos in get_line((25, 20+i), (49, 35+i)):
                river.add(pos)

for x in range(11, 14):
    for y in range(20, 25):
        river.discard((x, y))

for i in range(6):
    for pos in get_line((40, 21 + i), (34, 29 + i)):
        river.discard(pos)

for x in range(61, 64):
    for y in range(35, 40):
        river.discard((x, y))

def init_forest(p, width, height):
    return {
        (x,y)
        for x in range(width)
        for y in range(height)
        if random.random() < p and (x,y) not in river
    }

def count_neighbors(forest, x, y):
    n = 0
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            if dx==0 and dy==0: continue
            if (x+dx, y+dy) in forest:
                n += 1
    return n

def smooth_forest(forest, width, height, survive=4, birth=5):
    new = set()
    for x in range(width):
        for y in range(height):
            n = count_neighbors(forest, x, y)
            if ( (x,y) in forest and n >= survive ) or ( (x,y) not in forest and n >= birth ):
                new.add((x,y))
    return new

forest = init_forest(p=0.45, width=GRID_WIDTH, height=GRID_HEIGHT)
for _ in range(4):
    forest = smooth_forest(forest, GRID_WIDTH, GRID_HEIGHT)

cliff_defs = [((36,10),(44,12),(0, 1))]
cliffs = {}
for s,e,n in cliff_defs:
    for c in get_line(s,e):
        cliffs[c] = n

climb_entries = {
    (cx - nx, cy - ny): (cx, cy)
    for (cx, cy), (nx, ny) in cliffs.items()
}

forest_edge = [pos for pos in forest if any(n not in forest for n in neighbors(pos, in_bounds, river=river, cliffs=cliffs, climb_entries=climb_entries))]
forest -= set(forest_edge)