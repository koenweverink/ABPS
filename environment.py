import random
from terrain_utils import neighbors, get_line, in_bounds
from config import GRID_WIDTH, GRID_HEIGHT

# ---------- Tunables ----------
RIVER_Y_SHIFT = 10           # move river upward (increase y)
LEFT_BAND_WIDTH = 7
MID_Y0, MID_Y1 = 20, 25     # center band (inclusive start, exclusive end)
RIGHT_Y0, RIGHT_Y1 = 35, 40 # right band (inclusive start, exclusive end)

# 1-tile bridge positions
BRIDGE_X_LEFT = 12
BRIDGE_X_DIAG_ANCHOR = 40
BRIDGE_X_RIGHT = 62
# ------------------------------

def _shift_y(y):
    return max(0, min(GRID_HEIGHT - 1, y + RIVER_Y_SHIFT))

river = set()

# Left vertical band (thickness = LEFT_BAND_WIDTH), shifted up
river.update((x, _shift_y(y)) for x in range(LEFT_BAND_WIDTH) for y in range(0, MID_Y1))

# Center horizontal band
river.update((x, _shift_y(y)) for x in range(LEFT_BAND_WIDTH, 25) for y in range(MID_Y0, MID_Y1))

# Diagonal connector: five parallel lines for thickness
for i in range(5):
    y0, y1 = MID_Y0 + i, RIGHT_Y0 + i
    for pos in get_line((25, _shift_y(y0)), (49, _shift_y(y1))):
        if in_bounds(pos):
            river.add(pos)

# Right horizontal band
river.update((x, _shift_y(y)) for x in range(50, GRID_WIDTH) for y in range(RIGHT_Y0, RIGHT_Y1))

# --------- Ensure edge contact ---------
# Make sure the river touches the bottom-left border (y == 0 with x < LEFT_BAND_WIDTH).
# Extend straight down from the lowest y of the left band to y=0.
left_band_ys = [y for (x, y) in river if x < LEFT_BAND_WIDTH]
if left_band_ys:
    lowest_y_left = min(left_band_ys)
    river.update((x, y) for x in range(LEFT_BAND_WIDTH) for y in range(0, lowest_y_left + 1))
else:
    # Fallback: just lay a small vertical at the corner
    river.update((x, 0) for x in range(LEFT_BAND_WIDTH))

# Make sure the river touches the right edge (x == GRID_WIDTH-1) within the right band.
for y in range(_shift_y(RIGHT_Y0), _shift_y(RIGHT_Y1)):
    if in_bounds((GRID_WIDTH - 1, y)):
        river.add((GRID_WIDTH - 1, y))

# --------- Carve 1-tile bridges ---------
# Center horizontal band bridge
river.difference_update((BRIDGE_X_LEFT, _shift_y(y)) for y in range(MID_Y0, MID_Y1))

# Diagonal bridge: remove exactly one point per diagonal line near the anchor x
for i in range(5):
    y0, y1 = MID_Y0 + i, RIGHT_Y0 + i
    line = [p for p in get_line((25, _shift_y(y0)), (49, _shift_y(y1))) if in_bounds(p)]
    if not line:
        continue
    idx = min(range(len(line)), key=lambda k: abs(line[k][0] - BRIDGE_X_DIAG_ANCHOR))
    river.discard(line[idx])

# Right band bridge
river.difference_update((BRIDGE_X_RIGHT, _shift_y(y)) for y in range(RIGHT_Y0, RIGHT_Y1))


# --- Forest Generation ---

def init_forest(p, width, height):
    """Randomly populate forest cells based on probability p, excluding river tiles."""
    return {
        (x, y)
        for x in range(width)
        for y in range(height)
        if random.random() < p and (x, y) not in river
    }


def count_neighbors(forest, x, y):
    """Count how many of the 8 neighbors of a tile are also forested."""
    n = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            if (x + dx, y + dy) in forest:
                n += 1
    return n

def smooth_forest(forest, width, height, survive=4, birth=5):
    """Apply cellular automata smoothing rules to refine forest edges."""
    new = set()
    for x in range(width):
        for y in range(height):
            n = count_neighbors(forest, x, y)
            if ((x, y) in forest and n >= survive) or ((x, y) not in forest and n >= birth):
                new.add((x, y))
    return new

forest = init_forest(p=0.45, width=GRID_WIDTH, height=GRID_HEIGHT)
for _ in range(4):
    forest = smooth_forest(forest, GRID_WIDTH, GRID_HEIGHT)


# --- Cliff Definitions ---

cliff_defs = [((36, 10), (44, 12), (0, 1))]
cliffs = {}
for s, e, n in cliff_defs:
    for c in get_line(s, e):
        cliffs[c] = n

# Map from entry points to climbable cliff tiles
climb_entries = {
    (cx - nx, cy - ny): (cx, cy)
    for (cx, cy), (nx, ny) in cliffs.items()
}

# --- Forest Edge Cleanup ---

forest_edge = {
    pos
    for pos in forest
    if any(
        n not in forest
        for n in neighbors(
            pos, in_bounds, river=river, cliffs=cliffs, climb_entries=climb_entries
        )
    )
}
forest -= forest_edge
