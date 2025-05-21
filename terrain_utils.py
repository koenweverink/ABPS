def get_line(start, end):
    """Bresenham-style line drawing between two grid points"""
    x1, y1 = start
    x2, y2 = end
    line = []
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x2:
            line.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            line.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    line.append((x, y))
    return line


def in_bounds(pos, width=75, height=50):
    """Checks if a position is inside the grid"""
    x, y = pos
    return 0 <= x < width and 0 <= y < height


def manhattan(p, q):
    """Manhattan distance between two points"""
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def neighbors(pos, in_bounds_fn, river=None, cliffs=None, climb_entries=None):
    """Returns valid neighbor cells considering river/cliff constraints"""
    results = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        p = (pos[0] + dx, pos[1] + dy)
        if not in_bounds_fn(p):
            continue
        if river and p in river:
            continue
        if cliffs:
            if pos in cliffs:
                nx, ny = cliffs[pos]
                if p != (pos[0] - nx, pos[1] - ny):
                    continue
            if p in cliffs and climb_entries and climb_entries.get(pos) != p:
                continue
        results.append(p)
    return results


def sign(x):
    """Returns the sign of a number: -1, 0, or 1"""
    return 1 if x > 0 else -1 if x < 0 else 0
