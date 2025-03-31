# terrain.py

class Terrain:
    def __init__(self, width, height):
        """
        Initialize a terrain with the given width and height.
        The map is initially empty (all zeros).
        """
        self.width = width
        self.height = height
        self.map = [[0 for _ in range(width)] for _ in range(height)]

    def load_map(self, terrain_map):
        """
        Load a terrain map into the Terrain object.
        The map should be a 2D list of 0s (passable) and 1s (obstacles).
        """
        if len(terrain_map) != self.height or any(len(row) != self.width for row in terrain_map):
            raise ValueError(f"Terrain map dimensions must match {self.height}x{self.width}")
        self.map = [row[:] for row in terrain_map]  # Deep copy to avoid modifying the input

    def is_obstacle(self, x, y):
        """
        Check if the position (x, y) is an obstacle (1).
        Returns True if the position is an obstacle or out of bounds, False otherwise.
        """
        if not (0 <= x < self.height and 0 <= y < self.width):
            return True  # Out of bounds is treated as an obstacle
        return self.map[x][y] == 1

    def line_of_sight(self, x1, y1, x2, y2):
        """
        Check if there is a clear line of sight between (x1, y1) and (x2, y2).
        Returns True if there are no obstacles in the way, False otherwise.
        Uses Bresenham's line algorithm to trace the path.
        """
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2
        for _ in range(n):
            if (x, y) != (x1, y1) and (x, y) != (x2, y2):  # Skip start and end points
                if self.is_obstacle(x, y):
                    return False
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        return True

# Example terrain map (for testing purposes, can be removed if loaded elsewhere)
terrain_map = [
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
]