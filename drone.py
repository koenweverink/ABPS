import numpy as np
from config import GRID_WIDTH, GRID_HEIGHT
from log import logger


class Drone:
    """
    Simulates a surveillance drone that scans a predefined grid of areas,
    spotting units based on proximity and probability.
    """

    def __init__(self, side, target_side, n_cols=3, n_rows=2, stay_rounds=10, spot_prob=1):
        """
        Initialize a Drone instance.

        Args:
            side (str): The controlling side ('friendly' or 'enemy').
            target_side (str): The side whose units should be observed.
            n_cols (int): Number of grid columns.
            n_rows (int): Number of grid rows.
            stay_rounds (int): Rounds to stay in one area before moving.
            spot_prob (float): Probability of spotting a unit per candidate.
        """
        self.side = side
        self.target_side = target_side
        self.stay_rounds = stay_rounds
        self.spot_prob = spot_prob
        self.rounds_in_area = 0

        self.areas = self._define_areas(n_cols, n_rows)
        self.route = self._define_route()
        self.route_idx = 0
        self.current_area = self.route[self.route_idx]

        self.last_known = {}  # unit_name -> (x, y)

    def _define_areas(self, n_cols, n_rows):
        """Divide the grid into rectangular scanning areas."""
        w, h = GRID_WIDTH / n_cols, GRID_HEIGHT / n_rows
        return [
            (j * w, (j + 1) * w, i * h, (i + 1) * h)
            for i in range(n_rows)
            for j in range(n_cols)
        ]

    def _define_route(self):
        """Return the area scan order based on side logic."""
        # return [3, 4, 5, 2, 1, 0] if self.side == "friendly" else [2, 1, 0, 3, 4, 5]
        return [3, 0, 1, 2, 5, 4] if self.side == "friendly" else [2, 5, 4, 3, 0, 1]

    def _in_area(self, pos, bounds):
        """Check if a given position is inside a rectangular area."""
        x, y = pos
        xmin, xmax, ymin, ymax = bounds
        return xmin <= x < xmax and ymin <= y < ymax

    def _spot_units(self, units, bounds):
        """
        Spot units within the current area based on probability.

        Args:
            units (list): List of units to potentially spot.
            bounds (tuple): (xmin, xmax, ymin, ymax) of the current area.
        """
        candidates = [
            u for u in units
            if u.state.get("health", 0) > 0 and self._in_area(u.state["position"], bounds)
        ]

        if not candidates:
            return

        lam = self.spot_prob * len(candidates)
        k = min(np.random.poisson(lam), len(candidates))
        spotted = np.random.choice(candidates, size=k, replace=False)

        for u in spotted:
            self.last_known[u.name] = u.state["position"]
            logger.info(f"Drone ({self.side}) spotted {u.name} at {u.state['position']}")

    def update(self, sim):
        """
        Called once per simulation step.
        Attempts to spot enemy or friendly units in the current drone area.
        Then updates its scanning position if it has stayed long enough.
        """
        logger.info(f"Drone({self.side}) update: target_side={self.target_side}")

        units = sim.friendly_units if self.target_side == "friendly" else sim.enemy_units
        bounds = self.areas[self.current_area]

        self._spot_units(units, bounds)

        self.rounds_in_area += 1
        if self.rounds_in_area >= self.stay_rounds:
            self.rounds_in_area = 0
            self.route_idx = (self.route_idx + 1) % len(self.route)
            self.current_area = self.route[self.route_idx]
            logger.info(f"{self.side}'s drone moves to area #{self.current_area + 1}")
