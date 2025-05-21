from log import logger
from config import GRID_WIDTH, GRID_HEIGHT
import numpy as np

class Drone:
    def __init__(self, side, target_side, n_cols=3, n_rows=2, stay_rounds=10, spot_prob=0.2):
        # build 6 (xmin,xmax,ymin,ymax) areas
        w, h = GRID_WIDTH / n_cols, GRID_HEIGHT / n_rows
        self.areas = [
            (j*w, (j+1)*w, i*h, (i+1)*h)
            for i in range(n_rows)
            for j in range(n_cols)
        ]
        self.stay_rounds = stay_rounds
        self.spot_prob = spot_prob
        self.rounds_in_area = 0
        self.side = side
        self.target_side = target_side
        self.current_area = 3 if self.side == "friendly" else 2
        self.route = [3, 4, 5, 2, 1, 0] if self.side == "friendly" else [2, 1, 0, 3, 4, 5]
        self.route_idx = 0

        # maps unit_name -> (last_x, last_y)
        self.last_known = {}

    def _in_area(self, pos, bounds):
        x, y = pos
        xmin, xmax, ymin, ymax = bounds
        return xmin <= x < xmax and ymin <= y < ymax
    
    def update(self, sim):
        """
        Call once per Simulation.step().
        Spots any friendly or enemy in the current area,
        then advances the area counter every stay_rounds.
        """
        logger.info(f"Drone({self.side}) update: target_side={self.target_side}")
        if self.target_side == "friendly":
            units = sim.friendly_units
        else:
            units = sim.enemy_units
        
        bounds = self.areas[self.current_area]

        # try spotting every unit in this area
        in_area = [u for u in units 
               if u.state.get("health",0)>0 
               and self._in_area(u.state["position"], bounds)]
        
        if in_area:
            # Use Poisson approximation to sample the number of spots
            # 1) compute λ = sum of individual spot-probs
            ps = [self.spot_prob for _ in in_area]  # or customize per-unit
            lam = sum(ps)
            # 2) sample how many spots this round
            k = np.random.poisson(lam)
            k = min(k, len(in_area))
            # 3) pick k units to spot (here uniform)
            spotted = np.random.choice(in_area, size=k, replace=False)
            for u in spotted:
                self.last_known[u.name] = u.state["position"]
                logger.info(f"Drone ({self.side}) spotted {u.name} at {u.state['position']}")

        # move on after enough rounds
        self.rounds_in_area += 1
        if self.rounds_in_area >= self.stay_rounds:
            self.rounds_in_area = 0
            self.route_idx = (self.route_idx + 1) % len(self.route)
            self.current_area = self.route[self.route_idx]
            logger.info(f"{self.side}'s drone moves to area #{self.current_area + 1}")

