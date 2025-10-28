import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib import ticker
from utils import get_all_enemy_attack_zones
from config import GRID_WIDTH, GRID_HEIGHT, CELL_SIZE
from log import logger

# Style mapping for unit types
STYLE_MAP_ENEMY = {
    "tank":      {"marker": "s", "color": "darkgreen",  "label": "Enemy Tanks"},
    "infantry":  {"marker": "^", "color": "saddlebrown","label": "Enemy Infantry"},
    "anti-tank": {"marker": "X", "color": "gray",       "label": "Enemy AT"},
    "artillery": {"marker": "D", "color": "purple",     "label": "Enemy Artillery"},
}
STYLE_DEFAULT_ENEMY = {"marker": "o", "color": "green", "label": "Enemy"}

STYLE_MAP_FRIENDLY = {
    "tank":      {"marker": "s", "color": "red",      "label": "Tank"},
    "infantry":  {"marker": "o", "color": "blue",     "label": "Infantry"},
    "anti-tank": {"marker": "X", "color": "purple",   "label": "Anti-tank"},
    "artillery": {"marker": "D", "color": "orange",   "label": "Artillery"},
    "scout":     {"marker": "^", "color": "cyan",     "label": "Scout"},
}
STYLE_DEFAULT_FRIENDLY = {"marker": "o", "color": "gray", "label": "Friendly"}

class SimulationPlotter:
    def __init__(self, sim, visualize=True):
        self.sim = sim
        self.visualize = visualize
        self.paused = False

        # Initialize Matplotlib figure and axes with wider figure for text blocks
        self.fig, self.ax = plt.subplots(figsize=(18, 8))
        if not visualize:
            # Even in non-visual mode, we need to set up the plot for animation
            plt.ioff()  # Turn off interactive mode
            return
        plt.ion()
        plt.show(block=False)

        # Prepare lists to hold dynamic artists
        self.enemy_markers = {}
        self.enemy_arrows = {}
        self.enemy_texts = {}
        self.enemy_name_texts = {}
        self.enemy_attack_indicators = {}

        self.friendly_markers = {}
        self.friendly_arrows = {}
        self.friendly_texts = {}
        self.friendly_name_texts = {}
        self.friendly_attack_indicators = {}

        self.staging_markers = {}  # New: staging position markers
        self.flank_markers = {}
        self.defend_markers = {}   # Defensive position markers

        # Build the static background (terrain, grid, arrows, outpost, etc.)
        self._init_plot()

        # Capture static background for blitting
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        # Create two persistent Rectangle patches for the drones with different colors
        self.drone_patches = {}
        for drone in (self.sim.friendly_drone, self.sim.enemy_drone):
            xmin, xmax, ymin, ymax = drone.areas[drone.current_area]
            # Set edgecolor based on drone side
            edgecolor = "blue" if drone.side == "friendly" else "red"
            rect = Rectangle(
                (xmin * CELL_SIZE, ymin * CELL_SIZE),
                (xmax - xmin) * CELL_SIZE,
                (ymax - ymin) * CELL_SIZE,
                linewidth=2,
                edgecolor=edgecolor,
                facecolor="none",
                zorder=5
            )
            self.ax.add_patch(rect)
            self.drone_patches[drone] = rect

        # Connect pause/resume on 'p' key
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Force initial update to draw units at their starting positions
        if visualize:
            self.update()
        else:
            # In non-visual mode, still need to initialize the plot
            self._init_plot()
            self._setup_unit_artists()
            # Capture static background for non-visual mode
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def _init_plot(self):
        # --- build terrain RGB array ---
        grid = np.ones((GRID_HEIGHT, GRID_WIDTH, 3), dtype=float)

        # water = blue
        for (x, y) in self.sim.river:
            grid[y, x] = [0.0, 0.0, 1.0]
        # deep forest = dark green
        for (x, y) in self.sim.forest:
            grid[y, x] = [0.0, 0.5, 0.0]
        # forest edge = light green
        for (x, y) in self.sim.forest_edge:
            grid[y, x] = [0.7, 1.0, 0.7]
        # cliffs = brown
        for (x, y) in self.sim.cliffs:
            grid[y, x] = [0.6, 0.3, 0.0]

        # Draw the terrain
        self.ax.imshow(
            grid,
            origin='lower',
            extent=(
                -CELL_SIZE/2,
                GRID_WIDTH*CELL_SIZE - CELL_SIZE/2,
                -CELL_SIZE/2,
                GRID_HEIGHT*CELL_SIZE - CELL_SIZE/2
            ),
            zorder=0
        )

        # Draw climb-entry arrows
        for entry, cliff_cell in self.sim.climb_entries.items():
            ex, ey = entry
            cx, cy = cliff_cell
            self.ax.annotate(
                '',
                xy=(cx*CELL_SIZE + CELL_SIZE/2, cy*CELL_SIZE + CELL_SIZE/2),
                xytext=(ex*CELL_SIZE + CELL_SIZE/2, ey*CELL_SIZE + CELL_SIZE/2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1),
                zorder=1
            )

        # Draw outpost if defined on the sim or infer from unit state
        outpost_pos = getattr(self.sim, 'outpost_position', None)
        if not outpost_pos:
            # Try to infer from friendly units' state
            for u in getattr(self.sim, 'friendly_units', []):
                candidate = u.state.get('outpost_position')
                if candidate:
                    outpost_pos = candidate
                    break
        if not outpost_pos:
            # Fallback: try enemy units (in case it's stored there)
            for e in getattr(self.sim, 'enemy_units', []):
                candidate = e.state.get('outpost_position')
                if candidate:
                    outpost_pos = candidate
                    break
        if outpost_pos:
            ox, oy = outpost_pos
            px = ox * CELL_SIZE + CELL_SIZE / 2
            py = oy * CELL_SIZE + CELL_SIZE / 2
            self.ax.plot(
                px,
                py,
                marker='*', markersize=12,
                color='magenta', zorder=6
            )
            # Small label near the marker
            self.ax.text(
                px + 8, py + 8, "Outpost",
                fontsize=8, color='magenta', zorder=6,
                ha='left', va='bottom'
            )

        # Configure axes
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-CELL_SIZE/2, GRID_WIDTH * CELL_SIZE - CELL_SIZE/2)
        self.ax.set_ylim(-CELL_SIZE/2, GRID_HEIGHT * CELL_SIZE - CELL_SIZE/2)

        # Improved axis spacing to prevent overlap
        major_step = 1000  # Increased step size
        self.ax.set_xticks(np.arange(0, (GRID_WIDTH+1)*CELL_SIZE, major_step))
        self.ax.set_yticks(np.arange(0, (GRID_HEIGHT+1)*CELL_SIZE, major_step))
        
        # Format axes with better spacing and readability
        self.ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}m"))
        self.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}m"))
        
        # Rotate x-axis labels to prevent overlap
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add minor ticks for better grid visibility
        self.ax.set_xticks(np.arange(0, (GRID_WIDTH+1)*CELL_SIZE, 500), minor=True)
        self.ax.set_yticks(np.arange(0, (GRID_HEIGHT+1)*CELL_SIZE, 500), minor=True)
        
        self.ax.grid(True, alpha=0.3)
        self.ax.grid(True, which='minor', alpha=0.1)

        # Create artists for each enemy with initial positions
        seen_types = set()
        for enemy in self.sim.enemy_units:
            typ = enemy.state.get('type')
            style = STYLE_MAP_ENEMY.get(typ, STYLE_DEFAULT_ENEMY)
            label = style['label'] if typ not in seen_types else None
            seen_types.add(typ)

            # Get initial position
            x, y = enemy.state.get('position', (0, 0))
            cx = x * CELL_SIZE + CELL_SIZE/2
            cy = y * CELL_SIZE + CELL_SIZE/2

            # Scatter marker
            m = self.ax.scatter(cx, cy, marker=style['marker'],
                                s=100, color=style['color'], label=label, zorder=5)
            self.enemy_markers[enemy.name] = m
            # Facing arrow
            fx, fy = enemy.state.get('facing', (0, 1))
            norm = (fx**2 + fy**2)**0.5
            fx, fy = (fx/norm, fy/norm) if norm > 0 else (0, 1)
            a = self.ax.quiver(cx, cy, fx*CELL_SIZE*1.2, fy*CELL_SIZE*1.2,
                               color=style['color'], edgecolor='black', linewidth=0.5,
                               width=0.008, scale=1, scale_units='xy', angles='xy', zorder=4)
            self.enemy_arrows[enemy.name] = a
            # Health text
            t = self.ax.text(cx + 15, cy + 15,
                             f"{enemy.state['current_group_size']}/{enemy.state['health']:.0f}",
                             fontsize=6, color='black', zorder=6)
            self.enemy_texts[enemy.name] = t
            # Name text
            nt = self.ax.text(cx, cy - CELL_SIZE*0.3, enemy.name,
                              ha='center', va='top', fontsize='small', color='black', zorder=6)
            self.enemy_name_texts[enemy.name] = nt
            # Attack-indicator ring
            ring = Circle(
                (cx, cy),
                radius=CELL_SIZE*0.8,
                linewidth=3,
                edgecolor='black',
                facecolor='red',
                alpha=1,
                visible=False,
                zorder=1,
            )
            self.ax.add_patch(ring)
            self.enemy_attack_indicators[enemy.name] = ring

        # Create artists for each friendly with initial positions
        seen_types = set()
        for unit in self.sim.friendly_units:
            typ = unit.state.get('type')
            style = STYLE_MAP_FRIENDLY.get(typ, STYLE_DEFAULT_FRIENDLY)
            label = style['label'] if typ not in seen_types else None
            seen_types.add(typ)

            # Get initial position
            x, y = unit.state.get('position', (0, 0))
            cx = x * CELL_SIZE + CELL_SIZE/2
            cy = y * CELL_SIZE + CELL_SIZE/2

            # Scatter marker
            m = self.ax.scatter(cx, cy, marker=style['marker'],
                                s=100, color=style['color'], label=label, zorder=5)
            self.friendly_markers[unit.name] = m
            # Facing arrow
            fx, fy = unit.state.get('facing', (0, 1))
            norm = (fx**2 + fy**2)**0.5
            fx, fy = (fx/norm, fy/norm) if norm > 0 else (0, 1)
            a = self.ax.quiver(cx, cy, fx*CELL_SIZE*1.2, fy*CELL_SIZE*1.2,
                               color=style['color'], edgecolor='black', linewidth=0.5,
                               width=0.008, scale=1, scale_units='xy', angles='xy', zorder=4)
            self.friendly_arrows[unit.name] = a
            # Health text
            t = self.ax.text(cx + 15, cy + 15,
                             f"{unit.state['current_group_size']}/{unit.state['health']:.0f}",
                             fontsize=6, color='black', zorder=6)
            self.friendly_texts[unit.name] = t
            # Name text
            nt = self.ax.text(cx, cy - CELL_SIZE*0.3, unit.name,
                              ha='center', va='top', fontsize='small', color='black', zorder=6)
            self.friendly_name_texts[unit.name] = nt
            # Attack-indicator ring
            ring = Circle(
                (cx, cy),
                radius=CELL_SIZE*0.8,
                linewidth=3,
                edgecolor='black',
                facecolor='orange',
                alpha=1,
                visible=False,
                zorder=1,
            )
            self.ax.add_patch(ring)
            self.friendly_attack_indicators[unit.name] = ring

        # Make room on the right for the text panels (removed LOS)
        self.fig.subplots_adjust(right=0.65)

        # Task/Plan box - positioned outside plot area with better spacing
        self.task_text = self.ax.text(
            1.05, 1, "", transform=self.ax.transAxes,
            ha="left", va="top", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue",
                      edgecolor="navy", alpha=0.9, linewidth=1.5)
        )
        
        # Spotted units box - positioned outside plot area with better spacing
        self.spotted_text = self.ax.text(
            1.05, 0.5, "", transform=self.ax.transAxes,
            ha="left", va="top", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="lightgreen", edgecolor="darkgreen", alpha=0.9, linewidth=1.5)
        )
    
    def _setup_unit_artists(self):
        """Set up unit artists for non-visual mode."""
        # Prepare lists to hold dynamic artists
        self.enemy_markers = {}
        self.enemy_arrows = {}
        self.enemy_texts = {}
        self.enemy_name_texts = {}
        self.enemy_attack_indicators = {}

        self.friendly_markers = {}
        self.friendly_arrows = {}
        self.friendly_texts = {}
        self.friendly_name_texts = {}
        self.friendly_attack_indicators = {}

        self.staging_markers = {}  # New: staging position markers
        self.flank_markers = {}
        self.defend_markers = {}   # Defensive position markers

        # Create artists for each enemy with initial positions
        seen_types = set()
        for enemy in self.sim.enemy_units:
            typ = enemy.state.get('type')
            style = STYLE_MAP_ENEMY.get(typ, STYLE_DEFAULT_ENEMY)
            label = style['label'] if typ not in seen_types else None
            seen_types.add(typ)

            # Get initial position
            x, y = enemy.state.get('position', (0, 0))
            cx = x * CELL_SIZE + CELL_SIZE/2
            cy = y * CELL_SIZE + CELL_SIZE/2

            # Scatter marker
            m = self.ax.scatter(cx, cy, marker=style['marker'],
                                s=100, color=style['color'], label=label, zorder=5)
            self.enemy_markers[enemy.name] = m
            # Facing arrow
            fx, fy = enemy.state.get('facing', (0, 1))
            norm = (fx**2 + fy**2)**0.5
            fx, fy = (fx/norm, fy/norm) if norm > 0 else (0, 1)
            a = self.ax.quiver(cx, cy, fx*CELL_SIZE*1.2, fy*CELL_SIZE*1.2,
                               color=style['color'], edgecolor='black', linewidth=0.5,
                               width=0.008, scale=1, scale_units='xy', angles='xy', zorder=4)
            self.enemy_arrows[enemy.name] = a
            # Health text
            t = self.ax.text(cx + 15, cy + 15,
                             f"{enemy.state['current_group_size']}/{enemy.state['health']:.0f}",
                             fontsize=6, color='black', zorder=6)
            self.enemy_texts[enemy.name] = t
            # Name text
            nt = self.ax.text(cx, cy - CELL_SIZE*0.3, enemy.name,
                              ha='center', va='top', fontsize='small', color='black', zorder=6)
            self.enemy_name_texts[enemy.name] = nt
            # Attack-indicator ring
            ring = Circle(
                (cx, cy),
                radius=CELL_SIZE*0.8,
                linewidth=3,
                edgecolor='black',
                facecolor='red',
                alpha=1,
                visible=False,
                zorder=1,
            )
            self.ax.add_patch(ring)
            self.enemy_attack_indicators[enemy.name] = ring

        # Create artists for each friendly with initial positions
        seen_types = set()
        for unit in self.sim.friendly_units:
            typ = unit.state.get('type')
            style = STYLE_MAP_FRIENDLY.get(typ, STYLE_DEFAULT_FRIENDLY)
            label = style['label'] if typ not in seen_types else None
            seen_types.add(typ)

            # Get initial position
            x, y = unit.state.get('position', (0, 0))
            cx = x * CELL_SIZE + CELL_SIZE/2
            cy = y * CELL_SIZE + CELL_SIZE/2

            # Scatter marker
            m = self.ax.scatter(cx, cy, marker=style['marker'],
                                s=100, color=style['color'], label=label, zorder=5)
            self.friendly_markers[unit.name] = m
            # Facing arrow
            fx, fy = unit.state.get('facing', (0, 1))
            norm = (fx**2 + fy**2)**0.5
            fx, fy = (fx/norm, fy/norm) if norm > 0 else (0, 1)
            a = self.ax.quiver(cx, cy, fx*CELL_SIZE*1.2, fy*CELL_SIZE*1.2,
                               color=style['color'], edgecolor='black', linewidth=0.5,
                               width=0.008, scale=1, scale_units='xy', angles='xy', zorder=4)
            self.friendly_arrows[unit.name] = a
            # Health text
            t = self.ax.text(cx + 15, cy + 15,
                             f"{unit.state['current_group_size']}/{unit.state['health']:.0f}",
                             fontsize=6, color='black', zorder=6)
            self.friendly_texts[unit.name] = t
            # Name text
            nt = self.ax.text(cx, cy - CELL_SIZE*0.3, unit.name,
                              ha='center', va='top', fontsize='small', color='black', zorder=6)
            self.friendly_name_texts[unit.name] = nt
            # Attack-indicator ring
            ring = Circle(
                (cx, cy),
                radius=CELL_SIZE*0.8,
                linewidth=3,
                edgecolor='black',
                facecolor='orange',
                alpha=1,
                visible=False,
                zorder=1,
            )
            self.ax.add_patch(ring)
            self.friendly_attack_indicators[unit.name] = ring

        # Create two persistent Rectangle patches for the drones with different colors
        self.drone_patches = {}
        for drone in (self.sim.friendly_drone, self.sim.enemy_drone):
            xmin, xmax, ymin, ymax = drone.areas[drone.current_area]
            # Set edgecolor based on drone side
            edgecolor = "blue" if drone.side == "friendly" else "red"
            rect = Rectangle(
                (xmin * CELL_SIZE, ymin * CELL_SIZE),
                (xmax - xmin) * CELL_SIZE,
                (ymax - ymin) * CELL_SIZE,
                linewidth=2,
                edgecolor=edgecolor,
                facecolor="none",
                zorder=5
            )
            self.ax.add_patch(rect)
            self.drone_patches[drone] = rect

    def _on_key(self, event):
        # toggle pause when user presses 'p'
        if event.key == 'p':
            self.paused = not self.paused
            state = "PAUSED" if self.paused else "RUNNING"
            print(f"[Visualization {state}]")

    def update(self):
        if self.visualize and self.paused:
            return

        # only redraw if someone's state changed (skip this check in non-visual mode)
        if self.visualize:
            needs = any(u.needs_update()
                        for u in self.sim.enemy_units + self.sim.friendly_units)
            if not needs:
                return

            # restore static background
            self.fig.canvas.restore_region(self.background)

        # Clear previous avoid zone patches
        if hasattr(self, "avoid_patches"):
            for patch in self.avoid_patches:
                patch.remove()
        self.avoid_patches = []

        avoid_zones = get_all_enemy_attack_zones(self.sim)

        for pos in avoid_zones:
            x, y = pos
            cx = x * CELL_SIZE + CELL_SIZE / 2
            cy = y * CELL_SIZE + CELL_SIZE / 2
            rect = plt.Rectangle(
                (x * CELL_SIZE - CELL_SIZE / 2, y * CELL_SIZE - CELL_SIZE / 2),
                CELL_SIZE, CELL_SIZE,
                linewidth=0.2,
                edgecolor='none',
                facecolor='magenta',
                alpha=0.2,
                zorder=1
            )
            self.ax.add_patch(rect)
            self.avoid_patches.append(rect)


        # update title
        self.ax.set_title(f"Step {self.sim.step_count} - {self.sim.plan_name}")

        def _update_group(units, markers, arrows, texts, names, rings):
            for u in units:
                name = u.name
                alive = u.state.get("enemy_alive", u.state.get("health", 0) > 0)

                # Fetch artists by unit name
                marker = markers.get(name)
                arrow = arrows.get(name)
                text = texts.get(name)
                name_text = names.get(name)
                ring = rings.get(name)

                if not all([marker, arrow, text, name_text, ring]):
                    continue  # skip units that weren't initialized (shouldn't happen)

                if not alive:
                    marker.set_visible(False)
                    arrow.set_visible(False)
                    text.set_visible(False)
                    name_text.set_visible(False)
                    ring.set_visible(False)
                    continue

                # Make visible and update positions
                x, y = u.state['position']
                cx = x * CELL_SIZE + CELL_SIZE/2
                cy = y * CELL_SIZE + CELL_SIZE/2

                marker.set_visible(True)
                marker.set_offsets((cx, cy))

                fx, fy = u.state.get('facing', (0, 0))
                norm = (fx**2 + fy**2)**0.5
                fx, fy = (fx/norm, fy/norm) if norm > 0 else (0, 1)
                arrow.set_visible(True)
                arrow.set_UVC(fx*CELL_SIZE*1.2, fy*CELL_SIZE*1.2)
                arrow.set_offsets((cx, cy))

                text.set_visible(True)
                text.set_position((cx + 15, cy + 15))
                text.set_text(f"{u.state['current_group_size']}/{u.state['health']:.0f}")

                name_text.set_visible(True)
                name_text.set_position((cx, cy - CELL_SIZE*0.3))

                if u.state.get('is_attacking', False):
                    ring.center = (cx, cy)
                    ring.set_visible(True)
                else:
                    ring.set_visible(False)
                
                # update staging markers
                staging = u.state.get("staging_position")
                marker = self.staging_markers.get(name)
                if staging and alive and u.current_plan and u.current_plan[0][0] == "MoveToStaging":
                    sx, sy = staging
                    scx = sx * CELL_SIZE + CELL_SIZE / 2
                    scy = sy * CELL_SIZE + CELL_SIZE / 2
                    color = "black"
                    if u.state.get("staging_type") == "flank":
                        color = "gold"
                    elif u.state.get("staging_type") == "front":
                        color = "red"
                    if marker:
                        marker.set_offsets((scx, scy))
                        marker.set_color(color)
                        marker.set_visible(True)
                    else:
                        m = self.ax.scatter(scx, scy, marker="X", color=color, s=80, zorder=3)
                        self.staging_markers[name] = m
                elif marker:
                    marker.set_visible(False)

                # update flank markers (distinct from staging)
                # Draw flank markers
                flank = u.state.get("flank_position")
                flank_marker_key = name
                marker = self.flank_markers.get(flank_marker_key)
                if flank and alive and u.current_plan and u.current_plan[0][0] == "MoveToFlank":
                    fx, fy = flank
                    fcx = fx * CELL_SIZE + CELL_SIZE / 2
                    fcy = fy * CELL_SIZE + CELL_SIZE / 2
                    if marker:
                        marker.set_offsets((fcx, fcy))
                        marker.set_visible(True)
                    else:
                        m = self.ax.scatter(fcx, fcy, marker="+", color="gold", s=80, zorder=3)
                        self.flank_markers[flank_marker_key] = m
                elif marker:
                    marker.set_visible(False)

                logger.info(f"{u.name}: staging_type={u.state.get('staging_type')}, staging={u.state.get('staging_position')}, flank={u.state.get('flank_position')}")


                # update defend markers
                defend = u.state.get("defend_position")
                marker = self.defend_markers.get(name)
                if defend and alive:
                    dx, dy = defend
                    dcx = dx * CELL_SIZE + CELL_SIZE / 2
                    dcy = dy * CELL_SIZE + CELL_SIZE / 2
                    if marker:
                        marker.set_offsets((dcx, dcy))
                        marker.set_visible(True)
                    else:
                        m = self.ax.scatter(dcx, dcy, marker="P", color="gray", s=70, zorder=3)
                        self.defend_markers[name] = m
                elif marker:
                    marker.set_visible(False)

        # update enemies
        if hasattr(self, 'enemy_markers'):
            _update_group(self.sim.enemy_units,
                        self.enemy_markers, self.enemy_arrows,
                        self.enemy_texts, self.enemy_name_texts,
                        self.enemy_attack_indicators)

        # remove enemy visuals for units no longer in simulation
        if hasattr(self, 'enemy_markers'):
            live_enemy_names = {u.name for u in self.sim.enemy_units}
            for name in list(self.enemy_markers.keys()):
                if name not in live_enemy_names:
                    self._remove_unit_visuals(name, group='enemy')

        # update friendlies
        if hasattr(self, 'friendly_markers'):
            _update_group(self.sim.friendly_units,
                        self.friendly_markers, self.friendly_arrows,
                        self.friendly_texts, self.friendly_name_texts,
                        self.friendly_attack_indicators)

        # remove friendly visuals for units no longer in simulation
        if hasattr(self, 'friendly_markers'):
            live_friendly_names = {u.name for u in self.sim.friendly_units}
            for name in list(self.friendly_markers.keys()):
                if name not in live_friendly_names:
                    self._remove_unit_visuals(name, group='friendly')

        # move the persistent drone-area rectangles
        if hasattr(self, 'drone_patches'):
            for drone in (self.sim.friendly_drone, self.sim.enemy_drone):
                rect = self.drone_patches[drone]
                xmin, xmax, ymin, ymax = drone.areas[drone.current_area]
                rect.set_bounds(
                    xmin*CELL_SIZE,
                    ymin*CELL_SIZE,
                    (xmax-xmin)*CELL_SIZE,
                    (ymax-ymin)*CELL_SIZE
                )

        # redraw dynamic artists
        if self.visualize:
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

            # collect all the artists we want to blit
            artists = (
                [self.ax.title] +
                list(self.enemy_markers.values()) +
                list(self.enemy_arrows.values()) +
                list(self.enemy_texts.values()) +
                list(self.enemy_name_texts.values()) +
                list(self.enemy_attack_indicators.values()) +
                list(self.friendly_markers.values()) +
                list(self.friendly_arrows.values()) +
                list(self.friendly_texts.values()) +
                list(self.friendly_name_texts.values()) +
                list(self.friendly_attack_indicators.values())
            )

            for art in artists:
                self.ax.draw_artist(art)
        else:
            # In non-visual mode, just draw the figure
            self.fig.canvas.draw()

        # update the task/plan text box with better formatting
        if hasattr(self, 'task_text'):
            lines = ["CURRENT TASKS", "=" * 20]
            
            # Friendly units first
            friendly_units = [u for u in self.sim.friendly_units]
            if friendly_units:
                lines.append("\nFRIENDLY UNITS:")
                for u in friendly_units:
                    if u.current_plan:
                        # Truncate long plans to show only first task
                        if len(u.current_plan) > 1:
                            first_task = u.current_plan[0]
                            if isinstance(first_task, tuple):
                                op, tgt = first_task
                                lines.append(f"  {u.name:<15}: {op} {tgt}")
                            else:
                                lines.append(f"  {u.name:<15}: {first_task}")
                            # Add indicator that plan is truncated
                            lines.append(f"  {'':<15}  ... (+{len(u.current_plan)-1} more)")
                        else:
                            # Single task
                            if isinstance(u.current_plan[0], tuple):
                                op, tgt = u.current_plan[0]
                                lines.append(f"  {u.name:<15}: {op} {tgt}")
                            else:
                                lines.append(f"  {u.name:<15}: {u.current_plan[0]}")
                    else:
                        lines.append(f"  {u.name:<15}: Idle")
            
            # Enemy units
            enemy_units = [u for u in self.sim.enemy_units]
            if enemy_units:
                lines.append("\nENEMY UNITS:")
                for u in enemy_units:
                    if u.current_plan:
                        # Truncate long plans to show only first task
                        if len(u.current_plan) > 1:
                            first_task = u.current_plan[0]
                            if isinstance(first_task, tuple):
                                op, tgt = first_task
                                lines.append(f"  {u.name:<15}: {op} {tgt}")
                            else:
                                lines.append(f"  {u.name:<15}: {first_task}")
                            # Add indicator that plan is truncated
                            lines.append(f"  {'':<15}  ... (+{len(u.current_plan)-1} more)")
                        else:
                            # Single task
                            if isinstance(u.current_plan[0], tuple):
                                op, tgt = u.current_plan[0]
                                lines.append(f"  {u.name:<15}: {op} {tgt}")
                            else:
                                lines.append(f"  {u.name:<15}: {u.current_plan[0]}")
                    else:
                        lines.append(f"  {u.name:<15}: Idle")
            
            self.task_text.set_text("\n".join(lines))

        # update the spotted units box with better formatting
        if hasattr(self, 'spotted_text'):
            friendly_spotted = list(self.sim.friendly_drone.last_known.keys())
            enemy_spotted = list(self.sim.enemy_drone.last_known.keys())
            
            lines = ["DRONE SURVEILLANCE", "=" * 18]
            
            lines.append("\nFriendly Drone:")
            if friendly_spotted:
                lines.append(f"  Spotted: {len(friendly_spotted)} units")
                for n in friendly_spotted:
                    lines.append(f"    • {n}")
            else:
                lines.append("  No units spotted")
            
            lines.append("\nEnemy Drone:")
            if enemy_spotted:
                lines.append(f"  Spotted: {len(enemy_spotted)} units")
                for n in enemy_spotted:
                    lines.append(f"    • {n}")
            else:
                lines.append("  No units spotted")
            
            self.spotted_text.set_text("\n".join(lines))

        # LOS box removed as requested

        # --- DEBUG overlay for staging candidates and paths ---

        # Clear candidate markers
        if hasattr(self, "candidate_scatters"):
            for s in self.candidate_scatters:
                s.remove()
        self.candidate_scatters = []

        if hasattr(self.sim, "debug_candidates"):
            for pos in self.sim.debug_candidates:
                x, y = pos
                cx = x * CELL_SIZE + CELL_SIZE / 2
                cy = y * CELL_SIZE + CELL_SIZE / 2
                sc = self.ax.scatter(cx, cy, marker=".", color="magenta", s=30, zorder=3, alpha=0.5)
                self.candidate_scatters.append(sc)

        # Clear previous debug paths
        if hasattr(self, "debug_path_lines"):
            for l in self.debug_path_lines:
                l.remove()
        self.debug_path_lines = []

        if hasattr(self.sim, "debug_paths"):
            for debug_entry in self.sim.debug_paths:
                # Support both dict-style (with unit) and legacy list-style
                if isinstance(debug_entry, dict):
                    paths = []
                    unit = debug_entry.get("unit")

                    # collect all possible path keys
                    if debug_entry.get("path"):
                        paths.append(debug_entry["path"])
                    if debug_entry.get("path_to_staging"):
                        paths.append(debug_entry["path_to_staging"])
                    if debug_entry.get("path_to_flank"):
                        paths.append(debug_entry["path_to_flank"])
                    if debug_entry.get("path_to_enemy"):
                        paths.append(debug_entry["path_to_enemy"])

                    # determine color based on unit
                    color = "cyan"
                    if unit is not None:
                        if unit in self.sim.friendly_units:
                            style = STYLE_MAP_FRIENDLY.get(unit.state.get("type"), STYLE_DEFAULT_FRIENDLY)
                        else:
                            style = STYLE_MAP_ENEMY.get(unit.state.get("type"), STYLE_DEFAULT_ENEMY)
                        color = style["color"]

                    for path in paths:
                        if not path or len(path) < 2:
                            continue
                        xs = [pt[0] * CELL_SIZE + CELL_SIZE / 2 for pt in path]
                        ys = [pt[1] * CELL_SIZE + CELL_SIZE / 2 for pt in path]
                        line, = self.ax.plot(xs, ys, linestyle="--", color=color, alpha=0.6, zorder=2)
                        self.debug_path_lines.append(line)

                # In case older code pushes plain lists directly
                elif isinstance(debug_entry, list):
                    if len(debug_entry) < 2:
                        continue
                    xs = [pt[0] * CELL_SIZE + CELL_SIZE / 2 for pt in debug_entry]
                    ys = [pt[1] * CELL_SIZE + CELL_SIZE / 2 for pt in debug_entry]
                    line, = self.ax.plot(xs, ys, linestyle="--", color="cyan", alpha=0.6, zorder=2)
                    self.debug_path_lines.append(line)


        # blit and flush (only in visual mode)
        if self.visualize:
            self.fig.canvas.blit(self.ax.bbox)
            self.fig.canvas.flush_events()
            # plt.pause(0.001)

    def _remove_unit_visuals(self, name, group='enemy'):
        marker_dict      = self.enemy_markers      if group == 'enemy' else self.friendly_markers
        arrow_dict       = self.enemy_arrows       if group == 'enemy' else self.friendly_arrows
        text_dict        = self.enemy_texts        if group == 'enemy' else self.friendly_texts
        name_text_dict   = self.enemy_name_texts   if group == 'enemy' else self.friendly_name_texts
        ring_dict        = self.enemy_attack_indicators if group == 'enemy' else self.friendly_attack_indicators

        for d in [marker_dict, arrow_dict, text_dict, name_text_dict, ring_dict,
                 self.staging_markers, self.defend_markers]:
            art = d.pop(name, None)
            if art:
                art.remove()
