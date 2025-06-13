import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib import ticker
from config import GRID_WIDTH, GRID_HEIGHT, CELL_SIZE

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

        # Initialize Matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        if not visualize:
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

        # Draw outpost if defined
        if getattr(self.sim, 'outpost_position', None):
            ox, oy = self.sim.outpost_position
            self.ax.plot(
                ox*CELL_SIZE + CELL_SIZE/2,
                oy*CELL_SIZE + CELL_SIZE/2,
                marker='*', markersize=10,
                color='magenta', label='Outpost', zorder=2
            )

        # Configure axes
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-CELL_SIZE/2, GRID_WIDTH * CELL_SIZE - CELL_SIZE/2)
        self.ax.set_ylim(-CELL_SIZE/2, GRID_HEIGHT * CELL_SIZE - CELL_SIZE/2)

        major_step = 500
        self.ax.set_xticks(np.arange(0, (GRID_WIDTH+1)*CELL_SIZE, major_step))
        self.ax.set_yticks(np.arange(0, (GRID_HEIGHT+1)*CELL_SIZE, major_step))
        self.ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f} m"))
        self.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f} m"))
        self.ax.grid(True)

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
            ring = Circle((cx, cy), radius=CELL_SIZE*0.6, fill=False,
                          linewidth=2, edgecolor='red', visible=False, zorder=7)
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
            ring = Circle((cx, cy), radius=CELL_SIZE*0.6, fill=False,
                          linewidth=2, edgecolor='orange', visible=False, zorder=7)
            self.ax.add_patch(ring)
            self.friendly_attack_indicators[unit.name] = ring

        # Make room on the right for the plan/drone/LOS text panels
        self.fig.subplots_adjust(right=0.7)

        # Plan box
        self.task_text = self.ax.text(
            1.02, 1, "", transform=self.ax.transAxes,
            ha="left", va="top", fontsize="small", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="black", alpha=0.8)
        )
        # Spotted box
        self.spotted_text = self.ax.text(
            1.02, 0.6, "", transform=self.ax.transAxes,
            ha="left", va="top", fontsize="small", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="black", alpha=0.8)
        )
        # LOS box
        self.los_text = self.ax.text(
            1.02, 0, "", transform=self.ax.transAxes,
            ha="left", va="bottom", fontsize="small", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="black", alpha=0.8)
        )

    def _on_key(self, event):
        # toggle pause when user presses 'p'
        if event.key == 'p':
            self.paused = not self.paused
            state = "PAUSED" if self.paused else "RUNNING"
            print(f"[Visualization {state}]")

    def update(self):
        if not self.visualize or self.paused:
            return

        # only redraw if someone’s state changed
        needs = any(u.needs_update()
                    for u in self.sim.enemy_units + self.sim.friendly_units)
        if not needs:
            return

        # restore static background
        self.fig.canvas.restore_region(self.background)

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
                if staging and alive:
                    sx, sy = staging
                    scx = sx * CELL_SIZE + CELL_SIZE / 2
                    scy = sy * CELL_SIZE + CELL_SIZE / 2
                    if marker:
                        marker.set_offsets((scx, scy))
                        marker.set_visible(True)
                    else:
                        m = self.ax.scatter(scx, scy, marker="X", color="black", s=80, zorder=3)
                        self.staging_markers[name] = m
                elif marker:
                    marker.set_visible(False)

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
        _update_group(self.sim.enemy_units,
                    self.enemy_markers, self.enemy_arrows,
                    self.enemy_texts, self.enemy_name_texts,
                    self.enemy_attack_indicators)

        # remove enemy visuals for units no longer in simulation
        live_enemy_names = {u.name for u in self.sim.enemy_units}
        for name in list(self.enemy_markers.keys()):
            if name not in live_enemy_names:
                self._remove_unit_visuals(name, group='enemy')

        # update friendlies
        _update_group(self.sim.friendly_units,
                    self.friendly_markers, self.friendly_arrows,
                    self.friendly_texts, self.friendly_name_texts,
                    self.friendly_attack_indicators)

        # remove friendly visuals for units no longer in simulation
        live_friendly_names = {u.name for u in self.sim.friendly_units}
        for name in list(self.friendly_markers.keys()):
            if name not in live_friendly_names:
                self._remove_unit_visuals(name, group='friendly')

        # move the persistent drone-area rectangles
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

        # update the plan-text box
        lines = []
        for u in self.sim.friendly_units + self.sim.enemy_units:
            if u.current_plan and isinstance(u.current_plan[0], tuple):
                op, tgt = u.current_plan[0]
                lines.append(f"{u.name:<20}: {op} {tgt}")
            else:
                lines.append(f"{u.name:<20}: {u.current_plan or 'Idle'}")
        self.task_text.set_text("\n".join(lines))

        # update the spotted-by-drones box
        friendly_spotted = list(self.sim.friendly_drone.last_known.keys())
        enemy_spotted    = list(self.sim.enemy_drone.last_known.keys())
        lines = ["Friendly drone spotted:"]
        if friendly_spotted:
            lines += [f"  {n}" for n in friendly_spotted]
        else:
            lines += ["  (none)"]
        lines += ["" , "Enemy   drone spotted:"]
        if enemy_spotted:
            lines += [f"  {n}" for n in enemy_spotted]
        else:
            lines += ["  (none)"]
        self.spotted_text.set_text("\n".join(lines))

        # update the LOS box
        lines = ["LOS ⟶ Friendly sees per unit:"]
        for u in self.sim.friendly_units:
            seen = u.state.get("visible_enemies", [])
            lines.append(f"  {u.name}: {', '.join(seen) if seen else '(none)'}")
        lines += ["", "LOS ⟶ Enemy   sees per unit:"]
        for e in self.sim.enemy_units:
            seen = e.state.get("visible_enemies", [])
            lines.append(f"  {e.name}: {', '.join(seen) if seen else '(none)'}")
        self.los_text.set_text("\n".join(lines))

        # blit and flush
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()
        plt.pause(0.001)

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
