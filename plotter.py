import time
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from matplotlib import ticker

# Cell and grid constants (adjust or import as needed)
GRID_WIDTH = 75
GRID_HEIGHT = 50
CELL_SIZE = 100

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
        # initialize Matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        if not visualize:
            return
        plt.ion()
        plt.show(block=False)

        # Prepare artist lists
        self.enemy_markers = []
        self.enemy_arrows = []
        self.enemy_texts = []
        self.enemy_name_texts = []
        self.enemy_attack_indicators = []

        self.friendly_markers = []
        self.friendly_arrows = []
        self.friendly_texts = []
        self.friendly_name_texts = []
        self.friendly_attack_indicators = []

        # Build the plot
        self._init_plot()
        # Capture background for blitting
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        self.paused = False
        # connect key-press event
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)


    def _init_plot(self):
        # --- build terrain RGB array ---
        # start white
        grid = np.ones((GRID_HEIGHT, GRID_WIDTH, 3), dtype=float)

        # water = blue
        for (x,y) in self.sim.river:
            grid[y, x] = [0.0, 0.0, 1.0]

        # deep forest = dark green
        for (x,y) in self.sim.forest:
            grid[y, x] = [0.0, 0.5, 0.0]

        # forest edge = light green
        for (x,y) in self.sim.forest_edge:
            grid[y, x] = [0.7, 1.0, 0.7]

        # cliffs = brown
        for (x,y) in self.sim.cliffs:
            grid[y, x] = [0.6, 0.3, 0.0]

        # now blit it
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

        # and your climb‐entries & outpost on top of it...
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

        if getattr(self.sim, 'outpost_position', None):
            ox, oy = self.sim.outpost_position
            self.ax.plot(
                ox*CELL_SIZE + CELL_SIZE/2,
                oy*CELL_SIZE + CELL_SIZE/2,
                marker='*', markersize=10,
                color='magenta', label='Outpost', zorder=2
            )


        # Configure axes limits, grid, and labels
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-CELL_SIZE/2, GRID_WIDTH * CELL_SIZE - CELL_SIZE/2)
        self.ax.set_ylim(-CELL_SIZE/2, GRID_HEIGHT * CELL_SIZE - CELL_SIZE/2)
        major_step = 500
        self.ax.set_xticks(np.arange(0, (GRID_WIDTH+1)*CELL_SIZE, major_step))
        self.ax.set_yticks(np.arange(0, (GRID_HEIGHT+1)*CELL_SIZE, major_step))
        self.ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f} m"))
        self.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f} m"))
        self.ax.grid(True)

        # Enemy unit artists
        seen_types = set()
        for enemy in self.sim.enemy_units:
            typ = enemy.state.get('type')
            style = STYLE_MAP_ENEMY.get(typ, STYLE_DEFAULT_ENEMY)
            label = style['label'] if typ not in seen_types else None
            seen_types.add(typ)

            # Marker
            m = self.ax.scatter([], [], marker=style['marker'],
                                s=100, color=style['color'], label=label, zorder=5)
            self.enemy_markers.append(m)
            # Arrow
            a = self.ax.quiver([], [], [], [], color=style['color'],
                                edgecolor='black', linewidth=0.5, width=0.008,
                                scale=1, scale_units='xy', angles='xy', zorder=4)
            self.enemy_arrows.append(a)
            # Health text
            t = self.ax.text(0, 0, '', fontsize=6, color='black', zorder=6)
            self.enemy_texts.append(t)
            # Name text
            nt = self.ax.text(0, 0, enemy.name, ha='center', va='top',
                              fontsize='small', color='black', zorder=6)
            self.enemy_name_texts.append(nt)
            # Attack indicator ring
            ring = Circle((0,0), radius=CELL_SIZE*0.6, fill=False,
                          linewidth=2, edgecolor='red', visible=False, zorder=7)
            self.ax.add_patch(ring)
            self.enemy_attack_indicators.append(ring)

        # Friendly unit artists
        seen_types = set()
        for unit in self.sim.friendly_units:
            typ = unit.state.get('type')
            style = STYLE_MAP_FRIENDLY.get(typ, STYLE_DEFAULT_FRIENDLY)
            label = style['label'] if typ not in seen_types else None
            seen_types.add(typ)

            m = self.ax.scatter([], [], marker=style['marker'],
                                s=100, color=style['color'], label=label, zorder=5)
            self.friendly_markers.append(m)
            a = self.ax.quiver([], [], [], [], color=style['color'],
                                edgecolor='black', linewidth=0.5, width=0.008,
                                scale=1, scale_units='xy', angles='xy', zorder=4)
            self.friendly_arrows.append(a)
            t = self.ax.text(0, 0, '', fontsize=6, color='black', zorder=6)
            self.friendly_texts.append(t)
            nt = self.ax.text(0, 0, unit.name, ha='center', va='top',
                              fontsize='small', color='black', zorder=6)
            self.friendly_name_texts.append(nt)
            ring = Circle((0,0), radius=CELL_SIZE*0.6, fill=False,
                          linewidth=2, edgecolor='orange', visible=False, zorder=7)
            self.ax.add_patch(ring)
            self.friendly_attack_indicators.append(ring)

        # Legend
            # self.ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
            # self.fig.subplots_adjust(right=0.75)  # make room for the box
            # self.task_text = self.ax.text(
            #     1.02, 0.5, "", transform=self.ax.transAxes,
            #     ha="left", va="top", fontsize="small", family="monospace",
            #     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
            # )
        
        self.fig.subplots_adjust(right=0.7)

        self.task_text = self.ax.text(
            1.02,    # x in axes‐fraction coords
            1,     # y in axes‐fraction coords (middle of the spare right margin)
            "",      # initially empty
            transform=self.ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
        )

        # box showing what each drone has spotted
        self.spotted_text = self.ax.text(
            1.02, 0.6, "",  # down at 20% of the axis height
            transform=self.ax.transAxes,
            ha="left", va="top",
            fontsize="small", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3",
                    facecolor="white", edgecolor="black", alpha=0.8)
        )

        # box showing which units each side currently sees by LOS
        self.los_text = self.ax.text(
            1.02, 0,  # 5% up from the bottom
            "",
            transform=self.ax.transAxes,
            ha="left", va="bottom",
            fontsize="small", family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white", edgecolor="black", alpha=0.8
            )
        )

    def _on_key(self, event):
        # toggle pause when user presses 'p'
        if event.key == 'p':
            self.paused = not self.paused
            state = "PAUSED" if self.paused else "RUNNING"
            print(f"[Visualization {state}]")

    def update(self):
        if not self.visualize:
            return
        
        if self.paused:
            return
        
        # Check for any state changes
        needs = any(u.needs_update() for u in self.sim.enemy_units + self.sim.friendly_units)
        if not needs:
            return
        # Restore background
        self.fig.canvas.restore_region(self.background)
        # Update title
        self.ax.set_title(f"Step {self.sim.step_count} - {self.sim.plan_name}")

        def _update_group(units, markers, arrows, texts, names, rings):
            for i, u in enumerate(units):
                alive = u.state.get("enemy_alive", u.state.get("health", 0) > 0)
                if not alive:
                    # hide everything for that slot
                    markers[i].set_visible(False)
                    arrows[i].set_visible(False)
                    texts[i].set_visible(False)
                    names[i].set_visible(False)
                    rings[i].set_visible(False)
                    continue

                # ensure it’s visible if it was hidden last frame
                markers[i].set_visible(True)
                arrows[i].set_visible(True)
                texts[i].set_visible(True)
                names[i].set_visible(True)

                x, y = u.state['position']
                cx = x * CELL_SIZE + CELL_SIZE/2
                cy = y * CELL_SIZE + CELL_SIZE/2
                markers[i].set_offsets((cx, cy))
                texts[i].set_position((cx + 15, cy + 15))
                texts[i].set_text(f"{u.state['current_group_size']}/{u.state['health']:.0f}")
                names[i].set_position((cx, cy - CELL_SIZE*0.3))
                fx, fy = u.state.get('facing', (0,0))
                norm = (fx**2 + fy**2)**0.5
                if norm > 0:
                    fx, fy = fx/norm, fy/norm
                arrows[i].set_UVC(fx*CELL_SIZE*1.2, fy*CELL_SIZE*1.2)
                arrows[i].set_offsets((cx, cy))
                ring = rings[i]
                if u.state.get('is_attacking', False):
                    ring.center = (cx, cy)
                    ring.set_visible(True)
                else:
                    ring.set_visible(False)

        _update_group(self.sim.enemy_units,
                      self.enemy_markers, self.enemy_arrows,
                      self.enemy_texts, self.enemy_name_texts,
                      self.enemy_attack_indicators)
        _update_group(self.sim.friendly_units,
                      self.friendly_markers, self.friendly_arrows,
                      self.friendly_texts, self.friendly_name_texts,
                      self.friendly_attack_indicators)

        # Redraw & recapture
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        # Blit artists
        artists = [self.ax.title] + \
                  self.enemy_markers + self.enemy_arrows + \
                  self.enemy_texts + self.enemy_name_texts + \
                  self.enemy_attack_indicators + \
                  self.friendly_markers + self.friendly_arrows + \
                  self.friendly_texts + self.friendly_name_texts + \
                  self.friendly_attack_indicators
        for art in artists:
            self.ax.draw_artist(art)

        # build lines like "FriendlyTankGroup: AttackEnemy EnemyInfantryGroup1"
        lines = []
        for u in self.sim.friendly_units + self.sim.enemy_units:
            if u.current_plan and isinstance(u.current_plan[0], tuple):
                op, target = u.current_plan[0]
                lines.append(f"{u.name:<20}: {op} {target}")
            else:
                lines.append(f"{u.name:<20}: {u.current_plan or 'Idle'}")

        # join them into one text block
        self.task_text.set_text("\n".join(lines))

        # Build spotted lists from each drone
        friendly_spotted = list(self.sim.friendly_drone.last_known.keys())
        enemy_spotted    = list(self.sim.enemy_drone.last_known.keys())

        # Format into a multi‐line block
        lines = [
        "Friendly drone spotted:",
        *(["  "+n for n in friendly_spotted] if friendly_spotted else ["  (none)"]),
        "",
        "Enemy   drone spotted:",
        *(["  "+n for n in enemy_spotted]    if enemy_spotted    else ["  (none)"]),
        ]
        self.spotted_text.set_text("\n".join(lines))
        
        # --- per-unit LOS listings ---
        lines = ["LOS ⟶ Friendly sees per unit:"]
        for u in self.sim.friendly_units:
            seen = u.state.get("visible_enemies", [])
            if seen:
                lines.append(f"  {u.name}: {', '.join(sorted(seen))}")
            else:
                lines.append(f"  {u.name}: (none)")

        lines.append("")  # blank separator
        lines.append("LOS ⟶ Enemy   sees per unit:")
        for e in self.sim.enemy_units:
            seen = e.state.get("visible_enemies", [])
            if seen:
                lines.append(f"  {e.name}: {', '.join(sorted(seen))}")
            else:
                lines.append(f"  {e.name}: (none)")

        self.los_text.set_text("\n".join(lines))

        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()
        plt.pause(0.001)
