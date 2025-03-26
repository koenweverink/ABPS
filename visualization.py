# visualization.py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from copy import deepcopy
from terrain import terrain_map
from tasks import move, scout_area, attack, secure_outpost, update_detection

def visualize_plan(original_state, plan):
    state = deepcopy(original_state)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Military Operation Simulation")
    ax.set_xticks(range(len(terrain_map[0])))
    ax.set_yticks(range(len(terrain_map)))
    ax.set_xticklabels(range(len(terrain_map[0])))
    ax.set_yticklabels(range(len(terrain_map)))
    ax.grid(True)

    terrain_img = [[0 if cell == 0 else 0.5 for cell in row] for row in terrain_map]
    ax.imshow(terrain_img, cmap="Greys", alpha=0.5)

    scouts_plot, = ax.plot([], [], 'bo', label="Scouts", markersize=12, alpha=0.8)
    infantry_plot, = ax.plot([], [], 'ro', label="Infantry", markersize=12, alpha=0.8)
    tanks_plot, = ax.plot([], [], 'yo', label="Tanks", markersize=12, alpha=0.8)
    enemy_plot, = ax.plot([], [], 'k*', label="Enemy", markersize=15)

    ax.legend(loc="upper right")
    strength_text = ax.text(0.5, 1.05, f"Enemy Health: {state.data['enemy'].health}",
                            transform=ax.transAxes, ha="center", fontsize=10)

    detection_texts = {
        "scouts": ax.text(0, 0, "", color="magenta", fontsize=8, visible=False),
        "infantry": ax.text(0, 0, "", color="magenta", fontsize=8, visible=False),
        "tanks": ax.text(0, 0, "", color="magenta", fontsize=8, visible=False),
        "enemy": ax.text(0, 0, "", color="red", fontsize=8, visible=False)
    }

    # Health bars
    health_bars = {
        "scouts": ax.add_patch(Rectangle((0, 0), 0, 0, color="blue", alpha=0.5)),
        "infantry": ax.add_patch(Rectangle((0, 0), 0, 0, color="red", alpha=0.5)),
        "tanks": ax.add_patch(Rectangle((0, 0), 0, 0, color="yellow", alpha=0.5)),
        "enemy": ax.add_patch(Rectangle((0, 0), 0, 0, color="black", alpha=0.5))
    }

    def init():
        s = state.data["agents"]["scouts"]
        i = state.data["agents"]["infantry"]
        t = state.data["agents"]["tanks"]
        e = state.data["enemy"]
        scouts_plot.set_data([s.y], [s.x])
        infantry_plot.set_data([i.y], [i.x])
        tanks_plot.set_data([t.y], [t.x])
        enemy_plot.set_data([e.y], [e.x])
        strength_text.set_text(f"Enemy Health: {state.data['enemy'].health}")
        for name, txt in detection_texts.items():
            txt.set_visible(False)
        for name, bar in health_bars.items():
            bar.set_visible(False)
        return (scouts_plot, infantry_plot, tanks_plot, enemy_plot, strength_text,
                *detection_texts.values(), *health_bars.values())

    def update(frame):
        print(f"Processing frame {frame}/{len(plan)}")
        if frame < len(plan):
            task = plan[frame]
            print(f"Executing: {task}")
            if task[0] == "move":
                move(state, task[1], task[2], task[3])
            elif task[0] == "scout_area":
                scout_area(state, task[1])
            elif task[0] == "attack":
                attack(state, task[1])
            elif task[0] == "secure_outpost":
                secure_outpost(state)

        update_detection(state)

        s = state.data["agents"]["scouts"]
        i = state.data["agents"]["infantry"]
        t = state.data["agents"]["tanks"]
        e = state.data["enemy"]

        # Update positions
        if s.health > 0:
            scouts_plot.set_data([s.y], [s.x])
        else:
            scouts_plot.set_data([], [])
        if i.health > 0:
            infantry_plot.set_data([i.y], [i.x])
        else:
            infantry_plot.set_data([], [])
        if t.health > 0:
            tanks_plot.set_data([t.y], [t.x])
        else:
            tanks_plot.set_data([], [])

        if e.health <= 0:
            enemy_plot.set_data([], [])
            detection_texts["enemy"].set_visible(False)
        else:
            enemy_plot.set_data([e.y], [e.x])

        # Update detection labels
        for name, txt in detection_texts.items():
            agent = state.data["agents"].get(name, state.data["enemy"] if name == "enemy" else None)
            if name == "enemy":
                detected = any(a.detects_enemy for a in state.data["agents"].values()) or \
                          any(a.detected_by_enemy for a in state.data["agents"].values())
                if e.health > 0:
                    if detected:
                        txt.set_text("DETECTED")
                        x_pos = min(max(e.y + 0.2, 0.5), len(terrain_map[0]) - 1.5)
                        y_pos = min(max(e.x - 0.2, 0.5), len(terrain_map) - 0.5)
                        txt.set_position((x_pos, y_pos))
                        txt.set_visible(True)
                    else:
                        txt.set_visible(False)
                else:
                    txt.set_visible(False)
            else:
                detected = agent.detected_by_enemy or agent.detects_enemy
                if detected and agent.health > 0:
                    txt.set_text("DETECTED")
                    x_pos = min(max(agent.y + 0.2, 0.5), len(terrain_map[0]) - 1.5)
                    y_pos = min(max(agent.x - 0.2, 0.5), len(terrain_map) - 0.5)
                    txt.set_position((x_pos, y_pos))
                    txt.set_visible(True)
                else:
                    txt.set_visible(False)

        # Update health bars
        for name, bar in health_bars.items():
            agent = state.data["agents"].get(name, state.data["enemy"] if name == "enemy" else None)
            if agent.health > 0:
                health_ratio = agent.health / agent.max_health
                bar_width = 0.8 * health_ratio  # Max width of health bar is 0.8 units
                bar_height = 0.2  # Height of health bar
                bar.set_xy((agent.y - bar_width / 2, agent.x + 0.3))  # Position above the agent
                bar.set_width(bar_width)
                bar.set_height(bar_height)
                bar.set_visible(True)
            else:
                bar.set_visible(False)

        strength_text.set_text(f"Enemy Health: {state.data['enemy'].health}")
        return (scouts_plot, infantry_plot, tanks_plot, enemy_plot, strength_text,
                *detection_texts.values(), *health_bars.values())

    ani = FuncAnimation(fig, update, frames=len(plan), init_func=init,
                        blit=True, interval=300, repeat=False)
    plt.tight_layout()
    plt.show()