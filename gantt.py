import itertools
import textwrap
import re
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
from pathlib import Path

def _extract_target(label: str) -> str | None:
    """Return text inside the outermost parentheses of `label`, or None."""
    start = None
    depth = 0
    for i, ch in enumerate(label):
        if ch == '(':
            if start is None:
                start = i + 1
            depth += 1
        elif ch == ')' and depth:
            depth -= 1
            if depth == 0 and start is not None:
                inner = label[start:i].strip()
                return inner or None
    return None

def _acronym(s: str) -> str:
    """Ultra-short code: take first letter of alpha groups + keep trailing digits."""
    parts = re.findall(r'[A-Za-z]+|\d+', s.replace('_', ' '))
    letters = ''.join(p[0].upper() for p in parts if p.isalpha())
    digits  = ''.join(p for p in parts if p.isdigit())
    return (letters + digits)[:6] or s[:3].upper()

def _build_target_codes(all_targets, mode: str):
    """
    mode: 'code' -> numeric codes '1','2',...
          'acronym' -> ETG1, EIG1, etc. (ensures uniqueness with suffixes)
    """
    if mode == 'code':
        return {t: str(i+1) for i, t in enumerate(sorted(set(all_targets)))}

    base = {t: _acronym(t) for t in set(all_targets)}
    used = {}
    codes = {}
    for t in sorted(base.keys()):
        c = base[t]
        if c not in used:
            used[c] = 1
            codes[t] = c
        else:
            used[c] += 1
            codes[t] = f"{c}{used[c]}"
    return codes

def generate_gantt_chart(
    units, filename="gantt_chart.png",
    figsize=(20, 12), dpi=200, bar_height=0.85,
    fontsize=14, min_width_for_label=6,
    label_mode='code',
    include_target_key=True,
    save_target_key_txt=True
):
    """
    Bars = task colors (legend shows tasks).
    Bar text = tiny label per target (numeric code by default).
    A key on the bottom maps labels back to full targets.
    """
    colors = itertools.cycle(plt.cm.tab20.colors)
    task_colors = {}

    all_targets = []
    for u in units:
        for task_label, *_ in u.task_log:
            t = _extract_target(task_label)
            if t:
                all_targets.append(t)
    target_codes = _build_target_codes(all_targets, label_mode) if label_mode != 'none' else {}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for unit in units:
        for task_label, start, end in unit.task_log:
            width = max(0, end - start)
            base_task = task_label.split("(")[0]
            if base_task not in task_colors:
                task_colors[base_task] = next(colors)

            ax.barh(
                unit.name, width, left=start, height=bar_height,
                color=task_colors[base_task], edgecolor="black",
                linewidth=1.2, alpha=0.9,
            )

            if label_mode != 'none' and width >= min_width_for_label:
                tgt = _extract_target(task_label)
                if tgt and tgt in target_codes:
                    tiny = target_codes[tgt]
                    ax.text(
                        start + width/2, unit.name, tiny,
                        ha="center", va="center", fontsize=fontsize+2, weight="bold",
                        color="black",
                        path_effects=[pe.withStroke(linewidth=4, foreground="white")],
                        clip_on=True,
                    )

    # Legend at the bottom
    labels_sorted = sorted(task_colors.keys())
    handles = [plt.Rectangle((0,0), 1,1, color=task_colors[l]) for l in labels_sorted]
    ax.legend(handles, labels_sorted, title="Tasks",
              loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=len(labels_sorted), frameon=True,
              fontsize=fontsize, title_fontsize=fontsize+2)

    # Axis labels and title
    ax.set_xlabel("Simulation Step", fontsize=fontsize+2, fontweight='bold')
    ax.set_ylabel("Unit", fontsize=fontsize+2, fontweight='bold')
    ax.set_title("Gantt Chart of Best Simulation Run", fontsize=fontsize+6, fontweight='bold', pad=20)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.grid(True, axis="x", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.grid(False, axis="y")

    plt.tight_layout(rect=[0, 0.12, 1, 1])  # leave space at bottom

    if include_target_key and target_codes:
        key_text = ["Target Key:"]
        items = sorted(target_codes.items(), key=lambda kv: int(kv[1]) if kv[1].isdigit() else kv[1])
        for full, code in items:
            short_full = textwrap.shorten(full, width=40, placeholder="…")
            key_text.append(f"{code}: {short_full}")

        # Place key text as multiline at bottom center
        fig.text(0.5, 0.02, "\n".join(key_text), ha="center", va="bottom",
                 fontsize=fontsize, family="monospace")

        if save_target_key_txt:
            p = Path(filename)
            keyfile = p.with_suffix("").as_posix() + "_target_key.txt"
            with open(keyfile, "w", encoding="utf-8") as f:
                for full, code in items:
                    f.write(f"{code}\t{full}\n")

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.03)

    plt.savefig(filename)
    plt.close(fig)
