"""Utility to replay an HPO trial and plot the per-generation metrics."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "matplotlib is required for plotting. Install it with 'pip install matplotlib'"
    ) from exc

from ga_optimizer import genetic_optimize


_STAGE_KEY_MAP = {
    "low": "low_trials",
    "confirm": "confirms",
}


def _normalize_path(value: str) -> str:
    """Convert serialized Windows-style paths to the current platform."""
    if not value:
        return value
    return os.fspath(Path(value))


def _resolve_workers(option: str, recorded_value: Any) -> Any:
    """Determine the worker count to use based on the CLI option."""
    normalized = option.strip().lower()
    if normalized == "original":
        return recorded_value
    if normalized == "auto":
        return os.cpu_count() or recorded_value or 1

    try:
        workers = int(option)
    except ValueError as exc:  # pragma: no cover - argument validation
        raise SystemExit(
            "--workers must be 'original', 'auto', or a positive integer"
        ) from exc

    if workers < 1:  # pragma: no cover - argument validation
        raise SystemExit("--workers must be >= 1 when provided as an integer")

    return workers


def _prepare_run_cfg(
    cfg: Dict[str, Any],
    seed: int,
    output_dir: Path,
    tag: str,
    keep_checkpoints: bool,
    workers: Any,
) -> Dict[str, Any]:
    run_cfg = dict(cfg)
    run_cfg.pop("seed", None)

    # Replace serialized Windows paths with native ones.
    for key in ("checkpoint_path", "best_path", "resume_from", "worker_log_file"):
        if key in run_cfg and isinstance(run_cfg[key], str):
            run_cfg[key] = _normalize_path(run_cfg[key])

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"{tag}_checkpoint.pkl"
    best_path = output_dir / f"{tag}_best.pkl"
    worker_log_file = output_dir / f"{tag}_workers.log"

    run_cfg["checkpoint_path"] = os.fspath(checkpoint_path)
    run_cfg["best_path"] = os.fspath(best_path)
    run_cfg["worker_log_file"] = os.fspath(worker_log_file)
    run_cfg["rng"] = random.Random(seed)

    if workers is not None:
        run_cfg["workers"] = workers

    if not keep_checkpoints:
        run_cfg["checkpoint_every"] = 0
        run_cfg["resume_from"] = None
    else:
        # Ensure resume path matches the newly created checkpoint location.
        run_cfg["resume_from"] = run_cfg.get("resume_from") or os.fspath(checkpoint_path)

    return run_cfg


def _load_trial(results_path: Path, stage: str, index: int) -> Dict[str, Any]:
    with results_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    key = _STAGE_KEY_MAP.get(stage)
    if key is None:
        raise ValueError(f"Unknown stage '{stage}'. Valid options: {sorted(_STAGE_KEY_MAP)}")

    trials: List[Dict[str, Any]] = data.get(key, [])
    if not trials:
        raise ValueError(f"No trials recorded for stage '{stage}' in {results_path}")

    try:
        trial = trials[index]
    except IndexError as exc:
        raise IndexError(
            f"Stage '{stage}' only has {len(trials)} trial(s); requested index {index}"
        ) from exc

    return trial


def _extract_series(history: Iterable[Dict[str, Any]], metric: str) -> List[float]:
    series = []
    for row in history:
        if metric not in row:
            available = ", ".join(sorted(row))
            raise KeyError(f"Metric '{metric}' not present in history row; available keys: {available}")
        series.append(float(row[metric]))
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="hpo_minimal_results.json", type=Path,
                        help="Path to the saved HPO results JSON file.")
    parser.add_argument("--stage", choices=sorted(_STAGE_KEY_MAP), default="low",
                        help="Which fidelity stage to replay (low or confirm).")
    parser.add_argument("--trial", type=int, default=0,
                        help="Zero-based index of the trial within the chosen stage.")
    parser.add_argument("--metric", default="best_scalar_fitness",
                        help="History metric to plot against generation.")
    parser.add_argument("--output-dir", type=Path, default=Path("rerun_outputs"),
                        help="Directory where history JSON and plots will be written.")
    parser.add_argument("--plot-path", type=Path, default=None,
                        help="Optional explicit path for the generated plot image.")
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Keep GA checkpoints enabled instead of disabling them for the replay.")
    parser.add_argument("--workers", default="original",
                        help="Worker pool size: 'original', 'auto', or a positive integer.")

    args = parser.parse_args()

    trial = _load_trial(args.results, args.stage, args.trial)
    cfg_used = trial["cfg_used"]
    seed = int(cfg_used["seed"])

    tag = f"{args.stage}_trial{args.trial}"
    workers = _resolve_workers(args.workers, cfg_used.get("workers"))
    run_cfg = _prepare_run_cfg(
        cfg_used,
        seed,
        args.output_dir,
        tag,
        args.keep_checkpoints,
        workers,
    )

    effective_workers = run_cfg.get("workers", "auto")
    print(
        f"Replaying {args.stage} trial #{args.trial} with seed {seed} using "
        f"{effective_workers} worker(s)..."
    )
    result = genetic_optimize(**run_cfg)
    history = result.get("history", [])

    if not history:
        raise RuntimeError("genetic_optimize returned no history data; cannot plot.")

    history_path = args.output_dir / f"{tag}_history.json"
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"Saved history to {history_path}")

    gens = [row.get("gen", idx) for idx, row in enumerate(history)]
    series = _extract_series(history, args.metric)

    plot_path = args.plot_path or args.output_dir / f"{tag}_{args.metric}.png"

    plt.figure(figsize=(8, 5))
    plt.plot(gens, series, marker="o")
    plt.xlabel("Generation")
    plt.ylabel(args.metric.replace("_", " ").title())
    plt.title(f"{args.stage.title()} trial #{args.trial}: {args.metric}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
