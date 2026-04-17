from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import collect_instance_paths, load_instance
from .models import TrainingConfig
from .qlearning import train
from .render import render_solution_gif
from .solver import run_batch, solve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RBRP-dup proof-of-concept solver")
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser("solve", help="train and solve a single instance")
    solve_parser.add_argument("path", help="path to an instance file")
    solve_parser.add_argument("--trace", action="store_true", help="print the state trace")
    _add_training_arguments(solve_parser)

    batch_parser = subparsers.add_parser("batch", help="run a batch of instances")
    batch_parser.add_argument("path", help="path to a directory or a single instance file")
    batch_parser.add_argument("--limit", type=int, default=None, help="limit the number of instances")
    _add_training_arguments(batch_parser)

    gif_parser = subparsers.add_parser("gif", help="train, solve, and render a GIF for one instance")
    gif_parser.add_argument("path", help="path to an instance file")
    gif_parser.add_argument("--output", default=None, help="output GIF path")
    gif_parser.add_argument("--fps", type=float, default=1.25, help="animation frame rate")
    _add_training_arguments(gif_parser)

    args = parser.parse_args(argv)
    config = TrainingConfig(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        max_episodes=args.max_episodes,
        time_limit_seconds=args.time_limit_seconds,
        seed=args.seed,
    )

    if args.command == "solve":
        return _run_solve(Path(args.path), config, trace=args.trace)
    if args.command == "gif":
        return _run_gif(Path(args.path), config, output=args.output, fps=args.fps)
    return _run_batch(Path(args.path), config, limit=args.limit)


def _run_solve(path: Path, config: TrainingConfig, trace: bool) -> int:
    instance = load_instance(path)
    artifact = train(instance, config)
    solution = solve(instance, artifact=artifact, trace=trace)

    print(f"instance: {instance.instance_id}")
    print(f"episodes: {artifact.episodes_run}")
    print(f"relocations: {solution.relocations}")
    print(f"actions: {len(solution.actions)}")
    print(f"q-table actions: {solution.q_table_actions}")
    print(f"heuristic actions: {solution.heuristic_actions}")

    if trace:
        for index, step in enumerate(solution.trace, start=1):
            print(f"\nstep {index}: action={step.action} source={step.policy_source} reward={step.reward}")
            print("before:")
            print(step.before)
            print("after:")
            print(step.after)

    return 0


def _run_batch(path: Path, config: TrainingConfig, limit: int | None) -> int:
    paths = collect_instance_paths(path, limit=limit)
    if not paths:
        raise SystemExit(f"no instance files found under {path}")

    report = run_batch(path, config=config, limit=limit)
    print(f"instances: {report.count}")
    print(f"average relocations: {report.average_relocations:.3f}")
    print(f"total runtime (s): {report.total_runtime_seconds:.3f}")

    for item in report.items:
        print(
            f"{item.instance_id}: relocations={item.relocations} "
            f"actions={item.actions} episodes={item.episodes_run} "
            f"q={item.q_table_actions} heuristic={item.heuristic_actions}"
        )

    return 0


def _run_gif(path: Path, config: TrainingConfig, output: str | None, fps: float) -> int:
    instance = load_instance(path)
    artifact = train(instance, config)
    solution = solve(instance, artifact=artifact, trace=True)

    output_path = Path(output) if output else Path("out") / f"{path.stem}.gif"
    render_solution_gif(instance, solution, output_path, fps=fps)

    print(f"instance: {instance.instance_id}")
    print(f"episodes: {artifact.episodes_run}")
    print(f"relocations: {solution.relocations}")
    print(f"actions: {len(solution.actions)}")
    print(f"q-table actions: {solution.q_table_actions}")
    print(f"heuristic actions: {solution.heuristic_actions}")
    print(f"gif: {output_path}")
    return 0


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--max-episodes", type=int, default=250)
    parser.add_argument("--time-limit-seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
