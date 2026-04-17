from __future__ import annotations

import time
from pathlib import Path

from .bay import apply_action, legal_actions
from .dataset import collect_instance_paths, load_instance
from .heuristic import heuristic_action
from .models import (
    BatchItem,
    BatchReport,
    BayState,
    BayInstance,
    Solution,
    TraceStep,
    TrainingArtifact,
    TrainingConfig,
)
from .qlearning import train


def solve(
    instance: BayInstance,
    artifact: TrainingArtifact | None = None,
    trace: bool = False,
) -> Solution:
    state = instance.initial_state()
    actions: list[tuple[int, int]] = []
    trace_steps: list[TraceStep] = []
    relocations = 0
    q_table_actions = 0
    heuristic_actions = 0

    # The solve phase replays the instance greedily, using learned values when the
    # table knows the current state and falling back to the heuristic otherwise.
    while not state.is_terminal():
        action, source = _select_action(state, artifact)
        before = state.format_matrix() if trace else ""
        next_state, reward = apply_action(state, action)
        after = next_state.format_matrix() if trace else ""

        actions.append(action)
        relocations += -reward
        if source == "q_table":
            q_table_actions += 1
        else:
            heuristic_actions += 1

        if trace:
            trace_steps.append(
                TraceStep(
                    action=action,
                    reward=reward,
                    policy_source=source,
                    before=before,
                    after=after,
                )
            )

        state = next_state

    return Solution(
        instance_id=instance.instance_id,
        relocations=relocations,
        actions=actions,
        trace=trace_steps,
        q_table_actions=q_table_actions,
        heuristic_actions=heuristic_actions,
    )


def run_batch(
    path: str | Path,
    config: TrainingConfig | None = None,
    limit: int | None = None,
) -> BatchReport:
    training_config = config or TrainingConfig()
    items: list[BatchItem] = []

    for instance_path in collect_instance_paths(path, limit=limit):
        instance = load_instance(instance_path)
        start_time = time.monotonic()
        artifact = train(instance, training_config)
        solution = solve(instance, artifact=artifact, trace=False)
        runtime_seconds = time.monotonic() - start_time
        items.append(
            BatchItem(
                instance_id=instance.instance_id or str(instance_path),
                relocations=solution.relocations,
                actions=len(solution.actions),
                q_table_actions=solution.q_table_actions,
                heuristic_actions=solution.heuristic_actions,
                episodes_run=artifact.episodes_run,
                runtime_seconds=runtime_seconds,
            )
        )

    return BatchReport(items=items)


def _select_action(
    state: BayState,
    artifact: TrainingArtifact | None,
) -> tuple[tuple[int, int], str]:
    available_actions = legal_actions(state)
    if artifact is None:
        return heuristic_action(state), "heuristic"

    state_actions = artifact.q_table.get(state.flatten(), {})
    known_actions = {
        action: state_actions[action]
        for action in available_actions
        if action in state_actions
    }
    if not known_actions:
        # Sparse tabular learning means some legal states are unseen at solve time.
        return heuristic_action(state), "heuristic"

    best_value = max(known_actions.values())
    best_actions = [action for action, value in known_actions.items() if value == best_value]
    return min(best_actions), "q_table"
