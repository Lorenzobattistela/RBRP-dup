from __future__ import annotations

from .bay import apply_action, retrieval_actions, target_containers
from .models import Action, BayState


def choose_target_container(state: BayState) -> tuple[int, int, int]:
    candidates = target_containers(state)
    if not candidates:
        raise ValueError("cannot choose a target container from an empty state")
    # Rule 1: prefer the target-group container with the fewest blockers, then use
    # deterministic tie-breaks so training and tests remain reproducible.
    return min(candidates, key=lambda item: (item[2], item[0], -item[1]))


def choose_destination_stack(state: BayState, source_index: int) -> int:
    moving_priority = state.top(source_index)
    if moving_priority is None:
        raise ValueError("cannot choose a destination for an empty source stack")

    safe_candidates: list[tuple[int, int, int]] = []
    unsafe_candidates: list[tuple[int, int, int]] = []

    for destination_index, destination_stack in enumerate(state.stacks):
        if destination_index == source_index or state.is_full(destination_index):
            continue

        if not destination_stack:
            safe_candidates.append((0, 0, destination_index))
            continue

        minimum_priority = min(destination_stack)
        maximum_priority = max(destination_stack)
        candidate = (maximum_priority, len(destination_stack), destination_index)
        # A destination is "safe" when placing the moved container there does not
        # immediately create a new blocking relation against the current ordering.
        if moving_priority <= minimum_priority:
            safe_candidates.append(candidate)
        else:
            unsafe_candidates.append(candidate)

    if safe_candidates:
        # Min-Max rule: among safe destinations, preserve flexible stacks by choosing
        # the largest current maximum priority.
        safe_candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        return safe_candidates[0][2]

    if not unsafe_candidates:
        raise ValueError("no destination stack is available")

    # If every move creates a future blocker, choose the least damaging stack.
    unsafe_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return unsafe_candidates[0][2]


def heuristic_action(state: BayState) -> Action:
    retrieval = retrieval_actions(state)
    target_stack, _, blockers = choose_target_container(state)
    if retrieval and blockers == 0:
        # When the selected target container is already exposed, the heuristic
        # follows the same retrieval-only rule as the legal action generator.
        return (target_stack + 1, 0)

    destination_stack = choose_destination_stack(state, target_stack)
    return (target_stack + 1, destination_stack + 1)


def rollout_heuristic(state: BayState) -> tuple[int, list[Action]]:
    current_state = state
    relocations = 0
    actions: list[Action] = []

    # This rollout provides the heuristic upper bound used to seed unseen Q-values.
    while not current_state.is_terminal():
        action = heuristic_action(current_state)
        current_state, reward = apply_action(current_state, action)
        relocations += -reward
        actions.append(action)

    return relocations, actions
