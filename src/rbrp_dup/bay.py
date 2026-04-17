from __future__ import annotations

from .models import Action, BayState


def target_priority(state: BayState) -> int | None:
    priorities = [priority for stack in state.stacks for priority in stack]
    if not priorities:
        return None
    return min(priorities)


def retrieval_actions(state: BayState) -> list[Action]:
    target = target_priority(state)
    if target is None:
        return []
    return [
        (stack_index + 1, 0)
        for stack_index, stack in enumerate(state.stacks)
        if stack and stack[-1] == target
    ]


def relocation_source_stacks(state: BayState) -> list[int]:
    target = target_priority(state)
    if target is None:
        return []

    sources: list[int] = []
    for stack_index, stack in enumerate(state.stacks):
        if not stack or stack[-1] == target:
            continue
        # In the restricted variant, we can only relocate a top container if it is
        # currently blocking at least one container from the active target group.
        if target in stack:
            sources.append(stack_index)
    return sources


def relocation_actions(state: BayState) -> list[Action]:
    actions: list[Action] = []
    for source_index in relocation_source_stacks(state):
        for destination_index in range(state.stack_count):
            if destination_index == source_index or state.is_full(destination_index):
                continue
            actions.append((source_index + 1, destination_index + 1))
    return actions


def legal_actions(state: BayState) -> list[Action]:
    retrieval = retrieval_actions(state)
    if retrieval:
        # The paper's pruning rule applies here: once a target container is already
        # on top, relocations are ignored and only retrievals remain legal.
        return retrieval
    return relocation_actions(state)


def apply_action(state: BayState, action: Action) -> tuple[BayState, int]:
    source, destination = action
    if source < 1 or source > state.stack_count:
        raise ValueError(f"source stack out of bounds: {source}")
    if destination < 0 or destination > state.stack_count:
        raise ValueError(f"destination stack out of bounds: {destination}")

    source_index = source - 1
    source_stack = list(state.stacks[source_index])
    if not source_stack:
        raise ValueError("cannot move from an empty stack")

    moved_priority = source_stack.pop()
    updated_stacks = [list(stack) for stack in state.stacks]
    updated_stacks[source_index] = source_stack

    if destination == 0:
        current_target = target_priority(state)
        if moved_priority != current_target:
            raise ValueError("retrieval action must remove the current target group")
        reward = 0
    else:
        destination_index = destination - 1
        if destination_index == source_index:
            raise ValueError("relocation destination must differ from source")
        if state.is_full(destination_index):
            raise ValueError("cannot relocate into a full stack")
        updated_stacks[destination_index].append(moved_priority)
        # Relocations are the only penalized move type, so the total return is the
        # negative of the relocation count.
        reward = -1

    new_state = BayState(
        stacks=tuple(tuple(stack) for stack in updated_stacks),
        max_height=state.max_height,
    )
    return new_state, reward


def target_containers(state: BayState) -> list[tuple[int, int, int]]:
    target = target_priority(state)
    if target is None:
        return []

    candidates: list[tuple[int, int, int]] = []
    for stack_index, stack in enumerate(state.stacks):
        for tier_index, priority in enumerate(stack):
            if priority != target:
                continue
            # The heuristic ranks target-group containers by how many blockers still
            # sit above them in their current stack.
            blockers = len(stack) - tier_index - 1
            candidates.append((stack_index, tier_index, blockers))
    return candidates
