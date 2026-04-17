from __future__ import annotations

import random
import time

from .bay import apply_action, legal_actions
from .heuristic import rollout_heuristic
from .models import Action, BayInstance, BayState, QTable, TrainingArtifact, TrainingConfig


def train(instance: BayInstance, config: TrainingConfig | None = None) -> TrainingArtifact:
    training_config = config or TrainingConfig()
    rng = random.Random(training_config.seed)
    q_table: QTable = {}
    heuristic_cache: dict[tuple[int, ...], int] = {}

    initial_state = instance.initial_state()
    start_time = time.monotonic()
    episodes_run = 0
    best_episode_reward = float("-inf")

    for _ in range(training_config.max_episodes):
        if training_config.time_limit_seconds is not None:
            elapsed = time.monotonic() - start_time
            if elapsed >= training_config.time_limit_seconds:
                break

        state = initial_state
        total_reward = 0.0

        # Each episode simulates a full retrieval process from the original bay.
        while not state.is_terminal():
            actions = legal_actions(state)
            _initialize_actions(state, actions, q_table, heuristic_cache)
            action = _select_behavior_action(state, actions, q_table, training_config.epsilon, rng)
            next_state, reward = apply_action(state, action)

            if next_state.is_terminal():
                next_best = 0.0
            else:
                next_actions = legal_actions(next_state)
                _initialize_actions(next_state, next_actions, q_table, heuristic_cache)
                next_key = next_state.flatten()
                next_best = max(q_table[next_key][candidate] for candidate in next_actions)

            state_key = state.flatten()
            current_value = q_table[state_key][action]
            # Standard one-step Q-learning update using the best legal continuation.
            temporal_difference = reward + training_config.gamma * next_best - current_value
            q_table[state_key][action] = current_value + training_config.alpha * temporal_difference

            total_reward += reward
            state = next_state

        episodes_run += 1
        best_episode_reward = max(best_episode_reward, total_reward)

    return TrainingArtifact(
        q_table=q_table,
        config=training_config,
        episodes_run=episodes_run,
        best_episode_reward=best_episode_reward,
    )


def _initialize_actions(
    state: BayState,
    actions: list[Action],
    q_table: QTable,
    heuristic_cache: dict[tuple[int, ...], int],
) -> None:
    state_key = state.flatten()
    state_actions = q_table.setdefault(state_key, {})

    for action in actions:
        if action in state_actions:
            continue
        next_state, reward = apply_action(state, action)
        if next_state.is_terminal():
            optimistic_value = float(reward)
        else:
            next_key = next_state.flatten()
            remaining_relocations = heuristic_cache.get(next_key)
            if remaining_relocations is None:
                remaining_relocations, _ = rollout_heuristic(next_state)
                heuristic_cache[next_key] = remaining_relocations
            # The paper initializes unseen values with a heuristic estimate instead
            # of a uniform constant. Here we use immediate reward plus the negative
            # heuristic rollout cost from the successor state.
            optimistic_value = float(reward - remaining_relocations)
        state_actions[action] = optimistic_value


def _select_behavior_action(
    state: BayState,
    actions: list[Action],
    q_table: QTable,
    epsilon: float,
    rng: random.Random,
) -> Action:
    if rng.random() < epsilon:
        return rng.choice(actions)
    return _best_known_action(state, actions, q_table, rng)


def _best_known_action(
    state: BayState,
    actions: list[Action],
    q_table: QTable,
    rng: random.Random,
) -> Action:
    state_actions = q_table[state.flatten()]
    best_value = max(state_actions[action] for action in actions)
    best_actions = [action for action in actions if state_actions[action] == best_value]
    return rng.choice(best_actions)
