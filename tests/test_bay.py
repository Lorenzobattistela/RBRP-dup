from __future__ import annotations

import unittest

from rbrp_dup.bay import apply_action, legal_actions, relocation_source_stacks, retrieval_actions, target_priority
from rbrp_dup.dataset import load_instance
from rbrp_dup.heuristic import heuristic_action


class BayTests(unittest.TestCase):
    def test_optimal_rule_filters_to_retrievals(self) -> None:
        state = load_instance("data/fixtures/3-3-5/retrieval_only.txt").initial_state()
        self.assertEqual(target_priority(state), 1)
        self.assertEqual(retrieval_actions(state), [(2, 0)])
        self.assertEqual(legal_actions(state), [(2, 0)])

    def test_relocation_actions_use_all_blocked_target_stacks(self) -> None:
        state = load_instance("data/fixtures/3-3-5/blocked.txt").initial_state()
        self.assertEqual(target_priority(state), 1)
        self.assertEqual(retrieval_actions(state), [])
        self.assertEqual(relocation_source_stacks(state), [0, 1])
        self.assertEqual(
            legal_actions(state),
            [(1, 2), (1, 3), (2, 1), (2, 3)],
        )

    def test_heuristic_picks_a_deterministic_first_move(self) -> None:
        state = load_instance("data/fixtures/3-3-5/blocked.txt").initial_state()
        self.assertEqual(heuristic_action(state), (1, 3))

    def test_apply_action_updates_state_and_reward(self) -> None:
        state = load_instance("data/fixtures/3-3-5/blocked.txt").initial_state()
        next_state, reward = apply_action(state, (1, 3))
        self.assertEqual(reward, -1)
        self.assertEqual(next_state.stacks[0], (1,))
        self.assertEqual(next_state.stacks[2], (2, 3))


if __name__ == "__main__":
    unittest.main()
