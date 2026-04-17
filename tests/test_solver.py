from __future__ import annotations

import unittest

from rbrp_dup.dataset import load_instance
from rbrp_dup.models import TrainingConfig
from rbrp_dup.qlearning import train
from rbrp_dup.solver import run_batch, solve


class SolverTests(unittest.TestCase):
    def test_train_and_solve_small_fixture(self) -> None:
        instance = load_instance("data/fixtures/3-3-5/blocked.txt")
        artifact = train(instance, TrainingConfig(max_episodes=20, seed=7))
        solution = solve(instance, artifact=artifact, trace=False)

        self.assertGreater(artifact.episodes_run, 0)
        self.assertEqual(solution.relocations, 2)
        self.assertGreaterEqual(solution.q_table_actions + solution.heuristic_actions, 1)
        self.assertEqual(len(solution.actions), 7)

    def test_batch_runner_handles_fixture_directory(self) -> None:
        report = run_batch("data/fixtures", TrainingConfig(max_episodes=10, seed=3))
        self.assertEqual(report.count, 2)
        self.assertGreater(report.average_relocations, 0.0)


if __name__ == "__main__":
    unittest.main()
