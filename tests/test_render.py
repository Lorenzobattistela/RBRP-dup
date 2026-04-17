from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rbrp_dup.dataset import load_instance
from rbrp_dup.models import TrainingConfig
from rbrp_dup.qlearning import train
from rbrp_dup.render import render_solution_gif
from rbrp_dup.solver import solve


class RenderTests(unittest.TestCase):
    def test_render_solution_gif_creates_output(self) -> None:
        instance = load_instance("data/fixtures/3-3-5/blocked.txt")
        artifact = train(instance, TrainingConfig(max_episodes=5, seed=1))
        solution = solve(instance, artifact=artifact, trace=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "blocked.gif"
            render_solution_gif(instance, solution, output_path, fps=2.0)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
