from __future__ import annotations

import unittest
from pathlib import Path

from rbrp_dup.dataset import collect_instance_paths, load_instance


class DatasetTests(unittest.TestCase):
    def test_loads_fixture_and_infers_height_from_directory(self) -> None:
        instance = load_instance("data/fixtures/3-3-5/blocked.txt")
        self.assertEqual(instance.stack_count, 3)
        self.assertEqual(instance.max_height, 3)
        self.assertEqual(instance.block_count, 5)
        self.assertEqual(instance.stacks[0], (1, 3))
        self.assertEqual(instance.stacks[2], (2,))

    def test_loads_real_dataset_layout(self) -> None:
        instance = load_instance("data/dup_dataset/alpha=0.2/3-10-27/10001.txt")
        self.assertEqual(instance.stack_count, 10)
        self.assertEqual(instance.max_height, 3)
        self.assertEqual(instance.block_count, 27)
        self.assertEqual(instance.stacks[0], (3, 4, 2))
        self.assertEqual(len([stack for stack in instance.stacks if stack]), 10)

    def test_collects_paths_recursively(self) -> None:
        paths = collect_instance_paths(Path("data/fixtures"))
        self.assertEqual([path.name for path in paths], ["blocked.txt", "retrieval_only.txt"])


if __name__ == "__main__":
    unittest.main()
