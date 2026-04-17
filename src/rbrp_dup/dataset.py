from __future__ import annotations

import re
from pathlib import Path

from .models import BayInstance

INSTANCE_DIR_PATTERN = re.compile(r"^(?P<height>\d+)-(?P<stacks>\d+)-(?P<blocks>\d+)$")


def load_instance(path: str | Path) -> BayInstance:
    instance_path = Path(path)
    lines = [
        line.strip()
        for line in instance_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"{instance_path} is empty")

    header = [int(token) for token in lines[0].split()]
    if len(header) != 2:
        raise ValueError(f"invalid header in {instance_path}")
    stack_count, block_count = header

    stacks: list[tuple[int, ...]] = []
    for line in lines[1:]:
        values = [int(token) for token in line.split()]
        if not values:
            continue
        declared_count = values[0]
        # Dataset rows store each non-empty stack as "count priority priority ...",
        # ordered from bottom to top.
        priorities = values[1:]
        if declared_count != len(priorities):
            raise ValueError(f"stack line has wrong count in {instance_path}: {line}")
        stacks.append(tuple(priorities))

    if len(stacks) > stack_count:
        raise ValueError(f"{instance_path} contains more stacks than declared")

    while len(stacks) < stack_count:
        stacks.append(tuple())

    max_height = _infer_max_height(instance_path, stack_count, block_count, stacks)
    return BayInstance(
        stack_count=stack_count,
        max_height=max_height,
        block_count=block_count,
        stacks=tuple(stacks),
        instance_id=_instance_id(instance_path),
        source=str(instance_path),
    )


def collect_instance_paths(path: str | Path, limit: int | None = None) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]

    paths = sorted(candidate for candidate in root.rglob("*.txt") if candidate.is_file())
    if limit is not None:
        return paths[:limit]
    return paths


def _infer_max_height(
    path: Path,
    stack_count: int,
    block_count: int,
    stacks: list[tuple[int, ...]],
) -> int:
    for parent in [path.parent, *path.parents]:
        match = INSTANCE_DIR_PATTERN.match(parent.name)
        if not match:
            continue
        candidate_stacks = int(match.group("stacks"))
        candidate_blocks = int(match.group("blocks"))
        if candidate_stacks == stack_count and candidate_blocks == block_count:
            # The TT directory names encode H-W-N; prefer that over inferring height
            # from observed stacks because some instances may omit empty space.
            return int(match.group("height"))
    return max((len(stack) for stack in stacks), default=1)


def _instance_id(path: Path) -> str:
    parts = path.parts
    if "data" in parts:
        data_index = parts.index("data")
        return "/".join(parts[data_index + 1 :])
    return str(path)
