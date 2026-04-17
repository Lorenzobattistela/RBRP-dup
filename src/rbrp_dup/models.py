from __future__ import annotations

from dataclasses import dataclass, field

Action = tuple[int, int]
StateKey = tuple[int, ...]


@dataclass(frozen=True)
class BayState:
    stacks: tuple[tuple[int, ...], ...]
    max_height: int

    def __post_init__(self) -> None:
        if self.max_height <= 0:
            raise ValueError("max_height must be positive")
        for stack in self.stacks:
            if len(stack) > self.max_height:
                raise ValueError("stack height exceeds max_height")
            if any(priority <= 0 for priority in stack):
                raise ValueError("priorities must be positive integers")

    @property
    def stack_count(self) -> int:
        return len(self.stacks)

    @property
    def block_count(self) -> int:
        return sum(len(stack) for stack in self.stacks)

    def is_terminal(self) -> bool:
        return self.block_count == 0

    def top(self, stack_index: int) -> int | None:
        stack = self.stacks[stack_index]
        return stack[-1] if stack else None

    def height(self, stack_index: int) -> int:
        return len(self.stacks[stack_index])

    def is_full(self, stack_index: int) -> bool:
        return self.height(stack_index) >= self.max_height

    def flatten(self) -> StateKey:
        flat: list[int] = []
        for stack in self.stacks:
            flat.extend(stack)
            flat.extend([0] * (self.max_height - len(stack)))
        return tuple(flat)

    def format_matrix(self) -> str:
        rows: list[str] = []
        for tier in range(self.max_height - 1, -1, -1):
            cells: list[str] = []
            for stack in self.stacks:
                value = stack[tier] if tier < len(stack) else 0
                cells.append(f"{value:>2}" if value else " .")
            rows.append(" ".join(cells))
        footer = " ".join(f"{index + 1:>2}" for index in range(self.stack_count))
        return "\n".join([*rows, footer])


@dataclass(frozen=True)
class BayInstance:
    stack_count: int
    max_height: int
    block_count: int
    stacks: tuple[tuple[int, ...], ...]
    instance_id: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if self.stack_count <= 0:
            raise ValueError("stack_count must be positive")
        if len(self.stacks) != self.stack_count:
            raise ValueError("stacks length must match stack_count")
        if sum(len(stack) for stack in self.stacks) != self.block_count:
            raise ValueError("block_count does not match stacks")
        if any(len(stack) > self.max_height for stack in self.stacks):
            raise ValueError("stack height exceeds max_height")

    def initial_state(self) -> BayState:
        return BayState(stacks=self.stacks, max_height=self.max_height)


@dataclass(frozen=True)
class TrainingConfig:
    alpha: float = 0.75
    gamma: float = 1.0
    epsilon: float = 0.1
    max_episodes: int = 250
    time_limit_seconds: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        if not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if self.max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")


QTable = dict[StateKey, dict[Action, float]]


@dataclass
class TrainingArtifact:
    q_table: QTable
    config: TrainingConfig
    episodes_run: int
    best_episode_reward: float


@dataclass(frozen=True)
class TraceStep:
    action: Action
    reward: int
    policy_source: str
    before: str
    after: str


@dataclass
class Solution:
    instance_id: str | None
    relocations: int
    actions: list[Action]
    trace: list[TraceStep] = field(default_factory=list)
    q_table_actions: int = 0
    heuristic_actions: int = 0


@dataclass(frozen=True)
class BatchItem:
    instance_id: str
    relocations: int
    actions: int
    q_table_actions: int
    heuristic_actions: int
    episodes_run: int
    runtime_seconds: float


@dataclass
class BatchReport:
    items: list[BatchItem]

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def average_relocations(self) -> float:
        if not self.items:
            return 0.0
        return sum(item.relocations for item in self.items) / len(self.items)

    @property
    def total_runtime_seconds(self) -> float:
        return sum(item.runtime_seconds for item in self.items)
