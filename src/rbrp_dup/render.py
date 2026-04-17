from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .bay import apply_action, target_priority
from .models import BayInstance, BayState, Solution, TraceStep

Color = tuple[int, int, int]

BACKGROUND: Color = (247, 244, 236)
TEXT: Color = (44, 44, 44)
GRID: Color = (214, 209, 198)
GRID_FILL: Color = (252, 250, 245)
ACCENT: Color = (60, 102, 173)

PALETTE: tuple[Color, ...] = (
    (225, 102, 92),
    (235, 155, 71),
    (242, 201, 76),
    (135, 189, 99),
    (80, 166, 164),
    (81, 126, 217),
    (147, 103, 196),
    (198, 106, 156),
    (121, 120, 120),
    (89, 149, 92),
)

FONT_5X7: dict[str, tuple[str, ...]] = {
    " ": (".....",) * 7,
    ",": (".....", ".....", ".....", ".....", "..##.", "..##.", ".##.."),
    "-": (".....", ".....", ".###.", ".....", ".....", ".....", "....."),
    ".": (".....", ".....", ".....", ".....", ".....", "..##.", "..##."),
    "/": ("....#", "...#.", "..#..", ".#...", "#....", ".....", "....."),
    "(": ("...#.", "..#..", ".#...", ".#...", ".#...", "..#..", "...#."),
    ")": (".#...", "..#..", "...#.", "...#.", "...#.", "..#..", ".#..."),
    ":": (".....", "..##.", "..##.", ".....", "..##.", "..##.", "....."),
    "?": (".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."),
    "0": (".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."),
    "1": ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    "2": (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    "3": ("#####", "....#", "...#.", "..##.", "....#", "#...#", ".###."),
    "4": ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    "5": ("#####", "#....", "####.", "....#", "....#", "#...#", ".###."),
    "6": (".###.", "#...#", "#....", "####.", "#...#", "#...#", ".###."),
    "7": ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    "8": (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
    "9": (".###.", "#...#", "#...#", ".####", "....#", "#...#", ".###."),
    "A": (".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "B": ("####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."),
    "C": (".####", "#....", "#....", "#....", "#....", "#....", ".####"),
    "D": ("####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "#....", "####.", "#....", "#....", "#####"),
    "F": ("#####", "#....", "#....", "####.", "#....", "#....", "#...."),
    "G": (".####", "#....", "#....", "#.###", "#...#", "#...#", ".###."),
    "H": ("#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"),
    "I": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "#####"),
    "J": ("..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."),
    "K": ("#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "#...#", "####.", "#....", "#....", "#...."),
    "Q": (".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"),
    "S": (".####", "#....", "#....", ".###.", "....#", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"),
    "X": ("#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"),
    "Y": ("#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."),
    "Z": ("#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"),
}


class Raster:
    def __init__(self, width: int, height: int, background: Color) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(background * (width * height))

    def fill_rect(self, x: int, y: int, width: int, height: int, color: Color) -> None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + width)
        y1 = min(self.height, y + height)
        if x0 >= x1 or y0 >= y1:
            return

        row = bytes(color) * (x1 - x0)
        for row_index in range(y0, y1):
            start = (row_index * self.width + x0) * 3
            end = start + (x1 - x0) * 3
            self.pixels[start:end] = row

    def draw_rect(self, x: int, y: int, width: int, height: int, color: Color, thickness: int = 1) -> None:
        self.fill_rect(x, y, width, thickness, color)
        self.fill_rect(x, y + height - thickness, width, thickness, color)
        self.fill_rect(x, y, thickness, height, color)
        self.fill_rect(x + width - thickness, y, thickness, height, color)

    def draw_text(self, x: int, y: int, text: str, color: Color, scale: int = 2) -> None:
        cursor_x = x
        for character in text.upper():
            glyph = FONT_5X7.get(character, FONT_5X7["?"])
            for row_index, row in enumerate(glyph):
                for column_index, bit in enumerate(row):
                    if bit == "#":
                        self.fill_rect(
                            cursor_x + column_index * scale,
                            y + row_index * scale,
                            scale,
                            scale,
                            color,
                        )
            cursor_x += (len(glyph[0]) + 1) * scale

    def save_ppm(self, path: Path) -> None:
        with path.open("wb") as handle:
            handle.write(f"P6\n{self.width} {self.height}\n255\n".encode("ascii"))
            handle.write(self.pixels)


def render_solution_gif(
    instance: BayInstance,
    solution: Solution,
    output_path: str | Path,
    fps: float = 1.25,
) -> Path:
    if not solution.trace:
        raise ValueError("solution trace is required to render a GIF")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        frames_dir = Path(temp_dir)
        current_state = instance.initial_state()
        frame_index = 0
        total_steps = len(solution.trace)
        relocations = 0

        # Keep the initial and final states visible for longer by duplicating them.
        for _ in range(2):
            _render_frame(
                current_state,
                instance=instance,
                step_index=0,
                total_steps=total_steps,
                relocations=0,
                step=None,
                path=frames_dir / f"frame_{frame_index:04d}.ppm",
            )
            frame_index += 1

        for step_number, step in enumerate(solution.trace, start=1):
            current_state, reward = apply_action(current_state, step.action)
            relocations += -reward
            _render_frame(
                current_state,
                instance=instance,
                step_index=step_number,
                total_steps=total_steps,
                relocations=relocations,
                step=step,
                path=frames_dir / f"frame_{frame_index:04d}.ppm",
            )
            frame_index += 1

        for _ in range(2):
            _render_frame(
                current_state,
                instance=instance,
                step_index=total_steps,
                total_steps=total_steps,
                relocations=relocations,
                step=solution.trace[-1],
                path=frames_dir / f"frame_{frame_index:04d}.ppm",
            )
            frame_index += 1

        _build_gif(frames_dir, output, fps)

    return output


def _render_frame(
    state: BayState,
    *,
    instance: BayInstance,
    step_index: int,
    total_steps: int,
    relocations: int,
    step: TraceStep | None,
    path: Path,
) -> None:
    cell_width = 62
    cell_height = 42
    stack_gap = 14
    left_margin = 48
    top_margin = 94
    bottom_margin = 58
    width = left_margin * 2 + state.stack_count * cell_width + (state.stack_count - 1) * stack_gap
    height = top_margin + state.max_height * cell_height + bottom_margin

    raster = Raster(width, height, BACKGROUND)
    raster.draw_text(24, 18, f"ID {Path(instance.source or instance.instance_id or 'INSTANCE').stem}", TEXT, scale=3)

    if step is None:
        raster.draw_text(24, 48, f"STEP 0 OF {total_steps} START", ACCENT, scale=2)
        raster.draw_text(24, 68, "SOURCE NONE RELOCS 0", TEXT, scale=2)
    else:
        source_label = "QTABLE" if step.policy_source == "q_table" else "HEUR"
        raster.draw_text(
            24,
            48,
            f"STEP {step_index} OF {total_steps} ACTION ({step.action[0]},{step.action[1]})",
            ACCENT,
            scale=2,
        )
        raster.draw_text(24, 68, f"SOURCE {source_label} RELOCS {relocations}", TEXT, scale=2)

    current_target = target_priority(state)
    if current_target is not None:
        raster.draw_text(width - 190, 18, f"TARGET {current_target}", TEXT, scale=2)

    for stack_index in range(state.stack_count):
        x = left_margin + stack_index * (cell_width + stack_gap)
        for tier in range(state.max_height):
            y = top_margin + (state.max_height - tier - 1) * cell_height
            raster.fill_rect(x, y, cell_width, cell_height, GRID_FILL)
            raster.draw_rect(x, y, cell_width, cell_height, GRID, thickness=2)

            if tier >= len(state.stacks[stack_index]):
                continue

            priority = state.stacks[stack_index][tier]
            block_color = PALETTE[(priority - 1) % len(PALETTE)]
            raster.fill_rect(x + 3, y + 3, cell_width - 6, cell_height - 6, block_color)
            raster.draw_rect(x + 3, y + 3, cell_width - 6, cell_height - 6, TEXT, thickness=2)

            label = str(priority)
            label_width = _text_width(label, scale=3)
            label_color = (255, 255, 255) if _brightness(block_color) < 145 else TEXT
            raster.draw_text(
                x + (cell_width - label_width) // 2,
                y + 10,
                label,
                label_color,
                scale=3,
            )

        raster.draw_text(x + 18, top_margin + state.max_height * cell_height + 14, str(stack_index + 1), TEXT, scale=2)

    raster.save_ppm(path)


def _build_gif(frames_dir: Path, output_path: Path, fps: float) -> None:
    palette_path = frames_dir / "palette.png"
    input_pattern = str(frames_dir / "frame_%04d.ppm")

    palette_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        input_pattern,
        "-vf",
        "palettegen",
        str(palette_path),
    ]
    gif_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        input_pattern,
        "-i",
        str(palette_path),
        "-lavfi",
        "paletteuse",
        "-loop",
        "0",
        str(output_path),
    ]

    subprocess.run(palette_cmd, check=True)
    subprocess.run(gif_cmd, check=True)


def _text_width(text: str, scale: int) -> int:
    glyph_width = 5 * scale
    spacing = scale
    return sum(glyph_width + spacing for _ in text) - spacing


def _brightness(color: Color) -> float:
    red, green, blue = color
    return red * 0.299 + green * 0.587 + blue * 0.114
