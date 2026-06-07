"""Generate portfolio-friendly evaluation charts.

The metric values below are copied from the current v1.3
`evaluation_report_lastfm.md` Last.fm candidate-aware benchmark table.
Keeping them explicit avoids brittle Markdown parsing while making the chart
source easy to audit.
"""

from __future__ import annotations

from pathlib import Path
import struct
import zlib


try:  # Matplotlib is optional for local chart regeneration.
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    plt = None


CHART_DIR = Path("docs") / "images"

MODEL_LABELS = [
    "Content",
    "Collaborative",
    "Hybrid",
    "ALS",
    "Word2Vec",
    "Hybrid + ALS",
    "Hybrid + ALS + Word2Vec",
]

FALLBACK_MODEL_LABELS = [
    "CONTENT",
    "COLLAB",
    "HYBRID",
    "ALS",
    "WORD2VEC",
    "HYBRID+ALS",
    "HYBRID+ALS+W2V",
]

PRECISION_AT_K = [0.023, 0.000, 0.010, 0.000, 0.000, 0.009, 0.009]
NDCG_AT_K = [0.099, 0.001, 0.071, 0.003, 0.001, 0.069, 0.065]

BAR_COLORS = [
    "#1DB954",
    "#4C78A8",
    "#72B7B2",
    "#F58518",
    "#B279A2",
    "#54A24B",
    "#E45756",
    ]


SIMPLE_FONT = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    "+": ["000", "010", "010", "111", "010", "010", "000"],
    "-": ["000", "000", "000", "111", "000", "000", "000"],
    ".": ["000", "000", "000", "000", "000", "011", "011"],
    "/": ["001", "001", "010", "010", "100", "100", "000"],
    "@": ["01110", "10001", "10111", "10101", "10111", "10000", "01111"],
    "0": ["111", "101", "101", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "010", "010", "111"],
    "2": ["111", "001", "001", "111", "100", "100", "111"],
    "3": ["111", "001", "001", "111", "001", "001", "111"],
    "4": ["101", "101", "101", "111", "001", "001", "001"],
    "5": ["111", "100", "100", "111", "001", "001", "111"],
    "6": ["111", "100", "100", "111", "101", "101", "111"],
    "7": ["111", "001", "001", "010", "010", "100", "100"],
    "8": ["111", "101", "101", "111", "101", "101", "111"],
    "9": ["111", "101", "101", "111", "001", "001", "111"],
    "A": ["010", "101", "101", "111", "101", "101", "101"],
    "B": ["110", "101", "101", "110", "101", "101", "110"],
    "C": ["011", "100", "100", "100", "100", "100", "011"],
    "D": ["110", "101", "101", "101", "101", "101", "110"],
    "E": ["111", "100", "100", "110", "100", "100", "111"],
    "F": ["111", "100", "100", "110", "100", "100", "100"],
    "G": ["011", "100", "100", "101", "101", "101", "011"],
    "H": ["101", "101", "101", "111", "101", "101", "101"],
    "I": ["111", "010", "010", "010", "010", "010", "111"],
    "J": ["001", "001", "001", "001", "101", "101", "010"],
    "K": ["101", "101", "110", "100", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "100", "100", "111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["1001", "1101", "1011", "1001", "1001", "1001", "1001"],
    "O": ["010", "101", "101", "101", "101", "101", "010"],
    "P": ["110", "101", "101", "110", "100", "100", "100"],
    "Q": ["010", "101", "101", "101", "101", "110", "011"],
    "R": ["110", "101", "101", "110", "110", "101", "101"],
    "S": ["011", "100", "100", "010", "001", "001", "110"],
    "T": ["111", "010", "010", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "101", "101", "010"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["101", "101", "101", "010", "101", "101", "101"],
    "Y": ["101", "101", "101", "010", "010", "010", "010"],
    "Z": ["111", "001", "001", "010", "100", "100", "111"],
}


def main() -> None:
    """Create Precision@K and NDCG@K comparison charts."""

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    _write_chart(
        values=PRECISION_AT_K,
        title="Last.fm Evaluation: Precision@10 by Model",
        ylabel="Precision@10",
        output_path=CHART_DIR / "evaluation_precision.png",
    )
    _write_chart(
        values=NDCG_AT_K,
        title="Last.fm Evaluation: NDCG@10 by Model",
        ylabel="NDCG@10",
        output_path=CHART_DIR / "evaluation_ndcg.png",
    )


def _write_chart(values: list[float], title: str, ylabel: str, output_path: Path) -> None:
    """Write one bar chart with Matplotlib or a small dependency-free fallback."""

    if plt is None:
        _write_fallback_png(values=values, title=title, ylabel=ylabel, output_path=output_path)
        return

    fig, axis = plt.subplots(figsize=(10, 5.4))
    bars = axis.bar(MODEL_LABELS, values, color=BAR_COLORS)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, max(max(values) * 1.25, 0.12))
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.tick_params(axis="x", rotation=25)
    for label in axis.get_xticklabels():
        label.set_ha("right")
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_fallback_png(values: list[float], title: str, ylabel: str, output_path: Path) -> None:
    """Write a simple PNG bar chart without third-party dependencies."""

    width = 1200
    height = 620
    margin_left = 90
    margin_right = 50
    margin_top = 70
    margin_bottom = 130
    background = (255, 255, 255)
    axis_color = (45, 55, 72)
    text_color = (31, 41, 55)
    grid_color = (225, 232, 240)
    pixels = bytearray(background * width * height)

    def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(color)

    def fill_rect(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(max(y0, 0), min(y1, height)):
            for x in range(max(x0, 0), min(x1, width)):
                set_pixel(x, y, color)

    def draw_char(character: str, x: int, y: int, color: tuple[int, int, int], scale: int) -> int:
        glyph = SIMPLE_FONT.get(character.upper(), SIMPLE_FONT[" "])
        for row_index, row in enumerate(glyph):
            for column_index, pixel in enumerate(row):
                if pixel == "1":
                    fill_rect(
                        x + column_index * scale,
                        y + row_index * scale,
                        x + (column_index + 1) * scale,
                        y + (row_index + 1) * scale,
                        color,
                    )
        return (len(glyph[0]) + 1) * scale

    def draw_text(text: str, x: int, y: int, color: tuple[int, int, int], scale: int = 2) -> None:
        cursor_x = x
        for character in text:
            cursor_x += draw_char(character, cursor_x, y, color, scale)

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    base_y = margin_top + plot_height
    max_value = max(max(values) * 1.25, 0.12)

    for tick in range(6):
        y = int(base_y - (plot_height * tick / 5))
        fill_rect(margin_left, y, width - margin_right, y + 1, grid_color)
        tick_value = max_value * tick / 5
        draw_text(f"{tick_value:.2f}", 22, y - 8, text_color, scale=2)
    fill_rect(margin_left, margin_top, margin_left + 2, base_y, axis_color)
    fill_rect(margin_left, base_y, width - margin_right, base_y + 2, axis_color)
    draw_text(title.upper().replace("LAST.FM EVALUATION: ", "LAST.FM "), 85, 24, text_color, scale=3)
    draw_text(ylabel.upper(), 18, 40, text_color, scale=2)

    slot_width = plot_width / len(values)
    bar_width = int(slot_width * 0.58)
    for index, value in enumerate(values):
        x_center = int(margin_left + slot_width * index + slot_width / 2)
        bar_height = int((value / max_value) * plot_height)
        x0 = x_center - bar_width // 2
        y0 = base_y - bar_height
        color = _hex_to_rgb(BAR_COLORS[index])
        fill_rect(x0, y0, x0 + bar_width, base_y, color)
        draw_text(f"{value:.3f}", x_center - 24, max(y0 - 24, margin_top + 4), text_color, scale=2)
        draw_text(FALLBACK_MODEL_LABELS[index], x_center - 48, base_y + 16, text_color, scale=2)

    _write_png(output_path, width, height, bytes(pixels))


def _write_png(output_path: Path, width: int, height: int, rgb_pixels: bytes) -> None:
    """Write raw RGB pixels as a PNG file."""

    scanlines = b"".join(
        b"\x00" + rgb_pixels[row * width * 3 : (row + 1) * width * 3]
        for row in range(height)
    )

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + chunk(b"IEND", b"")
    )
    output_path.write_bytes(png_bytes)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Convert a hex color string to an RGB tuple."""

    normalized = value.lstrip("#")
    return (
        int(normalized[0:2], 16),
        int(normalized[2:4], 16),
        int(normalized[4:6], 16),
    )


if __name__ == "__main__":
    main()
