#!/usr/bin/env python3
"""Render the benchmark chart embedded in the README "Performance" section.

The data below mirrors the measured-performance tables in docs/performance.md
(H200 EP8 indexed prefill, H200 TP8 indexed mix, H200 EP8 indexed decode,
B300 EP8 indexed decode, H200 EP8 grouped prefill/decode); when those tables
are re-measured, update the numbers here and re-run:

    python docs/assets/benchmark_chart.py

which writes `benchmark_chart.svg` (light) and `benchmark_chart_dark.svg`
next to this script. Requires matplotlib.
"""

from pathlib import Path

import matplotlib.pyplot as plt

# (title, x label, x tick labels, humming gate_up, chord gate_up,
#  humming down, chord down) — all times are per-call microseconds.
SCENARIOS = [
    (
        "H200 EP8 indexed prefill (WGMMA)",
        "routed tokens (total)",
        ["1024", "2048", "4096", "8192", "16384"],
        [334.9, 385.4, 569.5, 999.5, 1952.7],
        [297.4, 367.4, 546.9, 1001.1, 1692.5],
        [169.1, 207.1, 296.8, 524.1, 1003.4],
        [154.1, 196.1, 293.9, 541.0, 959.0],
    ),
    (
        # TP8 slices moe_intermediate, so the same token count is 8x the local
        # routed rows EP8 sees; the x axis stays num_tokens_total so the two H200
        # chunk panels are read at the same serving load.
        "H200 TP8 indexed mix (WGMMA)",
        "tokens (total)",
        ["1024", "2048", "4096", "8192", "16384"],
        [340.4, 395.2, 594.3, 1029.3, 2019.6],
        [300.5, 378.5, 566.9, 1040.3, 1820.7],
        [207.5, 309.9, 597.8, 955.5, 1840.9],
        [197.7, 298.1, 476.7, 853.7, 1591.4],
    ),
    (
        "H200 EP8 indexed decode (MMA swap-AB)",
        "tokens per GPU",
        ["20", "30", "40", "50"],
        [226.3, 257.8, 253.4, 255.2],
        [210.3, 239.3, 240.0, 248.0],
        [116.5, 131.9, 134.8, 135.1],
        [112.5, 118.5, 123.2, 127.5],
    ),
    (
        "B300 EP8 indexed decode (MMA swap-AB)",
        "tokens per GPU",
        ["20", "30", "40", "50"],
        [179.1, 191.4, 192.5, 192.9],
        [147.2, 165.7, 180.1, 188.9],
        [110.1, 113.9, 114.2, 114.3],
        [83.7, 91.8, 91.8, 94.8],
    ),
    (
        # The grouped panels are a different kernel family (DeepGEMM-derived TMA
        # + persistent WGMMA) against Humming's own grouped path, so they are
        # labelled by mode rather than by the indexed profile name.  Rows per
        # expert are multiples of the 128-row tile boundary on both sides, so
        # each point is the same GEMM shape for both implementations.
        "H200 EP8 grouped prefill (contiguous)",
        "rows per expert",
        ["128", "256", "512"],
        [865.5, 1432.3, 2781.2],
        [635.6, 1231.6, 2402.7],
        [427.9, 731.4, 1441.6],
        [325.2, 628.0, 1234.0],
    ),
    (
        "H200 EP8 grouped decode (masked)",
        "tokens per expert",
        ["8", "16", "32", "64"],
        [253.1, 341.5, 375.3, 468.4],
        [268.1, 286.6, 356.9, 455.1],
        [132.0, 170.7, 191.1, 242.1],
        [138.2, 147.5, 175.4, 233.6],
    ),
]

THEMES = {
    # chord line colors track the novita.ai brand palette (purple-500/400 and
    # blue-400/200); humming baselines stay neutral gray.
    "light": {
        "suffix": "",
        "text": "#24292f",
        "grid": "#d0d7de",
        "humming_gu": "#6e7781",
        "humming_down": "#8b949e",
        "chord_gu": "#8c54f4",
        "chord_down": "#18bfff",
    },
    "dark": {
        "suffix": "_dark",
        "text": "#c9d1d9",
        "grid": "#30363d",
        "humming_gu": "#8b949e",
        "humming_down": "#6e7681",
        "chord_gu": "#a78bfa",
        "chord_down": "#89dfff",
    },
}


def layer_speedup_range(humming_gu, chord_gu, humming_down, chord_down):
    ratios = [
        (h_gu + h_dn) / (c_gu + c_dn)
        for h_gu, c_gu, h_dn, c_dn in zip(humming_gu, chord_gu, humming_down, chord_down)
    ]
    return min(ratios), max(ratios)


def render(theme):
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "text.color": theme["text"],
            "axes.edgecolor": theme["grid"],
            "axes.labelcolor": theme["text"],
            "xtick.color": theme["text"],
            "ytick.color": theme["text"],
        }
    )
    # Three columns keeps the figure wide rather than tall, which reads better
    # embedded in the README; the row count follows SCENARIOS so adding a
    # scenario does not silently drop a panel.
    columns = 3
    rows = -(-len(SCENARIOS) // columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(8.6 * columns / 2, 4.1 * rows)
    )
    axes = axes.flatten()

    series_styles = [
        ("humming gate_up", theme["humming_gu"], "--", "o"),
        ("chord gate_up", theme["chord_gu"], "-", "o"),
        ("humming down", theme["humming_down"], "--", "s"),
        ("chord down", theme["chord_down"], "-", "s"),
    ]

    for ax, (title, xlabel, ticks, h_gu, c_gu, h_dn, c_dn) in zip(axes, SCENARIOS):
        x = list(range(len(ticks)))
        data = [h_gu, c_gu, h_dn, c_dn]
        for (label, color, linestyle, marker), ys in zip(series_styles, data):
            ax.plot(
                x,
                ys,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4.5,
                linewidth=1.8,
            )
        ax.set_xticks(x, ticks)
        ax.set_title(title, fontsize=11, fontweight="bold", color=theme["text"])
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color=theme["grid"], alpha=0.6, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        lo, hi = layer_speedup_range(h_gu, c_gu, h_dn, c_dn)
        ax.text(
            0.03,
            0.97,
            f"gate_up + down speedup: {lo:.2f}\u2013{hi:.2f}\u00d7",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color=theme["chord_gu"],
        )
        ax.margins(y=0.12)

    # Label the leftmost panel of every row, and hide any trailing empty cell an
    # odd scenario count leaves behind.
    for index in range(0, len(SCENARIOS), columns):
        axes[index].set_ylabel("\u00b5s per call (lower is better)")
    for ax in axes[len(SCENARIOS):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    # The legend sits above the grid, so reserve a slice of the figure for it that
    # shrinks as rows are added (a fixed 0.93 would eat a whole row's title space).
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.07 / rows))

    out = Path(__file__).with_name(f"benchmark_chart{theme['suffix']}.svg")
    fig.savefig(out, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    for theme in THEMES.values():
        render(theme)
