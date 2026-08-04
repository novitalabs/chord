#!/usr/bin/env python3
"""Render the benchmark chart embedded in the README "Performance" section.

The data below mirrors the three measured-performance tables in
docs/performance.md (H200 EP8 prefill, H200 EP8 decode, B300 EP8 decode); when
those tables are re-measured, update the numbers here and re-run:

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
        "H200 EP8 prefill (WGMMA)",
        "routed tokens (total)",
        ["1024", "2048", "4096", "8196", "16384"],
        [375.0, 466.3, 608.4, 1085.7, 1903.0],
        [313.5, 383.4, 545.8, 997.7, 1743.1],
        [195.9, 235.1, 337.4, 606.4, 1059.4],
        [162.9, 204.5, 293.2, 533.4, 937.3],
    ),
    (
        "H200 EP8 decode (MMA swap-AB)",
        "tokens per GPU",
        ["20", "30", "40", "50"],
        [267.6, 281.3, 284.0, 296.4],
        [221.9, 245.6, 253.7, 259.9],
        [146.2, 152.2, 153.3, 156.3],
        [111.9, 116.8, 123.0, 126.4],
    ),
    (
        "B300 EP8 decode (MMA swap-AB)",
        "tokens per GPU",
        ["20", "30", "40", "50"],
        [319.1, 320.4, 320.6, 320.9],
        [146.0, 162.3, 175.2, 182.9],
        [174.8, 181.3, 181.4, 181.7],
        [83.8, 91.3, 91.9, 95.1],
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
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1))

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

    axes[0].set_ylabel("\u00b5s per call (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = Path(__file__).with_name(f"benchmark_chart{theme['suffix']}.svg")
    fig.savefig(out, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    for theme in THEMES.values():
        render(theme)
