#!/usr/bin/env python3
"""Render the README performance chart from the tables in docs/performance.md.

    python docs/assets/benchmark_chart.py

writes benchmark_chart.svg and benchmark_chart_dark.svg next to this script.
Each bar is speedup(gate_up + down) at one size: Humming's summed gate_up and
down time over chord's, on the same GPU.  Requires matplotlib.
"""

from pathlib import Path

import matplotlib.pyplot as plt

# (group label, size labels, humming gate_up, chord gate_up, humming down,
# chord down); per-call microseconds from docs/performance.md.
SCENARIOS = [
    ("H200 EP8 prefill\nindexed, tokens", ["1k", "2k", "4k", "8k", "16k"],
     [334.9, 385.4, 569.5, 999.5, 1952.7], [297.4, 367.4, 546.9, 1001.1, 1692.5],
     [169.1, 207.1, 296.8, 524.1, 1003.4], [154.1, 196.1, 293.9, 541.0, 959.0]),
    ("H200 TP8\nindexed, tokens", ["1k", "2k", "4k", "8k", "16k"],
     [340.4, 395.2, 594.3, 1029.3, 2019.6], [300.5, 378.5, 566.9, 1040.3, 1820.7],
     [207.5, 309.9, 597.8, 955.5, 1840.9], [197.7, 298.1, 476.7, 853.7, 1591.4]),
    ("H200 EP8 decode\nindexed, tokens/GPU", ["20", "30", "40", "50"],
     [226.3, 257.8, 253.4, 255.2], [210.3, 239.3, 240.0, 248.0],
     [116.5, 131.9, 134.8, 135.1], [112.5, 118.5, 123.2, 127.5]),
    ("B300 EP8 decode\nindexed, tokens/GPU", ["20", "30", "40", "50"],
     [179.1, 191.4, 192.5, 192.9], [147.2, 165.7, 180.1, 188.9],
     [110.1, 113.9, 114.2, 114.3], [83.7, 91.8, 91.8, 94.8]),
    ("H200 EP8 prefill\ncontiguous, rows/expert", ["128", "256", "512"],
     [865.5, 1432.3, 2781.2], [635.6, 1231.6, 2402.7],
     [427.9, 731.4, 1441.6], [325.2, 628.0, 1234.0]),
    ("H200 EP8 decode\nmasked, tokens/expert", ["8", "16", "32", "64"],
     [253.1, 341.5, 375.3, 468.4], [268.1, 286.6, 356.9, 455.1],
     [132.0, 170.7, 191.1, 242.1], [138.2, 147.5, 175.4, 233.6]),
]

THEMES = {
    "light": {"suffix": "", "text": "#24292f", "grid": "#d0d7de",
              "bar": "#8c54f4", "below": "#8b949e", "base": "#57606a"},
    "dark": {"suffix": "_dark", "text": "#c9d1d9", "grid": "#30363d",
             "bar": "#a78bfa", "below": "#6e7681", "base": "#8b949e"},
}


def render(theme: dict) -> None:
    plt.rcParams.update({"font.size": 9, "text.color": theme["text"],
                         "axes.edgecolor": theme["grid"], "xtick.color": theme["text"],
                         "ytick.color": theme["text"], "axes.labelcolor": theme["text"],
                         "svg.hashsalt": "chord"})
    fig, ax = plt.subplots(figsize=(13, 3.6))
    x, gap, centers = 0.0, 1.0, []
    for label, sizes, h_gu, c_gu, h_dn, c_dn in SCENARIOS:
        speedups = [(a + b) / (c + d) for a, c, b, d in zip(h_gu, c_gu, h_dn, c_dn)]
        xs = [x + i for i in range(len(sizes))]
        colors = [theme["bar"] if s >= 1 else theme["below"] for s in speedups]
        ax.bar(xs, speedups, width=0.72, color=colors, zorder=3)
        for xi, s in zip(xs, speedups):
            ax.text(xi, s + 0.01, f"{s:.2f}", ha="center", va="bottom", fontsize=7.5)
        centers.append((sum(xs) / len(xs), label, xs, sizes))
        x = xs[-1] + 1 + gap
    ax.axhline(1.0, color=theme["base"], linestyle="--", linewidth=1, zorder=2)
    ax.text(x - gap, 1.0, " Humming", ha="left", va="center", fontsize=8,
            color=theme["base"], clip_on=False)
    ax.set_xticks([t for _, _, xs, _ in centers for t in xs],
                  [s for _, _, _, sizes in centers for s in sizes], fontsize=8)
    ax.tick_params(axis="x", length=0)
    for center, label, _, _ in centers:
        ax.text(center, -0.13, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9)
    ax.set_xlim(-0.8, x - gap - 0.2)
    ax.set_ylim(0.8, 1.45)
    ax.set_ylabel("speedup(gate_up + down)")
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = Path(__file__).with_name(f"benchmark_chart{theme['suffix']}.svg")
    fig.savefig(out, transparent=True, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    for theme in THEMES.values():
        render(theme)
