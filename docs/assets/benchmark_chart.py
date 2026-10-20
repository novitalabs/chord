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
     [299.9, 364.7, 563.1, 1032.3, 1939.1], [296.8, 367.0, 545.9, 999.0, 1698.2],
     [154.5, 193.9, 296.6, 525.3, 1003.0], [153.3, 195.9, 293.9, 542.1, 962.5]),
    ("H200 TP8\nindexed, tokens", ["1k", "2k", "4k", "8k", "16k"],
     [307.0, 373.1, 603.3, 1070.8, 2064.1], [300.7, 377.7, 567.4, 1039.7, 1826.1],
     [208.1, 311.1, 519.5, 956.0, 1841.8], [198.8, 298.1, 474.2, 852.3, 1584.9]),
    ("H200 EP8 decode\nindexed, tokens/GPU", ["20", "30", "40", "50"],
     [215.6, 246.8, 253.8, 256.0], [210.7, 236.8, 239.6, 248.8],
     [116.2, 132.1, 134.4, 135.1], [112.4, 118.6, 122.5, 127.0]),
    ("B300 EP8 decode\nindexed, tokens/GPU", ["20", "30", "40", "50"],
     [149.1, 168.7, 181.2, 185.6], [147.8, 167.2, 181.9, 183.8],
     [85.1, 94.0, 98.1, 102.3], [83.8, 92.0, 92.0, 95.0]),
    ("H200 EP8 prefill\ncontiguous, rows/expert", ["128", "256", "512"],
     [750.5, 1338.2, 2251.0], [596.5, 1137.3, 2238.8],
     [384.2, 673.1, 1178.5], [322.8, 581.5, 1122.7]),
    ("H200 EP8 decode\nmasked, tokens/expert", ["8", "16", "32", "64"],
     [253.0, 287.3, 353.7, 531.0], [241.6, 261.1, 338.1, 464.4],
     [132.0, 147.2, 171.9, 267.8], [124.5, 134.3, 169.6, 229.4]),
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
