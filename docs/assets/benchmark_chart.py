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
     [300.0, 364.9, 563.8, 1031.4, 1947.2], [297.1, 367.9, 545.3, 999.9, 1697.8],
     [154.9, 195.4, 297.2, 524.4, 1004.5], [154.1, 195.8, 293.2, 540.8, 961.0]),
    ("H200 TP8\nindexed, tokens", ["1k", "2k", "4k", "8k", "16k"],
     [307.1, 372.4, 603.0, 1068.0, 2061.3], [300.4, 377.2, 566.0, 1042.5, 1821.3],
     [209.2, 311.8, 519.5, 956.4, 1840.0], [199.0, 297.9, 475.4, 852.5, 1585.3]),
    ("H200 EP8 decode\nindexed, tokens/GPU", ["20", "30", "40", "50"],
     [215.0, 247.1, 253.9, 255.8], [209.3, 238.8, 239.4, 248.2],
     [116.2, 132.2, 134.1, 136.0], [112.2, 118.5, 123.0, 127.2]),
    ("B300 EP8 decode\nindexed, tokens/GPU", ["20", "30", "40", "50"],
     [149.1, 168.5, 181.5, 185.5], [147.7, 167.2, 181.7, 189.5],
     [85.2, 94.2, 98.0, 102.3], [83.8, 92.0, 92.0, 95.0]),
    ("H200 EP8 prefill\ncontiguous, rows/expert", ["128", "256", "512"],
     [762.3, 1338.4, 2249.0], [631.2, 1220.5, 2369.3],
     [383.7, 674.4, 1183.3], [322.9, 620.5, 1223.8]),
    ("H200 EP8 decode\nmasked, tokens/expert", ["8", "16", "32", "64"],
     [253.1, 287.9, 354.5, 531.4], [242.1, 261.8, 340.6, 464.4],
     [131.8, 147.7, 172.2, 267.7], [124.7, 134.1, 170.3, 229.8]),
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
