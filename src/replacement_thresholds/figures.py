"""Figures 1–3 of the paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

STATIC_COLOR, DYNAMIC_COLOR, INK = "#2a78d6", "#eb6834", "#222222"
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5,
    "mathtext.fontset": "stix",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#6b6b6b",
    "axes.linewidth": 0.8,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.grid": True,
    "grid.color": "#e4e4e4",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
}


def _save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{name}.tiff", dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def replacement_cycle(directory):
    fig, ax = plt.subplots(figsize=(4.72, 2.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    width, height = 4.3, 1.7
    boxes = [
        (0.3, 3.6, "1. Score each active item:\n$Z_{j,t}$ from responses up to period $t-1$"),
        (5.4, 3.6, "2. Replace the top item only if\nmax $Z_{j,t}$ > threshold ($z_S$ or $\\tau_t$)"),
        (5.4, 0.9, "3. If replaced: $m$ responses go to\ncalibration; reserve $R_t$ decreases by 1"),
        (0.3, 0.9, "4. Remaining responses are diagnostic;\nupdate the learner filter\nand item sums $S, V$"),
    ]
    for x, y, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#f4f6f9", ec="#6b6b6b", lw=0.8
            )
        )
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=7.0, color=INK)
    arrow = dict(arrowstyle="-|>", color="#444444", lw=0.8)
    ax.annotate("", xy=(5.4, 4.45), xytext=(4.6, 4.45), arrowprops=arrow)
    ax.annotate("", xy=(7.55, 2.6), xytext=(7.55, 3.6), arrowprops=arrow)
    ax.annotate("", xy=(4.6, 1.75), xytext=(5.4, 1.75), arrowprops=arrow)
    ax.annotate("", xy=(2.45, 3.6), xytext=(2.45, 2.6), arrowprops=dict(arrow, color="#888888", ls="--"))
    ax.text(2.6, 3.1, "next period", fontsize=6.8, color="#555555", va="center")
    ax.text(
        5.0,
        5.7,
        "Each period: 96 responses; at most one replacement; none in period 1 or after exhaustion",
        ha="center",
        fontsize=7,
        color=INK,
    )
    _save(fig, directory, "Fig1_cycle")


def contrasts(directory, analysis, by_period):
    registered, equivalence = analysis["registered"], analysis["equivalence"]
    rows = [
        ("A: all cells (97.5%)", registered["dynamic_minus_static"]),
        ("A: equivalence (95%)", equivalence),
        ("C: $R_0=3$", registered["dynamic_minus_static_R3"]),
        ("C: $R_0=6$", registered["dynamic_minus_static_R6"]),
        ("C: $R_0=12$", registered["dynamic_minus_static_R12"]),
        ("D: zero practice", registered["dynamic_minus_static_zero_practice"]),
    ]
    fig, (a, b) = plt.subplots(2, 1, figsize=(4.72, 5.2), gridspec_kw={"height_ratios": [1, 1.1]})
    y = list(range(len(rows)))[::-1]
    for x, color in ((-1.0, "#777777"), (1.0, "#777777"), (0.0, "#999999")):
        a.axvline(x, color=color, lw=0.9 if x else 0.8, ls=":" if x else "-")
    for yi, (_, row) in zip(y, rows, strict=True):
        a.plot([row["lower"] * 1e3, row["upper"] * 1e3], [yi, yi], color=INK, lw=1.6, solid_capstyle="round")
        a.plot(row["mean"] * 1e3, yi, "o", ms=4, color=INK)
    a.set_ylim(-0.6, 6.0)
    a.set_yticks(y)
    a.set_yticklabels([label for label, _ in rows], fontsize=8)
    a.grid(axis="y", visible=False)
    a.set_xlim(-1.1, 1.1)
    a.set_xlabel("Dynamic − static RMSE (×10$^{-3}$)")
    a.text(
        0.95,
        5.55,
        "dotted: ±0.001 equivalence\nmargin (family A only)",
        fontsize=6.3,
        color="#555555",
        ha="right",
        va="center",
    )
    a.set_title("(a) Registered contrasts, full horizon", fontsize=8.5, loc="left")

    t = by_period["period"]
    mean = [m * 1e3 for m in by_period["mean"]]
    lower = [(m - 1.96 * s) * 1e3 for m, s in zip(by_period["mean"], by_period["se"], strict=True)]
    upper = [(m + 1.96 * s) * 1e3 for m, s in zip(by_period["mean"], by_period["se"], strict=True)]
    b.axhline(0, color="#999999", lw=0.8)
    b.fill_between(t, lower, upper, color="#d9d9d9", lw=0)
    b.plot(t, mean, color=INK, lw=1.5, marker="o", ms=2.5)
    b.axvline(20.5, color="#bbbbbb", lw=0.8, ls=":")
    b.text(20.7, 0.3, "early | late window", fontsize=6.5, color="#777777", ha="center")
    b.set_xlabel("Period")
    b.set_ylabel("Dynamic − static\nper-period RMSE (×10$^{-3}$)", fontsize=8.5)
    b.set_title("(b) By period (post hoc, descriptive)", fontsize=8.5, loc="left")
    b.set_xlim(0.5, 30.5)
    fig.tight_layout()
    _save(fig, directory, "Fig2_contrasts")


def resources(directory, rows):
    reserves = sorted({r["initial_reserve"] for r in rows})
    fig, axes = plt.subplots(2, len(reserves), figsize=(4.72, 3.4), sharex=True)
    for j, r0 in enumerate(reserves):
        for policy, color, style, label in (
            ("static", STATIC_COLOR, "-", "Static"),
            ("dynamic", DYNAMIC_COLOR, "--", "Dynamic"),
        ):
            series = [r for r in rows if r["initial_reserve"] == r0 and r["policy"] == policy]
            t = [r["period"] for r in series]
            line = dict(color=color, ls=style, lw=1.6, label=label)
            axes[0, j].plot(t, [r["reserve_left"] for r in series], **line)
            axes[1, j].plot(t, [r["cumulative_calibration"] for r in series], **line)
        axes[0, j].set_title(f"$R_0={r0}$", fontsize=8.5)
        axes[0, j].set_ylim(0, r0 * 1.05)
        axes[1, j].set_xlabel("Period")
    axes[0, 0].set_ylabel("Mean reserve left")
    axes[1, 0].set_ylabel("Cumulative calibration\nresponses")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    _save(fig, directory, "Fig3_resources")


def make_all(results_dir, figures_dir):
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    analysis = json.loads((results_dir / "analysis.json").read_text())
    by_period = json.loads((results_dir / "period_difference.json").read_text())
    rows = json.loads((results_dir / "resources.json").read_text())
    replacement_cycle(figures_dir)
    contrasts(figures_dir, analysis, by_period)
    resources(figures_dir, rows)
