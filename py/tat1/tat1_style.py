"""Shared plotting style for TAT1 paper figures.

Use this helper in each TAT1 figure script so fonts, ticks, legends, and
exports stay visually consistent while individual figure sizes can differ.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_DIR / "figures" / "tat1"
PNG_DIR = FIGURE_DIR / "pngfiles"


def apply_tat1_style(font_size: int = 12) -> None:
    """Apply the shared SciencePlots-based TAT1 figure style."""
    try:
        import scienceplots  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The TAT1 figure style requires the `scienceplots` package. "
            "Install or activate the SCUBAS environment before plotting."
        ) from exc

    plt.style.use(["science", "ieee"])
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Tahoma",
        "DejaVu Sans",
        "Lucida Grande",
        "Verdana",
    ]
    plt.rcParams["text.usetex"] = False
    mpl.rcParams.update(
        {
            "axes.edgecolor": "0.20",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.labelsize": font_size + 1,
            "axes.linewidth": 0.9,
            "axes.titlesize": font_size + 2,
            "figure.dpi": 180,
            "figure.facecolor": "white",
            "figure.titlesize": font_size + 3,
            "font.size": font_size,
            "grid.alpha": 0.22,
            "grid.color": "0.70",
            "grid.linewidth": 0.45,
            "legend.edgecolor": "0.75",
            "legend.facecolor": "white",
            "legend.fontsize": max(font_size - 1, 9),
            "legend.framealpha": 0.90,
            "legend.frameon": True,
            "lines.linewidth": 1.45,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "xtick.direction": "in",
            "xtick.labelsize": font_size,
            "xtick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.direction": "in",
            "ytick.labelsize": font_size,
            "ytick.major.size": 3.5,
            "ytick.minor.size": 2.0,
        }
    )


def finish_axis(
    ax: plt.Axes,
    *,
    grid: bool = True,
    zero_line: bool = False,
    legend: bool = False,
    legend_loc: str = "best",
    legend_ncol: int = 1,
) -> None:
    """Apply common final touches to one axis."""
    if grid:
        ax.grid(True, which="major", alpha=0.22, linewidth=0.45)
        ax.grid(True, which="minor", alpha=0.10, linewidth=0.30)
    if zero_line:
        ax.axhline(0, color="0.15", linewidth=0.8, linestyle="--", alpha=0.45)
    if legend:
        leg = ax.legend(loc=legend_loc, ncol=legend_ncol, frameon=True)
        if leg is not None:
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("0.75")
            leg.get_frame().set_alpha(0.90)


def panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = 0.02,
    y: float = 0.96,
    font_size: int | None = None,
) -> None:
    """Place a consistent panel label inside an axis."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=font_size,
        fontweight="bold",
        color="0.10",
    )


def save_figure(fig: plt.Figure, name: str, *, dpi: int = 300) -> Path:
    """Save a figure as PNG under figures/tat1/pngfiles."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    output = PNG_DIR / f"{Path(name).stem}.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    return output
