from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path(__file__).resolve().parent
IMAGE_PATH = DATA_DIR / "fig3_source_600dpi_cropped.png"
DEFAULT_CONFIG_PATH = DATA_DIR / "tat1.json"


@dataclass(frozen=True)
class PanelSpec:
    name: str
    csv_name: str
    color: str
    x_left: float
    x_right: float
    y_center: float
    half_height: float
    bias: float = 0.0


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(config_path: Path) -> dict:
    config_path = config_path.resolve()
    with config_path.open() as fp:
        config = json.load(fp)

    config["image"] = resolve_path(config["image"], config_path.parent)
    config["output"] = resolve_path(config["output"], config_path.parent)
    for panel in config["panels"]:
        panel["csv"] = resolve_path(panel["csv"], config_path.parent)
        panel["x_left"] = float(panel["x_left"])
        panel["x_right"] = float(panel["x_right"])
    return config


def load_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x_values = []
    y_values = []
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None or {"Date", "H"} - set(reader.fieldnames):
            raise ValueError(f"{path.name} must contain 'Date' and 'H' columns.")
        for row in reader:
            x_values.append(float(row["Date"]))
            y_values.append(float(row["H"]))

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    unique_x, unique_idx = np.unique(x, return_index=True)
    return unique_x, y[unique_idx]


def robust_scale(values: np.ndarray, target_half_height: float) -> tuple[float, float]:
    center = float(np.nanmedian(values))
    q05, q95 = np.nanquantile(values, [0.05, 0.95])
    spread = max(center - q05, q95 - center, float(np.nanstd(values)), 1e-9)
    scale = 0.88 * target_half_height / spread
    return center, scale


def map_trace_to_panel(
    x: np.ndarray,
    y: np.ndarray,
    panel: PanelSpec,
) -> tuple[np.ndarray, np.ndarray]:
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if x_max == x_min:
        raise ValueError(f"{panel.name} has zero x-range.")

    x_pixels = panel.x_left + (x - x_min) / (x_max - x_min) * (panel.x_right - panel.x_left)

    center, scale = robust_scale(y, panel.half_height)
    target_center = panel.y_center + panel.bias * panel.half_height
    y_pixels = target_center - (y - center) * scale
    return x_pixels, y_pixels


def load_panel_specs(config: dict) -> list[PanelSpec]:
    return [
        PanelSpec(
            name=panel["name"],
            csv_name=str(panel["csv"]),
            color=panel["color"],
            x_left=float(panel["x_left"]),
            x_right=float(panel["x_right"]),
            y_center=float(panel["y_center"]),
            half_height=float(panel["half_height"]),
            bias=float(panel.get("bias", 0.0)),
        )
        for panel in config["panels"]
    ]


def main() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    panel_specs = load_panel_specs(config)
    image = mpimg.imread(config["image"])
    height, width = image.shape[0], image.shape[1]

    fig = plt.figure(figsize=(width / 600.0, height / 600.0), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, extent=(0, width, height, 0), origin="upper", aspect="auto")

    for panel in panel_specs:
        csv_path = Path(panel.csv_name)
        x, y = load_trace(csv_path)
        x_pixels, y_pixels = map_trace_to_panel(x, y, panel)
        ax.plot(
            x_pixels,
            y_pixels,
            color=panel.color,
            lw=1.2,
            alpha=0.92,
            zorder=3,
            solid_joinstyle="round",
            solid_capstyle="round",
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    config["output"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config["output"], dpi=600, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(config["output"])


if __name__ == "__main__":
    main()
