from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = DATA_DIR / "tat1.json"
DEFAULT_EVENT_DATE = date(1958, 2, 11)
UPDATED_DIR = DATA_DIR / "updated"
MIN_DIR = DATA_DIR / "1min"
SEC_DIR = DATA_DIR / "1sec"
IAGA_MIN_DIR = DATA_DIR / "IAGA_min"
IAGA_SEC_DIR = DATA_DIR / "IAGA_sec"


@dataclass(frozen=True)
class PanelSpec:
    name: str
    csv_path: Path
    station_name: str
    iaga_code: str
    source: str
    latitude: float
    longitude: float
    elevation: float


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def derive_code(name: str) -> str:
    chars = [ch for ch in name.upper() if ch.isalnum()]
    code = "".join(chars)[:3]
    return code.ljust(3, "X")


def load_config(config_path: Path) -> dict:
    config_path = config_path.resolve()
    with config_path.open() as fp:
        config = json.load(fp)

    config["event_date"] = date.fromisoformat(config.get("event_date", DEFAULT_EVENT_DATE.isoformat()))
    config["source"] = config.get("source", "Digitized TAT1 figure")
    for panel in config["panels"]:
        panel["csv"] = resolve_path(panel["csv"], config_path.parent)
    return config


def load_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None or {"Date", "H"} - set(reader.fieldnames):
            raise ValueError(f"{path.name} must contain Date and H columns.")
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


def collapse_duplicates(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        raise ValueError("Empty series.")

    unique_x = []
    unique_y = []
    current_x = x[0]
    bucket: list[float] = [float(y[0])]

    for xi, yi in zip(x[1:], y[1:], strict=False):
        if np.isclose(xi, current_x):
            bucket.append(float(yi))
        else:
            unique_x.append(float(current_x))
            unique_y.append(float(np.nanmean(bucket)))
            current_x = float(xi)
            bucket = [float(yi)]

    unique_x.append(float(current_x))
    unique_y.append(float(np.nanmean(bucket)))
    return np.asarray(unique_x, dtype=float), np.asarray(unique_y, dtype=float)


def scale_to_day(x: np.ndarray) -> np.ndarray:
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if xmax == xmin:
        raise ValueError("Cannot scale a zero-width time axis.")
    return (x - xmin) / (xmax - xmin) * 24.0


def regular_grid(step_hours: float) -> np.ndarray:
    if step_hours <= 0:
        raise ValueError("Interpolation step must be positive.")
    return np.arange(0.0, 24.0, step_hours, dtype=float)


def interpolate_series(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, x, y, left=y[0], right=y[-1])


def to_datetime(event_day: date, hour_value: float) -> datetime:
    return datetime.combine(event_day, datetime.min.time()) + timedelta(hours=float(hour_value))


def format_header(label: str, value: str) -> str:
    return f" {label:<23}{value:<45}|"


def format_comment(text: str) -> str:
    return f" # {text}"[:69].ljust(69) + "|"


def load_panel_specs(config: dict) -> list[PanelSpec]:
    panels = []
    for panel in config["panels"]:
        panels.append(
            PanelSpec(
                name=panel["name"],
                csv_path=panel["csv"],
                station_name=panel.get("station_name", panel["name"]),
                iaga_code=panel.get("iaga_code", derive_code(panel["name"])),
                source=panel.get("source", config["source"]),
                latitude=float(panel.get("latitude", 0.0)),
                longitude=float(panel.get("longitude", 0.0)),
                elevation=float(panel.get("elevation", 0.0)),
            )
        )
    return panels


def write_scaled_csv(path: Path, x_scaled: np.ndarray, y: np.ndarray) -> None:
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Date", "H"])
        for xi, yi in zip(x_scaled, y, strict=False):
            writer.writerow([f"{float(xi):.12f}", f"{float(yi):.12f}"])


def write_regular_csv(path: Path, x_hours: np.ndarray, y: np.ndarray) -> None:
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["Date", "H"])
        for xi, yi in zip(x_hours, y, strict=False):
            writer.writerow([f"{float(xi):.12f}", f"{float(yi):.12f}"])


def write_iaga2002_file(
    path: Path,
    panel: PanelSpec,
    event_day: date,
    cadence_label: str,
    digital_sampling: str,
    data_interval_type: str,
    x_hours: np.ndarray,
    y: np.ndarray,
) -> None:
    with path.open("w", newline="\n") as fp:
        fp.write(format_header("Format", "IAGA-2002") + "\n")
        fp.write(format_header("Source of Data", panel.source) + "\n")
        fp.write(format_header("Station Name", panel.station_name) + "\n")
        fp.write(format_header("IAGA Code", panel.iaga_code) + "\n")
        fp.write(format_header("Geodetic Latitude", f"{panel.latitude:.3f}") + "\n")
        fp.write(format_header("Geodetic Longitude", f"{panel.longitude:.3f}") + "\n")
        fp.write(format_header("Elevation", f"{panel.elevation:.0f}") + "\n")
        fp.write(format_header("Reported", "H") + "\n")
        fp.write(format_header("Sensor Orientation", "H") + "\n")
        fp.write(format_header("Digital Sampling", digital_sampling) + "\n")
        fp.write(format_header("Data Interval Type", data_interval_type) + "\n")
        fp.write(format_header("Data Type", "Variation") + "\n")
        fp.write(format_comment("Generated from digitized H component") + "\n")
        fp.write(format_comment("Scaled to 0-24 hours before interpolation") + "\n")
        fp.write(format_comment(f"Cadence: {cadence_label}") + "\n")
        fp.write(" DATE       TIME         DOY     H".ljust(69) + "|\n")
        for xi, yi in zip(x_hours, y, strict=False):
            dt = to_datetime(event_day, float(xi))
            doy = dt.timetuple().tm_yday
            line = f" {dt:%Y-%m-%d %H:%M:%S.%f} {doy:03d} {float(yi):10.2f}"
            fp.write(line[:69].ljust(69) + "|\n")


def main() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    panels = load_panel_specs(config)
    event_day: date = config["event_date"]

    for directory in (UPDATED_DIR, MIN_DIR, SEC_DIR, IAGA_MIN_DIR, IAGA_SEC_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    for panel in panels:
        x_raw, y_raw = load_trace(panel.csv_path)
        x_unique, y_unique = collapse_duplicates(x_raw, y_raw)
        x_scaled = scale_to_day(x_unique)

        raw_out = UPDATED_DIR / f"{panel.csv_path.stem}_updated.csv"
        write_scaled_csv(raw_out, x_scaled, y_unique)

        minute_grid = regular_grid(1.0 / 60.0)
        second_grid = regular_grid(1.0 / 3600.0)
        y_minute = interpolate_series(x_scaled, y_unique, minute_grid)
        y_second = interpolate_series(x_scaled, y_unique, second_grid)

        minute_csv = MIN_DIR / f"{panel.csv_path.stem}_1min.csv"
        second_csv = SEC_DIR / f"{panel.csv_path.stem}_1sec.csv"
        write_regular_csv(minute_csv, minute_grid, y_minute)
        write_regular_csv(second_csv, second_grid, y_second)

        iaga_base = f"{panel.iaga_code.lower()}{event_day:%Y%m%d}v"
        minute_iaga = IAGA_MIN_DIR / f"{iaga_base}min.min"
        second_iaga = IAGA_SEC_DIR / f"{iaga_base}sec.sec"
        write_iaga2002_file(
            minute_iaga,
            panel,
            event_day,
            "1-minute",
            "60 seconds",
            "1-minute instantaneous",
            minute_grid,
            y_minute,
        )
        write_iaga2002_file(
            second_iaga,
            panel,
            event_day,
            "1-second",
            "1 seconds",
            "1-second instantaneous",
            second_grid,
            y_second,
        )

        print(raw_out)
        print(minute_csv)
        print(second_csv)
        print(minute_iaga)
        print(second_iaga)


if __name__ == "__main__":
    main()
