#!/usr/bin/env python3
"""Convert graph image data to Excel.

Workflow:
1. Calibrate axes by clicking known points on the graph image.
2. Capture data points manually or auto-detect a colored line.
3. Export points to an Excel file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


@dataclass
class Calibration:
    x_pixel_min: float
    x_pixel_max: float
    y_pixel_min: float
    y_pixel_max: float
    x_value_min: float
    x_value_max: float
    y_value_min: float
    y_value_max: float

    def pixel_to_data(self, px: float, py: float) -> tuple[float, float]:
        # Linear map pixel coordinates to data coordinates.
        x = self.x_value_min + (
            (px - self.x_pixel_min)
            / (self.x_pixel_max - self.x_pixel_min)
            * (self.x_value_max - self.x_value_min)
        )
        y = self.y_value_min + (
            (py - self.y_pixel_min)
            / (self.y_pixel_max - self.y_pixel_min)
            * (self.y_value_max - self.y_value_min)
        )
        return x, y


def parse_hex_color(color: str) -> tuple[int, int, int]:
    c = color.strip().lstrip("#")
    if len(c) != 6:
        raise ValueError("Color must be a 6-char hex value like #FF0000")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def collect_clicks(image: np.ndarray, title: str, n: int | None = None) -> list[tuple[float, float]]:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("on")

    points = plt.ginput(n=n, timeout=0)
    plt.close(fig)
    return points


def calibrate(image: np.ndarray) -> Calibration:
    print("Calibration step:")
    print("Click these 4 points in order:")
    print("1) X-axis minimum point")
    print("2) X-axis maximum point")
    print("3) Y-axis minimum point")
    print("4) Y-axis maximum point")

    pts = collect_clicks(
        image,
        "Click: X-min, X-max, Y-min, Y-max (in that order)",
        n=4,
    )

    if len(pts) != 4:
        raise RuntimeError("Calibration cancelled or not enough points clicked.")

    x_min_val = float(input("Real X-axis minimum value: ").strip())
    x_max_val = float(input("Real X-axis maximum value: ").strip())
    y_min_val = float(input("Real Y-axis minimum value: ").strip())
    y_max_val = float(input("Real Y-axis maximum value: ").strip())

    return Calibration(
        x_pixel_min=pts[0][0],
        x_pixel_max=pts[1][0],
        y_pixel_min=pts[2][1],
        y_pixel_max=pts[3][1],
        x_value_min=x_min_val,
        x_value_max=x_max_val,
        y_value_min=y_min_val,
        y_value_max=y_max_val,
    )


def collect_manual_points(image: np.ndarray) -> list[tuple[float, float]]:
    print("Manual mode:")
    print("Click as many graph points as you want.")
    print("Press Enter in the plot window when finished.")
    return collect_clicks(image, "Click graph points. Press Enter to finish.", n=None)


def collect_auto_points(
    image: np.ndarray,
    rgb_target: tuple[int, int, int],
    tolerance: int,
) -> list[tuple[float, float]]:
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Auto mode requires an RGB image.")

    rgb = image[:, :, :3].astype(np.int16)
    target = np.array(rgb_target, dtype=np.int16)

    dist = np.linalg.norm(rgb - target[None, None, :], axis=2)
    mask = dist <= tolerance

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []

    # Collapse pixels by X so we get one Y per X-column (median for stability).
    points = []
    unique_x = np.unique(xs)
    for x in unique_x:
        y_vals = ys[xs == x]
        y_med = float(np.median(y_vals))
        points.append((float(x), y_med))

    points.sort(key=lambda p: p[0])
    return points


def export_to_excel(points_data: list[tuple[float, float]], output_file: Path) -> None:
    df = pd.DataFrame(points_data, columns=["x", "y"])
    df.to_excel(output_file, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert graph image to Excel data points.")
    parser.add_argument("image", type=Path, help="Path to input graph image (png/jpg)")
    parser.add_argument("output", type=Path, help="Path to output Excel file (.xlsx)")
    parser.add_argument(
        "--mode",
        choices=["manual", "auto"],
        default="manual",
        help="manual: click points; auto: detect line by color",
    )
    parser.add_argument(
        "--line-color",
        default="#FF0000",
        help="Line color in hex for auto mode (default: #FF0000)",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=35,
        help="Color tolerance for auto mode (default: 35)",
    )
    args = parser.parse_args()

    image = np.array(Image.open(args.image).convert("RGB"))

    calibration = calibrate(image)

    if args.mode == "manual":
        pixel_points = collect_manual_points(image)
    else:
        rgb = parse_hex_color(args.line_color)
        pixel_points = collect_auto_points(image, rgb, args.tolerance)
        if not pixel_points:
            raise RuntimeError(
                "No matching line pixels found in auto mode. Try another --line-color or --tolerance."
            )

    data_points = [calibration.pixel_to_data(px, py) for px, py in pixel_points]
    export_to_excel(data_points, args.output)

    print(f"Saved {len(data_points)} points to {args.output}")


if __name__ == "__main__":
    main()
