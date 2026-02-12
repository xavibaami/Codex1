#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd

# ----------------------------
# User settings (edit if needed)
# ----------------------------
IMAGE_PATH = "200nm.jpg"  # input JPG/PNG file
OUTPUT_XLSX = "Rings_trans_from_graph.xlsx"  # output Excel file

# Axis numeric ranges (edit if needed)
X_MIN_VAL = 300.0
X_MAX_VAL = 2500.0
Y_MIN_VAL = 0.0
Y_MAX_VAL = 1.0

STEP_NM = 5.0  # spacing in nm


# ----------------------------
# Click-based calibration
# ----------------------------
clicked_points = []


def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Clicked: ({x}, {y})")


def get_calibration_points(img):
    """
    Ask user to click 4 points:
      1) point on x-axis at x_min  -> (x_min_pix, y_xaxis_pix)
      2) point on x-axis at x_max  -> (x_max_pix, y_xaxis_pix)
      3) point on y-axis at y_min  -> (x_yaxis_pix, y_min_pix)
      4) point on y-axis at y_max  -> (x_yaxis_pix, y_max_pix)
    """
    global clicked_points
    clicked_points = []

    display = img.copy()
    cv2.namedWindow("Click calibration points", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click calibration points", mouse_callback)

    print("\nINSTRUCTIONS:")
    print("Click 4 points in this order:")
    print(f"  1) On the x-axis at {X_MIN_VAL:g} (x-min)")
    print(f"  2) On the x-axis at {X_MAX_VAL:g} (x-max)")
    print(f"  3) On the y-axis at {Y_MIN_VAL:g} (y-min)")
    print(f"  4) On the y-axis at {Y_MAX_VAL:g} (y-max)")
    print("Then press any key in the image window.\n")

    while True:
        temp = display.copy()
        for (cx, cy) in clicked_points:
            cv2.circle(temp, (cx, cy), 6, (0, 0, 255), -1)
        cv2.imshow("Click calibration points", temp)

        key = cv2.waitKey(20) & 0xFF
        if key != 255 and len(clicked_points) >= 4:
            break

    cv2.destroyWindow("Click calibration points")
    return clicked_points[:4]


# ----------------------------
# Curve extraction
# ----------------------------
def extract_curve_points(img_bgr, plot_rect):
    """
    Extract curve pixels from the plot rectangle using thresholding.
    Assumes curve is darker than background.

    plot_rect: (x0, y0, x1, y1) inclusive bounds where (x0,y0) is top-left,
               (x1,y1) is bottom-right.
    Returns arrays of (x_pix, y_pix) in original image coordinates.
    """
    x0, y0, x1, y1 = plot_rect
    roi = img_bgr[y0 : y1 + 1, x0 : x1 + 1].copy()

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    ys = []
    xs = []
    h, w = bw.shape
    _ = h  # kept for readability

    for col in range(w):
        rows = np.where(bw[:, col] > 0)[0]
        if rows.size > 0:
            y_med = int(np.median(rows))
            xs.append(col + x0)
            ys.append(y_med + y0)

    return np.array(xs), np.array(ys), bw


# ----------------------------
# Pixel -> data mapping
# ----------------------------
def build_pixel_to_data_mapping(p1, p2, p3, p4):
    """
    p1: x-axis at x_min
    p2: x-axis at x_max
    p3: y-axis at y_min
    p4: y-axis at y_max
    """
    (x_min_pix, y_xaxis_1) = p1
    (x_max_pix, y_xaxis_2) = p2
    (x_yaxis_1, y_min_pix) = p3
    (x_yaxis_2, y_max_pix) = p4

    y_xaxis_pix = int(round((y_xaxis_1 + y_xaxis_2) / 2.0))
    x_yaxis_pix = int(round((x_yaxis_1 + x_yaxis_2) / 2.0))

    x0 = min(x_yaxis_pix, x_min_pix, x_max_pix)
    x1 = max(x_yaxis_pix, x_min_pix, x_max_pix)
    y0 = min(y_max_pix, y_min_pix, y_xaxis_pix)
    y1 = max(y_max_pix, y_min_pix, y_xaxis_pix)
    plot_rect = (x0, y0, x1, y1)

    if x_max_pix == x_min_pix:
        raise ValueError("Invalid x calibration: x-min and x-max pixels are identical.")
    if y_min_pix == y_max_pix:
        raise ValueError("Invalid y calibration: y-min and y-max pixels are identical.")

    def xpix_to_wl(xpix):
        return X_MIN_VAL + (xpix - x_min_pix) * (X_MAX_VAL - X_MIN_VAL) / (x_max_pix - x_min_pix)

    def ypix_to_t(ypix):
        return Y_MIN_VAL + (y_min_pix - ypix) * (Y_MAX_VAL - Y_MIN_VAL) / (y_min_pix - y_max_pix)

    return plot_rect, xpix_to_wl, ypix_to_t


def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    p1, p2, p3, p4 = get_calibration_points(img)
    plot_rect, xpix_to_wl, ypix_to_t = build_pixel_to_data_mapping(p1, p2, p3, p4)

    xs_pix, ys_pix, _ = extract_curve_points(img, plot_rect)
    if xs_pix.size < 10:
        raise RuntimeError(
            "Too few curve points detected. Curve may be faint or close to background/grid."
        )

    wl = np.array([xpix_to_wl(x) for x in xs_pix], dtype=float)
    tr = np.array([ypix_to_t(y) for y in ys_pix], dtype=float)

    mask = (wl >= X_MIN_VAL) & (wl <= X_MAX_VAL) & (tr >= -0.2) & (tr <= 1.2)
    wl = wl[mask]
    tr = tr[mask]

    df_raw = pd.DataFrame({"Wavelength_nm": wl, "Transmission": tr})
    df_raw = df_raw.sort_values("Wavelength_nm")
    df_raw["Wavelength_nm_round"] = df_raw["Wavelength_nm"].round(2)
    df_raw = df_raw.groupby("Wavelength_nm_round", as_index=False)["Transmission"].median()
    df_raw = df_raw.rename(columns={"Wavelength_nm_round": "Wavelength_nm"})

    if df_raw.empty:
        raise RuntimeError("No valid points remained after filtering.")

    wl_grid = np.arange(X_MIN_VAL, X_MAX_VAL + 1e-9, STEP_NM)
    tr_grid = np.interp(wl_grid, df_raw["Wavelength_nm"].values, df_raw["Transmission"].values)
    tr_grid = np.clip(tr_grid, 0.0, 1.0)

    df_out = pd.DataFrame({"Wavelength_nm": wl_grid, "Transmission": tr_grid})

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="T_vs_WL_5nm")
        df_raw.to_excel(writer, index=False, sheet_name="Extracted_raw")

    print(f"\nSaved: {OUTPUT_XLSX}")
    print("Sheets:")
    print("  - T_vs_WL_5nm: resampled every 5 nm")
    print("  - Extracted_raw: extracted points before resampling")


if __name__ == "__main__":
    main()
