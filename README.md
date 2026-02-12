# Graph to Excel (Python)

Convert points from a graph image (`.png`, `.jpg`) into an Excel file (`.xlsx`).

## Features
- Manual mode: click data points on the graph
- Auto mode: detect a colored line from the graph
- Axis calibration: map pixel positions to real chart values
- OpenCV mode: threshold-based extraction from dark curves

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Matplotlib manual mode:
```bash
python graph_to_excel.py path/to/graph.png output.xlsx --mode manual
```

Matplotlib auto mode (for red line by default):
```bash
python graph_to_excel.py path/to/graph.png output.xlsx --mode auto --line-color "#FF0000" --tolerance 35
```

OpenCV mode (based on your provided workflow):
```bash
python graph_to_excel_cv2.py
```
Then edit constants at the top of `graph_to_excel_cv2.py`:
- `IMAGE_PATH`
- `OUTPUT_XLSX`
- `X_MIN_VAL`, `X_MAX_VAL`, `Y_MIN_VAL`, `Y_MAX_VAL`
- `STEP_NM`

## Calibration flow
When the image opens, click in this exact order:
1. X-axis minimum point
2. X-axis maximum point
3. Y-axis minimum point
4. Y-axis maximum point

Then enter the real axis values in the terminal.

## Notes
- Auto mode works best when the line color is clearly distinct from grid/background.
- If auto mode finds no points, increase `--tolerance` or set a different `--line-color`.
