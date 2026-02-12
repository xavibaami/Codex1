# Radial Distribution Function

Compute the radial distribution function `g(r)` from a microscopy image by:
- segmenting particles,
- extracting particle centroids,
- applying translation edge correction,
- exporting `r` (nm) and `g(r)` to Excel.

## Files
- `radial_distribution_function.py`: main script
- `requirements.txt`: Python dependencies

## What The Script Does
1. Loads an input image (`.tif`, `.jpg`, `.png`).
2. Segments particles using Otsu thresholding.
3. Removes small objects (`MIN_PARTICLE_AREA_PX`).
4. Computes centroid positions for all detected particles.
5. Calculates pair distances and translation-corrected RDF.
6. Plots `g(r)` vs `r (nm)`.
7. Exports results to an `.xlsx` file.

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
From the `Radial Distribution Function` folder:
```bash
python radial_distribution_function.py
```

Before running, edit user inputs at the top of the script:
- `IMAGE_PATH`
- `NM_PER_PIXEL`
- `MIN_PARTICLE_AREA_PX`
- `OUTPUT_XLSX`

## Parameter Notes
- `NM_PER_PIXEL`: conversion factor from pixels to nm.
- `MIN_PARTICLE_AREA_PX`: filters noise/speckles; increase if false particles appear.
- `DR_PX`: radial bin width in pixels. Smaller values give more detail but noisier curves.
- `R_MAX_FRACTION`: max analyzed radius as fraction of image size (helps reduce edge bias).

## Tips For Better Results
- If particles are dark, change:
```python
binary = img < th
```
instead of `img > th`.
- Use images with clear contrast and minimal illumination gradients.
- Validate detection quality by checking `Particles detected: N` and the RDF plot shape.

## Output
- Excel file (`OUTPUT_XLSX`) with:
  - `r_nm`: radial distance in nanometers
  - `g_r`: radial distribution function values

## Common Issues
- `Not enough particles detected.`:
  - improve segmentation contrast
  - reduce `MIN_PARTICLE_AREA_PX`
  - verify threshold polarity (`img > th` vs `img < th`)
- Very noisy `g(r)`:
  - increase `DR_PX`
  - use a cleaner image / better segmentation
