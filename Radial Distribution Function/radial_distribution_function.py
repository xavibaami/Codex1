#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology
import pandas as pd

# ---- User inputs ----
IMAGE_PATH = "200nm_20K.tif"  # The name of your tif image (can also be jpg)
NM_PER_PIXEL = 5  # 5 for 200 / 7 for 300 / 9 for 500 / 13 for 800
MIN_PARTICLE_AREA_PX = 20
OUTPUT_XLSX = "200_rdf.xlsx"  # Edit the name

# ---- Load & segment ----
img = io.imread(IMAGE_PATH, as_gray=True)
th = filters.threshold_otsu(img)
binary = img > th  # flip to img < th if particles are dark
binary = morphology.remove_small_objects(binary, MIN_PARTICLE_AREA_PX)

label = measure.label(binary)
props = measure.regionprops(label)
centers = np.array([[p.centroid[1], p.centroid[0]] for p in props])  # (x,y)

H, W = img.shape
N = len(centers)
print(f"Particles detected: {N}")
if N < 2:
    raise RuntimeError("Not enough particles detected.")

# ---- RDF bins ----
DR_PX = 2
R_MAX_FRACTION = 0.45  # safe max to avoid extreme edge effects
r_max_px = int(R_MAX_FRACTION * min(H, W))
bins_px = np.arange(0, r_max_px + DR_PX, DR_PX)
r_px = 0.5 * (bins_px[1:] + bins_px[:-1])

# ---- Pair vectors (upper triangle only) ----
idx_i, idx_j = np.triu_indices(N, k=1)
dx = centers[idx_i, 0] - centers[idx_j, 0]
dy = centers[idx_i, 1] - centers[idx_j, 1]
r = np.sqrt(dx * dx + dy * dy)

# ---- Translation edge correction ----
# overlap area between window and shifted window
overlap = (W - np.abs(dx)) * (H - np.abs(dy))
valid = overlap > 0
r = r[valid]
overlap = overlap[valid]

# histogram weighted by 1/overlap
hist = np.zeros(len(r_px), dtype=float)
bin_idx = np.floor(r / DR_PX).astype(int)
mask = (bin_idx >= 0) & (bin_idx < len(hist))
for b, w in zip(bin_idx[mask], overlap[mask]):
    hist[b] += 1.0 / w

# ---- Normalize ----
lam = N / (W * H)  # intensity per px^2

# IMPORTANT: multiply by 2 because we only used i<j pairs
g = (2 * hist) / (2 * np.pi * r_px * DR_PX * lam**2)

# ---- Convert to nm for plotting ----
r_nm = r_px * NM_PER_PIXEL

# ---- Plot ----
plt.figure(figsize=(6, 4))
plt.plot(r_nm, g, lw=2)
plt.xlabel("r (nm)")
plt.ylabel("g(r)")
plt.title("Radial Distribution Function (Translation Corrected)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

df = pd.DataFrame({"r_nm": r_nm, "g_r": g})
df.to_excel(OUTPUT_XLSX, index=False)
print("Saved xlsx file")
