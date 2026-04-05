"""
Task 2 – Corner Detection
=========================
Two methods compared:
  1. Harris Corner Detector  – classic eigenvalue-based corner response
  2. Shi-Tomasi              – improved corner score (min eigenvalue)

Run from the project root:
    python src/task2_corner_detection.py
Outputs saved to:  images/corners/
"""

import cv2
import numpy as np
import os

SRC    = "images/original/photo.jpg"
OUTDIR = "images/corners"
os.makedirs(OUTDIR, exist_ok=True)

img  = cv2.imread(SRC)
assert img is not None, f"Cannot read {SRC}"
gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_f32   = np.float32(gray)

# ── 1. Harris Corner Detector ─────────────────────────────────────────────────
# cornerHarris computes:
#   R = det(M) − k · trace(M)²     where M is the structure tensor
# R > 0  →  corner  |  R < 0  →  edge  |  |R| ≈ 0  →  flat
harris_resp = cv2.cornerHarris(
    gray_f32,
    blockSize=4,     # size of neighbourhood for M computation
    ksize=3,         # Sobel aperture size
    k=0.04           # Harris free parameter (empirical range 0.04–0.06)
)

# Dilate response to make local maxima more visible
harris_resp = cv2.dilate(harris_resp, None)

harris_out = img.copy()
# Threshold: keep pixels whose response exceeds 1 % of global max
threshold = 0.01 * harris_resp.max()
harris_out[harris_resp > threshold] = [0, 0, 255]   # red pixels

n_harris = int((harris_resp > threshold).sum())
cv2.imwrite(f"{OUTDIR}/harris.jpg", harris_out)
print(f"[1/3] Saved {OUTDIR}/harris.jpg  ({n_harris} corner pixels)")

# ── 2. Shi-Tomasi (Good Features To Track) ────────────────────────────────────
# Scores each candidate corner by its MINIMUM eigenvalue of M.
# More numerically stable than Harris; preferred for tracking pipelines.
corners = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=120,     # return at most this many corners
    qualityLevel=0.01,  # fraction of best corner quality; lower = more corners
    minDistance=12,     # minimum pixel distance between returned corners
    blockSize=5
)

shi_out = img.copy()
n_shi = 0
if corners is not None:
    corners = np.intp(corners)
    n_shi   = len(corners)
    for c in corners:
        x, y = c.ravel()
        cv2.circle(shi_out, (x, y), 6, (0, 255, 0), 2)    # green ring
        cv2.circle(shi_out, (x, y), 2, (0, 255, 0), -1)   # green dot

cv2.imwrite(f"{OUTDIR}/shi_tomasi.jpg", shi_out)
print(f"[2/3] Saved {OUTDIR}/shi_tomasi.jpg  ({n_shi} corners)")

# ── Comparison ────────────────────────────────────────────────────────────────
def label(bgr, text):
    out = bgr.copy()
    cv2.putText(out, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (255, 255, 0), 2, cv2.LINE_AA)
    return out

comparison = np.hstack([label(harris_out, "Harris"), label(shi_out, "Shi-Tomasi")])
cv2.imwrite(f"{OUTDIR}/comparison.jpg", comparison)
print(f"[3/3] Saved {OUTDIR}/comparison.jpg")

print("""
Harris vs Shi-Tomasi
--------------------
Harris  – sensitive but can flag many redundant pixels along a strong edge.
Shi-Tomasi – scores by minimum eigenvalue, so both eigenvalues must be large
             (true corner), giving cleaner, well-separated detections.
""")
