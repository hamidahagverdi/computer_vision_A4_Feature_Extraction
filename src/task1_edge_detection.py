"""
Task 1 – Edge Detection
=======================
Compares three methods:
  1. Sobel        – first-order gradient magnitude
  2. Laplacian of Gaussian (LoG) – second-order derivative after smoothing
  3. Canny        – the chosen best result (non-max suppression + hysteresis)

Run from the project root:
    python src/task1_edge_detection.py
Outputs saved to:  images/edge_detection/
"""

import cv2
import numpy as np
import os

SRC    = "images/original/photo.jpg"
OUTDIR = "images/edge_detection"
os.makedirs(OUTDIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
img  = cv2.imread(SRC)
assert img is not None, f"Cannot read {SRC}"
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ── 1. Sobel (gradient magnitude) ────────────────────────────────────────────
# Compute partial derivatives with a 3×3 Sobel kernel
gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # ∂I/∂x
gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # ∂I/∂y
magnitude = np.sqrt(gx**2 + gy**2)                 # ||∇I||
sobel_out = np.uint8(np.clip(magnitude, 0, 255))
cv2.imwrite(f"{OUTDIR}/sobel.jpg", sobel_out)
print(f"[1/4] Saved {OUTDIR}/sobel.jpg")

# ── 2. Laplacian of Gaussian (LoG) ───────────────────────────────────────────
# Gaussian blur first (sigma=1.5) to suppress noise, then Laplacian (∇²I)
blurred_log = cv2.GaussianBlur(gray, (9, 9), sigmaX=1.5)
lap         = cv2.Laplacian(blurred_log, cv2.CV_64F, ksize=3)
log_out     = np.uint8(np.clip(np.abs(lap), 0, 255))
cv2.imwrite(f"{OUTDIR}/laplacian_of_gaussian.jpg", log_out)
print(f"[2/4] Saved {OUTDIR}/laplacian_of_gaussian.jpg")

# ── 3. Canny (best result) ────────────────────────────────────────────────────
# Auto-select thresholds: median-based rule (robust to image brightness)
median_val  = float(np.median(gray))
low_thresh  = max(0,   int(0.66 * median_val))
high_thresh = min(255, int(1.33 * median_val))
blurred_canny = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.4)
canny_out     = cv2.Canny(blurred_canny, low_thresh, high_thresh)
cv2.imwrite(f"{OUTDIR}/canny.jpg", canny_out)
print(f"[3/4] Saved {OUTDIR}/canny.jpg  "
      f"(thresholds: low={low_thresh}, high={high_thresh})")

# ── Side-by-side comparison ───────────────────────────────────────────────────
def to_bgr_labeled(gray_img, text):
    out = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(out, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (0, 200, 255), 2, cv2.LINE_AA)
    return out

comparison = np.hstack([
    to_bgr_labeled(sobel_out,  "1) Sobel"),
    to_bgr_labeled(log_out,    "2) LoG"),
    to_bgr_labeled(canny_out,  "3) Canny [best]"),
])
cv2.imwrite(f"{OUTDIR}/comparison.jpg", comparison)
print(f"[4/4] Saved {OUTDIR}/comparison.jpg")

print("""
Why Canny is the best choice
-----------------------------
• Gaussian pre-blur removes high-frequency noise before differentiation.
• Non-maximum suppression keeps edges exactly ONE pixel wide.
• Hysteresis thresholding (two thresholds) joins weak edge pixels to
  strong ones, so real contours are complete and isolated noise is dropped.
Sobel keeps thin but is noisy; LoG is noisier due to the second derivative.
""")
