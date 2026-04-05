"""
Task 3 – Line and Circle / Ellipse Detection
============================================
  1. Standard Hough Line Transform
  2. Probabilistic Hough Line Transform  (more efficient, returns segments)
  3. Hough Circle Transform
  4. Ellipse detection via contour fitting  (fitEllipse)

Run from the project root:
    python src/task3_lines_circles.py
Outputs saved to:  images/lines_circles/
"""

import cv2
import numpy as np
import os

SRC    = "images/original/photo.jpg"
OUTDIR = "images/lines_circles"
os.makedirs(OUTDIR, exist_ok=True)

img  = cv2.imread(SRC)
assert img is not None, f"Cannot read {SRC}"
gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.4)
edges   = cv2.Canny(blurred, 50, 150)

# ── 1. Standard Hough Lines ───────────────────────────────────────────────────
# Accumulates votes in (ρ, θ) space; each edge pixel votes for all
# lines passing through it.  A peak in the accumulator = a line.
lines_std = cv2.HoughLines(
    edges,
    rho=1,                   # ρ resolution in pixels
    theta=np.pi / 180,       # θ resolution (1 degree)
    threshold=130            # minimum accumulator votes to accept a line
)

std_out = img.copy()
if lines_std is not None:
    for rho, theta in lines_std[:, 0]:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x0, y0 = cos_t * rho, sin_t * rho
        pt1 = (int(x0 + 1000 * (-sin_t)), int(y0 + 1000 * cos_t))
        pt2 = (int(x0 - 1000 * (-sin_t)), int(y0 - 1000 * cos_t))
        cv2.line(std_out, pt1, pt2, (0, 0, 220), 2)
    print(f"[1/5] Standard Hough: {len(lines_std)} lines  →  {OUTDIR}/hough_lines_standard.jpg")

cv2.imwrite(f"{OUTDIR}/hough_lines_standard.jpg", std_out)

# ── 2. Probabilistic Hough Lines ─────────────────────────────────────────────
# Uses a random subset of edge pixels → faster.  Returns line SEGMENTS.
plines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=80,
    minLineLength=55,    # ignore segments shorter than this
    maxLineGap=10        # bridge gaps shorter than this
)

prob_out = img.copy()
if plines is not None:
    for x1, y1, x2, y2 in plines[:, 0]:
        cv2.line(prob_out, (x1, y1), (x2, y2), (255, 80, 0), 2)
    print(f"[2/5] Probabilistic Hough: {len(plines)} segments  →  {OUTDIR}/hough_lines_probabilistic.jpg")

cv2.imwrite(f"{OUTDIR}/hough_lines_probabilistic.jpg", prob_out)

# ── 3. Hough Circle Transform ─────────────────────────────────────────────────
# For each edge pixel, votes in a 3-D (cx, cy, r) accumulator.
circles = cv2.HoughCircles(
    blurred,
    method=cv2.HOUGH_GRADIENT,
    dp=1,           # inverse accumulator resolution ratio
    minDist=55,     # minimum distance between detected circle centres
    param1=80,      # upper Canny threshold (used internally)
    param2=28,      # accumulator threshold – lower = more circles (more FP)
    minRadius=18,
    maxRadius=180
)

circ_out = img.copy()
n_circles = 0
if circles is not None:
    circles_int = np.uint16(np.around(circles))
    n_circles   = len(circles_int[0])
    for cx, cy, r in circles_int[0]:
        cv2.circle(circ_out, (cx, cy), r,  (0, 220, 0), 3)   # circle ring
        cv2.circle(circ_out, (cx, cy), 4,  (0, 0, 255), -1)  # centre dot

print(f"[3/5] Hough Circles: {n_circles} circles  →  {OUTDIR}/hough_circles.jpg")
cv2.imwrite(f"{OUTDIR}/hough_circles.jpg", circ_out)

# ── 4. Ellipse fitting via contour analysis ───────────────────────────────────
# findContours extracts connected components from the edge map.
# fitEllipse fits the minimum bounding ellipse to each contour (needs ≥5 pts).
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

ellipse_out = img.copy()
n_ell = 0
for cnt in contours:
    if len(cnt) < 5:
        continue                         # fitEllipse requires at least 5 points
    area = cv2.contourArea(cnt)
    if area < 400:
        continue                         # skip tiny noise contours

    ellipse = cv2.fitEllipse(cnt)
    (ex, ey), (ma, mi), angle = ellipse
    if mi < 1:
        continue                         # degenerate ellipse
    aspect = ma / mi
    if aspect < 4.0:                     # ignore very elongated fits (likely lines)
        cv2.ellipse(ellipse_out, ellipse, (0, 200, 255), 2)
        n_ell += 1

print(f"[4/5] Ellipse fitting: {n_ell} ellipses  →  {OUTDIR}/ellipses.jpg")
cv2.imwrite(f"{OUTDIR}/ellipses.jpg", ellipse_out)

# ── 5. Comparison grid ────────────────────────────────────────────────────────
def label(bgr, text):
    out = bgr.copy()
    cv2.putText(out, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.78, (255, 255, 0), 2, cv2.LINE_AA)
    return out

top    = np.hstack([label(std_out,    "Standard Hough Lines"),
                    label(prob_out,   "Probabilistic Hough Lines")])
bottom = np.hstack([label(circ_out,  "Hough Circles"),
                    label(ellipse_out,"Ellipse via fitEllipse")])
comparison = np.vstack([top, bottom])
cv2.imwrite(f"{OUTDIR}/comparison.jpg", comparison)
print(f"[5/5] Saved {OUTDIR}/comparison.jpg")
