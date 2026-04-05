# computer_vision_A4_Feature_Extraction
# Computer Vision Assignment

A Python-based computer vision project covering edge detection, corner detection,
line/circle detection, active contours (snakes), and interest point matching with ORB.

---

## Project Structure

```
cv_assignment/
├── images/
│   ├── original/               ← source photos
│   │   ├── photo.jpg
│   │   └── photo_perspective.jpg
│   ├── edge_detection/         ← Task 1 outputs
│   ├── corners/                ← Task 2 outputs
│   ├── lines_circles/          ← Task 3 outputs
│   ├── active_contour/         ← Task 4 outputs
│   └── interest_points/        ← Task 5 outputs
└── src/
    ├── task1_edge_detection.py
    ├── task2_corner_detection.py
    ├── task3_lines_circles.py
    ├── task4_active_contour.py
    └── task5_interest_points.py
```

---

## Requirements

```bash
pip install opencv-python scikit-image numpy
```

Python 3.9+ is recommended.

---

## How to Run

**All scripts must be run from the project root (`cv_assignment/`).**

```bash
cd cv_assignment
python src/task1_edge_detection.py
python src/task2_corner_detection.py
python src/task3_lines_circles.py
python src/task4_active_contour.py   # interactive (needs a display)
python src/task4_active_contour.py --auto   # headless / server mode
python src/task5_interest_points.py
```

---

## Task Details

### Task 1 – Edge Detection (`task1_edge_detection.py`)

Three methods are compared side-by-side:

| Method | Core operation | Weakness |
|---|---|---|
| **Sobel** | First-order gradient magnitude `√(Gx²+Gy²)` | Thick edges, sensitive to noise |
| **Laplacian of Gaussian (LoG)** | Second derivative after Gaussian blur | Amplifies noise; blurry edges |
| **Canny** ✅ *chosen* | Gaussian blur → gradient → NMS → hysteresis | Needs two threshold parameters |

**Why Canny is best:** It applies Gaussian blur to suppress noise, then non-maximum
suppression (NMS) to thin edges to exactly 1 pixel, then hysteresis thresholding to keep
only real edges (connected strong+weak pixels). This gives the cleanest, thinnest contours
without noise.

Output files:
- `sobel.jpg`, `laplacian_of_gaussian.jpg`, `canny.jpg`, `comparison.jpg`

---

### Task 2 – Corner Detection (`task2_corner_detection.py`)

| Method | Score function | Result |
|---|---|---|
| **Harris** | `R = det(M) − k·trace(M)²` | Very sensitive; detects many corner pixels |
| **Shi-Tomasi** ✅ *preferred* | `R = min(λ₁, λ₂)` | Cleaner, well-separated corners |

Harris marks pixels red where the response exceeds 1 % of its global maximum.  
Shi-Tomasi returns up to 120 well-separated corners drawn as green circles.

Output files: `harris.jpg`, `shi_tomasi.jpg`, `comparison.jpg`

---

### Task 3 – Line & Circle Detection (`task3_lines_circles.py`)

All methods operate on Canny edges computed from the image.

| Method | What it detects | Output |
|---|---|---|
| **Standard Hough Lines** | Infinite lines in (ρ, θ) space | `hough_lines_standard.jpg` |
| **Probabilistic Hough Lines** | Finite line segments | `hough_lines_probabilistic.jpg` |
| **Hough Circles** | Circles of variable radius | `hough_circles.jpg` |
| **fitEllipse on contours** | Ellipses fitted to edge contours | `ellipses.jpg` |

Output files: above 4 + `comparison.jpg`

---

### Task 4 – Active Contour / Snakes (`task4_active_contour.py`)

Implements the snake algorithm using `skimage.segmentation.active_contour`.

**Interactive mode** (requires a display):
1. Run `python src/task4_active_contour.py`
2. **Left-click** around the object you want to segment (place ≥ 3 seed points).
3. Press **ENTER** to evolve the snake.
4. Press **R** to reset and try again.
5. Press **Q** (or close window) to quit and save.

**Auto mode** (headless / server):
```bash
python src/task4_active_contour.py --auto
```
Places a circular contour automatically near the largest circle in the scene.

Snake parameters:
- `alpha=0.015` – elasticity (resists stretching)
- `beta=10` – rigidity (resists bending)
- `w_edge=1` – attracted to image gradients (edges)

Output files: `snake_interactive.jpg` or `snake_auto.jpg`
- **Yellow** = initial contour placed by user/auto
- **Green** = final snake result

---

### Task 5 – Interest Point Detection & Matching (`task5_interest_points.py`)

Uses **ORB** (Oriented FAST + Rotated BRIEF):
- Rotation-invariant and scale-invariant via an image pyramid.
- Binary descriptor → matched with **Hamming distance** (fast).
- Free of patent restrictions (unlike SIFT/SURF).

Pipeline:
1. Detect keypoints and compute descriptors in both images.
2. **kNN matching** (k=2) with a **Brute-Force** matcher.
3. **Lowe's ratio test** (0.75) filters out ambiguous matches.
4. **RANSAC homography** removes geometric outliers.
5. Image 1 is warped into image 2's perspective using the estimated homography.

Output files:
- `keypoints_img1.jpg`, `keypoints_img2.jpg` – detected keypoints (with scale/orientation)
- `matches.jpg` – all good matches (top 60)
- `inlier_matches.jpg` – geometrically verified matches after RANSAC
- `warped_img1_to_img2.jpg` – image 1 re-projected into image 2's viewpoint

---

## Implementation Notes

- **No "black-box" final functions** are used: e.g., instead of a hypothetical
  `detect_all_edges(img)`, the code manually calls `cv2.GaussianBlur`, `cv2.Sobel`,
  `cv2.Canny`, etc.
- All output directories are created automatically by the scripts.
- Scripts are self-contained and import only standard libraries + opencv/skimage/numpy.
