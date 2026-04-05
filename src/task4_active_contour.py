"""
Task 4 – Active Contour (Snakes)
=================================
Interactive mode (when a display is available):
  • Left-click on the image to place seed points around an object.
  • Press ENTER to run the snake.
  • Press R to reset and start over.
  • Press Q (or close the window) to quit and save.

Headless / auto mode (no display, or pass --auto flag):
  • A circular initial contour is placed automatically around the
    largest circle in the scene and the snake is evolved.

Run from the project root:
    python src/task4_active_contour.py          # interactive
    python src/task4_active_contour.py --auto   # headless

Outputs saved to:  images/active_contour/
"""

import sys
import os
import cv2
import numpy as np
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.color import rgb2gray

SRC    = "images/original/photo.jpg"
OUTDIR = "images/active_contour"
os.makedirs(OUTDIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
img_bgr = cv2.imread(SRC)
assert img_bgr is not None, f"Cannot read {SRC}"
img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = rgb2gray(img_rgb)          # float64 [0, 1]  for skimage

# ── Snake helper ──────────────────────────────────────────────────────────────
def interpolate_contour(pts, n=300):
    """Convert a list of (x, y) seed pixels into a smooth closed (row, col) array."""
    pts_arr = np.array(pts, dtype=float)
    pts_arr = np.vstack([pts_arr, pts_arr[0]])        # close the loop
    # Arc-length parameterisation
    diffs = np.diff(pts_arr, axis=0)
    t = np.concatenate([[0], np.cumsum(np.hypot(diffs[:,0], diffs[:,1]))])
    t /= t[-1]
    t_new = np.linspace(0, 1, n)
    x_new = np.interp(t_new, t, pts_arr[:, 0])
    y_new = np.interp(t_new, t, pts_arr[:, 1])
    return np.column_stack([y_new, x_new])            # skimage: (row, col)

def run_snake(init_contour, sigma=3.0):
    """Run the active contour algorithm on img_gray."""
    smoothed = gaussian(img_gray, sigma=sigma)
    snake = active_contour(
        smoothed,
        init_contour,
        alpha=0.015,      # elasticity – resists stretching
        beta=10,          # rigidity   – resists bending
        gamma=0.001,      # gradient-descent step size
        w_line=0,         # line energy  (0 = not attracted to bright regions)
        w_edge=1,         # edge energy  (attracted to intensity gradients)
        max_num_iter=2500,
    )
    return snake

def save_result(init_contour, snake, suffix="result"):
    result = img_bgr.copy()
    # Initial contour – yellow dashed look
    ipts = np.int32(init_contour[:, ::-1])   # (col, row) for cv2
    cv2.polylines(result, [ipts], isClosed=True, color=(0, 255, 255), thickness=1)
    # Snake result – green
    spts = np.int32(snake[:, ::-1])
    cv2.polylines(result, [spts], isClosed=True, color=(0, 255, 0),   thickness=2)
    # Legend
    cv2.putText(result, "Yellow: initial contour", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, "Green:  snake result",    (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0),   2, cv2.LINE_AA)
    path = f"{OUTDIR}/snake_{suffix}.jpg"
    cv2.imwrite(path, result)
    print(f"Saved {path}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# AUTO MODE  (headless)
# ══════════════════════════════════════════════════════════════════════════════
def run_auto():
    print("Running in AUTO mode (no display needed).")
    h, w = img_gray.shape

    # ── initial circular contour around main circle (centre ~310, 310, r~130)
    cx, cy, r = w * 0.39, h * 0.52, min(h, w) * 0.23
    theta = np.linspace(0, 2 * np.pi, 300)
    init  = np.column_stack([cy + r * np.sin(theta),
                              cx + r * np.cos(theta)])

    print("Evolving snake (this takes a few seconds) …")
    snake = run_snake(init)
    save_result(init, snake, suffix="auto")
    print("Auto mode complete.")


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MODE  (needs display)
# ══════════════════════════════════════════════════════════════════════════════
INSTRUCTIONS = [
    "Left-click: add seed point",
    "ENTER: run snake    R: reset    Q: quit & save",
]

seed_points = []

def draw_ui(base):
    out = base.copy()
    for i, (x, y) in enumerate(seed_points):
        cv2.circle(out, (x, y), 5, (0, 255, 255), -1)
        if i > 0:
            cv2.line(out, seed_points[i-1], (x, y), (0, 255, 255), 1)
    if len(seed_points) > 1:
        cv2.line(out, seed_points[-1], seed_points[0], (0, 255, 255), 1)
    # HUD
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], 56), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    for i, txt in enumerate(INSTRUCTIONS):
        cv2.putText(out, txt, (10, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        seed_points.append((x, y))
        cv2.imshow("Active Contour – Snake", draw_ui(img_bgr))

def run_interactive():
    cv2.namedWindow("Active Contour – Snake", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Active Contour – Snake", 900, 680)
    cv2.setMouseCallback("Active Contour – Snake", mouse_cb)
    cv2.imshow("Active Contour – Snake", draw_ui(img_bgr))
    print("Click seed points, then press ENTER to run the snake.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        closed = cv2.getWindowProperty(
            "Active Contour – Snake", cv2.WND_PROP_VISIBLE) < 1

        if key == 13 and len(seed_points) >= 3:         # ENTER
            print(f"Running snake with {len(seed_points)} seed points …")
            init  = interpolate_contour(seed_points)
            snake = run_snake(init)
            result = save_result(init, snake, suffix="interactive")
            cv2.imshow("Active Contour – Snake", draw_ui(result))

        elif key == ord('r'):                            # R – reset
            seed_points.clear()
            cv2.imshow("Active Contour – Snake", draw_ui(img_bgr))
            print("Reset.")

        elif key == ord('q') or closed:                 # Q / window closed
            break

    cv2.destroyAllWindows()
    print("Interactive mode done.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if "--auto" in sys.argv:
        run_auto()
    else:
        # Try to open a display; fall back to auto if unavailable
        try:
            test_win = "test_display"
            cv2.namedWindow(test_win, cv2.WINDOW_NORMAL)
            cv2.destroyWindow(test_win)
            run_interactive()
        except cv2.error:
            print("No display detected. Switching to --auto mode.")
            run_auto()
