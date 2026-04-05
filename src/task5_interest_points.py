"""
Task 5 – Interest Point Detection & Description (ORB)
======================================================
Uses ORB (Oriented FAST + Rotated BRIEF) to:
  1. Detect and visualise keypoints in both images.
  2. Compute binary descriptors and match them with a Brute-Force matcher.
  3. Apply Lowe's ratio test to keep only reliable matches.
  4. Estimate a homography with RANSAC and warp image 1 into image 2's view.

Run from the project root:
    python src/task5_interest_points.py
Outputs saved to:  images/interest_points/
"""

import cv2
import numpy as np
import os

IMG1   = "images/original/photo.jpg"
IMG2   = "images/original/photo_perspective.jpg"
OUTDIR = "images/interest_points"
os.makedirs(OUTDIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
img1 = cv2.imread(IMG1)
img2 = cv2.imread(IMG2)
assert img1 is not None, f"Cannot read {IMG1}"
assert img2 is not None, f"Cannot read {IMG2}"

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ── ORB detector + descriptor ─────────────────────────────────────────────────
# ORB = Oriented FAST keypoint detector + Rotated BRIEF descriptor.
# Benefits: rotation-invariant, scale-invariant (pyramid), royalty-free,
#           very fast, descriptor is binary → Hamming distance matching.
orb = cv2.ORB_create(
    nfeatures=1000,       # maximum keypoints to detect
    scaleFactor=1.2,      # scale between pyramid levels
    nlevels=8,            # number of pyramid levels
    edgeThreshold=31,     # border margin where keypoints are not detected
    patchSize=31,         # patch size for BRIEF orientation computation
    fastThreshold=20,     # FAST corner response threshold
)

# detectAndCompute returns (keypoints, descriptors)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)
print(f"Image 1 keypoints: {len(kp1)}")
print(f"Image 2 keypoints: {len(kp2)}")

# ── Visualise keypoints (rich: shows scale and orientation) ───────────────────
kp1_img = cv2.drawKeypoints(
    img1, kp1, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    color=(0, 200, 255)
)
kp2_img = cv2.drawKeypoints(
    img2, kp2, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    color=(0, 200, 255)
)
cv2.imwrite(f"{OUTDIR}/keypoints_img1.jpg", kp1_img)
cv2.imwrite(f"{OUTDIR}/keypoints_img2.jpg", kp2_img)
print(f"[1/4] Saved keypoint images")

# ── Brute-Force Matching with Lowe's ratio test ───────────────────────────────
# NORM_HAMMING is correct for binary (ORB) descriptors.
# crossCheck=False because we use knnMatch (k=2) + ratio test instead.
bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
kmatches = bf.knnMatch(des1, des2, k=2)

# Lowe's ratio test: keep a match only if it is significantly better
# than the second-best match (distance_1 < 0.75 × distance_2).
good = [m for m, n in kmatches if m.distance < 0.75 * n.distance]
print(f"[2/4] Good matches after Lowe's ratio test: {len(good)}")

# Draw top-60 matches
top60 = sorted(good, key=lambda x: x.distance)[:60]
match_img = cv2.drawMatches(
    img1, kp1, img2, kp2, top60, None,
    matchColor=(0, 255, 0),
    singlePointColor=(180, 180, 180),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite(f"{OUTDIR}/matches.jpg", match_img)
print(f"[2/4] Saved {OUTDIR}/matches.jpg")

# ── Homography estimation with RANSAC ─────────────────────────────────────────
if len(good) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts, dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0   # max reprojection error (px) for an inlier
    )
    inliers        = int(mask.sum()) if mask is not None else 0
    inlier_matches = [m for m, ok in zip(good, mask.ravel()) if ok]
    print(f"[3/4] RANSAC inliers: {inliers}/{len(good)}")

    # Draw only geometrically verified (inlier) matches
    inlier_img = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        matchColor=(0, 255, 100),
        singlePointColor=(160, 160, 160),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.putText(inlier_img, f"RANSAC inliers: {inliers}/{len(good)}",
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 100), 2)
    cv2.imwrite(f"{OUTDIR}/inlier_matches.jpg", inlier_img)
    print(f"[3/4] Saved {OUTDIR}/inlier_matches.jpg")

    # Warp image 1 into the perspective of image 2
    h, w = img2.shape[:2]
    warped = cv2.warpPerspective(img1, H, (w, h))
    cv2.imwrite(f"{OUTDIR}/warped_img1_to_img2.jpg", warped)
    print(f"[4/4] Saved {OUTDIR}/warped_img1_to_img2.jpg")
else:
    print("Not enough matches to estimate homography.")

print("\nInterest point detection complete.")
