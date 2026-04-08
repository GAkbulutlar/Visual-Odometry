import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def robust_orb_matches(
    img1,
    img2,
    nfeatures=3000,
    fast_threshold=10,
    ratio_thresh=0.65,
    ransac_reproj_thresh=1.5,
):
    """Return ORB matches that pass ratio test and fundamental-matrix RANSAC."""
    # ORB gives fast binary descriptors suitable for real-time pipelines.
    orb = cv.ORB_create(nfeatures=nfeatures, fastThreshold=fast_threshold)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        # No descriptors means nothing to match.
        return kp1, kp2, [], {"raw": 0, "ratio": 0, "inliers": 0}

    # Brute-force Hamming matcher for ORB's binary descriptors.
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test removes ambiguous nearest-neighbor matches.
    ratio_matches = [
        m for pair in raw_matches if len(pair) == 2 for m, n in [pair] if m.distance < ratio_thresh * n.distance
    ]

    if len(ratio_matches) < 8:
        # Need enough correspondences for fundamental matrix estimation.
        return kp1, kp2, ratio_matches, {
            "raw": len(raw_matches),
            "ratio": len(ratio_matches),
            "inliers": len(ratio_matches),
        }

    # Convert matched keypoints to arrays for robust geometric filtering.
    pts1 = np.float32([kp1[m.queryIdx].pt for m in ratio_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in ratio_matches])

    # RANSAC rejects matches inconsistent with epipolar geometry.
    _, inlier_mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC, ransac_reproj_thresh, 0.99)
    if inlier_mask is None:
        return kp1, kp2, ratio_matches, {
            "raw": len(raw_matches),
            "ratio": len(ratio_matches),
            "inliers": len(ratio_matches),
        }

    # Keep only geometrically consistent matches.
    inlier_mask = inlier_mask.ravel().astype(bool)
    inlier_matches = [m for m, keep in zip(ratio_matches, inlier_mask) if keep]
    # Sort by descriptor distance so strongest matches are first.
    inlier_matches.sort(key=lambda m: m.distance)

    return kp1, kp2, inlier_matches, {
        "raw": len(raw_matches),
        "ratio": len(ratio_matches),
        "inliers": len(inlier_matches),
    }


def estimate_pose_from_pair(dataset, idx=0, max_depth=1000.0):
    """Estimate camera pose from two consecutive frames using PnP RANSAC."""
    # Select consecutive frames and intrinsics.
    img1 = dataset.images[idx]
    img2 = dataset.images[idx + 1]
    depth1 = dataset.depth_maps[idx]
    k = dataset.k

    # Obtain robust 2D-2D correspondences between frames.
    kp1, kp2, matches, _ = robust_orb_matches(img1, img2, ratio_thresh=0.7)
    if not matches:
        return None

    # Build 3D points from frame idx and their 2D correspondences in frame idx+1.
    image2_points = []
    object_points = []

    # K^-1 converts pixel coordinates into normalized camera rays.
    k_inv = np.linalg.inv(k)
    for m in matches:
        u1, v1 = kp1[m.queryIdx].pt
        u2, v2 = kp2[m.trainIdx].pt

        # Use depth at the first-image keypoint to recover its 3D location.
        s = depth1[int(v1), int(u1)]
        if s >= max_depth:
            # Skip far/noisy depth values that destabilize PnP.
            continue

        # Back-project pixel to 3D camera coordinates.
        p_c = k_inv @ (s * np.array([u1, v1, 1]))
        image2_points.append([u2, v2])
        object_points.append(p_c)

    if len(object_points) < 6:
        # PnP needs enough correspondences to be reliable.
        return None

    object_points = np.vstack(object_points)
    image_points = np.array(image2_points, dtype=np.float32)

    # Estimate rotation and translation robustly in presence of outliers.
    ok, rvec, tvec, inliers = cv.solvePnPRansac(object_points, image_points, k, None)
    if not ok:
        return None

    # Convert Rodrigues rotation vector to a 3x3 rotation matrix.
    rmat, _ = cv.Rodrigues(rvec)
    return rmat, tvec, inliers


def plot_matched_features(
    dataset,
    idx1=None,
    idx2=None,
    n_matches=60,
    max_frame_gap=4,
    ratio_thresh=0.65,
    ransac_reproj_thresh=1.5,
):
    """Plot robust feature matches for a random or specified frame pair."""
    # Pick frame indices unless user provided explicit pair.
    num = dataset.num_frames
    if idx1 is None:
        idx1 = random.randint(0, num - 2)

    if idx2 is None:
        upper = min(num - 1, idx1 + max_frame_gap)
        if upper <= idx1:
            upper = min(num - 1, idx1 + 1)
        idx2 = random.randint(idx1 + 1, upper)

    img1 = dataset.images[idx1]
    img2 = dataset.images[idx2]

    # Compute robust inlier matches and quality statistics.
    kp1, kp2, inlier_matches, stats = robust_orb_matches(
        img1,
        img2,
        ratio_thresh=ratio_thresh,
        ransac_reproj_thresh=ransac_reproj_thresh,
    )

    if not inlier_matches:
        print("No robust matches found for this pair.")
        return

    # Draw up to n_matches best inliers for visual inspection.
    matched_img = cv.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        inlier_matches[:n_matches],
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Show match image and include matching statistics in the title.
    plt.figure(figsize=(16, 6))
    plt.imshow(matched_img, cmap="gray")
    plt.title(
        f"Inlier ORB matches: frame {idx1 + 1} vs {idx2 + 1} | "
        f"raw={stats['raw']}, ratio={stats['ratio']}, inliers={stats['inliers']}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()
