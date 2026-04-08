import numpy as np

from .dataset import DatasetHandler
from .matching import estimate_pose_from_pair, plot_matched_features
from .visualization import visualize_trajectory


def build_trajectory(dataset_handler):
    """Estimate and return camera trajectory for all frame pairs.

    Returns:
        np.ndarray: 3xN matrix where each column is camera position [x, y, z].
    """
    # T_i0 maps points from frame 0 coordinates to frame i coordinates.
    t_i0 = np.eye(4, dtype=np.float64)

    # Start at origin for frame 0 camera center.
    centers = [np.zeros(3, dtype=np.float64)]

    failed_pairs = 0
    for idx in range(dataset_handler.num_frames - 1):
        pose = estimate_pose_from_pair(dataset_handler, idx=idx)
        if pose is None:
            failed_pairs += 1
            # Keep trajectory length aligned with frame count if one pair fails.
            centers.append(centers[-1].copy())
            continue

        rmat, tvec, _ = pose

        # Relative transform from frame idx to frame idx + 1.
        t_rel = np.eye(4, dtype=np.float64)
        t_rel[:3, :3] = rmat
        t_rel[:3, 3] = tvec.reshape(3)

        # Compose transforms: T_(idx+1,0) = T_(idx+1,idx) * T_(idx,0).
        t_i0 = t_rel @ t_i0

        # Camera center in frame-0 coordinates: C = -R^T * t.
        r_i0 = t_i0[:3, :3]
        t_i0_vec = t_i0[:3, 3]
        c_i0 = -r_i0.T @ t_i0_vec
        centers.append(c_i0)

    if failed_pairs:
        print(f"Warning: pose estimation failed for {failed_pairs} frame pair(s).")

    return np.array(centers, dtype=np.float64).T


def main():
    # Load camera intrinsics, grayscale/RGB frames, and depth maps into memory.
    dataset_handler = DatasetHandler()

    # Estimate relative pose between frame 0 and frame 1.
    pose = estimate_pose_from_pair(dataset_handler, idx=0)
    if pose is None:
        # Common failure cases: insufficient robust matches or invalid depth correspondences.
        print("Pose estimation failed for the selected pair.")
    else:
        # Pose tuple contains rotation matrix, translation vector, and inlier indices.
        _, tvec, inliers = pose
        inlier_count = 0 if inliers is None else len(inliers)
        print(f"Pose estimated. PnP inliers: {inlier_count}, translation: {tvec.ravel()}")

    # Build and plot full trajectory across the sequence.
    trajectory = build_trajectory(dataset_handler)
    visualize_trajectory(trajectory)

    # Display robust ORB feature matches for a random frame pair.
    plot_matched_features(dataset_handler)
