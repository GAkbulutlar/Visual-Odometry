import math

import cv2 as cv
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import cast


def visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=False):
    # Work on copies so callers keep original arrays unchanged.
    image1 = image1.copy()
    image2 = image2.copy()

    for i in range(0, len(image1_points)):
        # Coordinates of a point on t frame
        p1 = (int(image1_points[i][0]), int(image1_points[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(image2_points[i][0]), int(image2_points[i][1]))

        # Draw start point, displacement arrow, and end point.
        cv.circle(image1, p1, 5, (0, 255, 0), 1)
        cv.arrowedLine(image1, p1, p2, (0, 255, 0), 1)
        cv.circle(image1, p2, 5, (255, 0, 0), 1)

        if is_show_img_after_move:
            cv.circle(image2, p2, 5, (255, 0, 0), 1)

    if is_show_img_after_move:
        return image2
    return image1


def visualize_trajectory(trajectory):
    # Split trajectory into per-axis arrays for plotting.
    loc_x = []
    loc_y = []
    loc_z = []
    # Track global min/max to force equal plot scales across subplots.
    max_val = -math.inf
    min_val = math.inf

    # Track Y bounds separately to keep side views centered and readable.
    max_y = -math.inf
    min_y = math.inf

    for i in range(0, trajectory.shape[1]):
        current_pos = trajectory[:, i]

        loc_x.append(current_pos.item(0))
        loc_y.append(current_pos.item(1))
        loc_z.append(current_pos.item(2))
        if np.amax(current_pos) > max_val:
            max_val = np.amax(current_pos)
        if np.amin(current_pos) < min_val:
            min_val = np.amin(current_pos)

        if current_pos.item(1) > max_y:
            max_y = current_pos.item(1)
        if current_pos.item(1) < min_y:
            min_y = current_pos.item(1)

    # Midline reference used to symmetrically expand Y limits.
    aux_y_line = loc_y[0] + loc_y[-1]
    if max_val > 0 and min_val > 0:
        min_y = aux_y_line - (max_val - min_val) / 2
        max_y = aux_y_line + (max_val - min_val) / 2
    elif max_val < 0 and min_val < 0:
        min_y = aux_y_line + (min_val - max_val) / 2
        max_y = aux_y_line - (min_val - max_val) / 2
    else:
        min_y = aux_y_line - (max_val - min_val) / 2
        max_y = aux_y_line + (max_val - min_val) / 2

    # Global style configuration for cleaner plots.
    mpl.rc("figure", facecolor="white")
    # Newer Matplotlib releases renamed seaborn styles under the v0_8 prefix.
    available_styles = set(plt.style.available)
    if "seaborn-v0_8-whitegrid" in available_styles:
        plt.style.use("seaborn-v0_8-whitegrid")
    elif "seaborn-whitegrid" in available_styles:
        plt.style.use("seaborn-whitegrid")
    else:
        plt.style.use("default")

    # Create 2D and 3D subplots in one figure.
    fig = plt.figure(figsize=(8, 6), dpi=100)
    gspec = gridspec.GridSpec(3, 3)
    zy_plt = plt.subplot(gspec[0, 1:])
    yx_plt = plt.subplot(gspec[1:, 0])
    traj_main_plt = plt.subplot(gspec[1:, 1:])
    d3_plt = cast(Axes3D, plt.subplot(gspec[0, 0], projection="3d"))

    # Main trajectory plot in Z-X plane.
    toffset = 1.06
    traj_main_plt.set_title("Autonomous vehicle trajectory (Z, X)", y=toffset)
    traj_main_plt.set_title("Trajectory (Z, X)", y=1)
    traj_main_plt.plot(loc_z, loc_x, ".-", label="Trajectory", zorder=1, linewidth=1, markersize=4)
    traj_main_plt.set_xlabel("Z")
    # Auxiliary line from start to end to visualize drift direction.
    traj_main_plt.plot([loc_z[0], loc_z[-1]], [loc_x[0], loc_x[-1]], "--", label="Auxiliary line", zorder=0, linewidth=1)
    # Mark the origin (initial camera location).
    traj_main_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    traj_main_plt.set_xlim(min_val, max_val)
    traj_main_plt.set_ylim(min_val, max_val)
    traj_main_plt.legend(loc=1, title="Legend", borderaxespad=0.0, fontsize="medium", frameon=True)

    # Top subplot: Z-Y projection.
    zy_plt.set_ylabel("Y", labelpad=-4)
    zy_plt.xaxis.set_ticklabels([])
    zy_plt.plot(loc_z, loc_y, ".-", linewidth=1, markersize=4, zorder=0)
    zy_plt.plot([loc_z[0], loc_z[-1]], [(loc_y[0] + loc_y[-1]) / 2, (loc_y[0] + loc_y[-1]) / 2], "--", linewidth=1, zorder=1)
    zy_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    zy_plt.set_xlim(min_val, max_val)
    zy_plt.set_ylim(min_y, max_y)

    # Left subplot: Y-X projection.
    yx_plt.set_ylabel("X")
    yx_plt.set_xlabel("Y")
    yx_plt.plot(loc_y, loc_x, ".-", linewidth=1, markersize=4, zorder=0)
    yx_plt.plot([(loc_y[0] + loc_y[-1]) / 2, (loc_y[0] + loc_y[-1]) / 2], [loc_x[0], loc_x[-1]], "--", linewidth=1, zorder=1)
    yx_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    yx_plt.set_xlim(min_y, max_y)
    yx_plt.set_ylim(min_val, max_val)

    # 3D subplot for full trajectory intuition.
    d3_plt.set_title("3D trajectory", y=toffset)
    d3_plt.plot3D(loc_x, loc_z, loc_y, zorder=0)
    d3_plt.scatter(xs=0, ys=0, zs=0, s=8, c="red", zorder=1)
    d3_plt.set_xlim3d(min_val, max_val)
    d3_plt.set_ylim3d(min_val, max_val)
    d3_plt.set_zlim3d(min_val, max_val)
    d3_plt.tick_params(direction="out", pad=-2)
    d3_plt.set_xlabel("X", labelpad=0)
    d3_plt.set_ylabel("Z", labelpad=0)
    d3_plt.set_zlabel("Y", labelpad=-2)

    # Adjust camera view for a more informative perspective.
    d3_plt.view_init(45, azim=30)
    fig.tight_layout()
    plt.show()
