import os

import cv2 as cv
import numpy as np


class DatasetHandler:
    def __init__(self):
        # Total number of synchronized RGB/depth frames in the sample dataset.
        self.num_frames = 52

        # Resolve data directories relative to repository root.
        module_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir = os.path.abspath(os.path.join(module_dir, "..", ".."))
        self.image_dir = os.path.join(root_dir, "data", "rgb")
        self.depth_dir = os.path.join(root_dir, "data", "depth")

        # Grayscale images for matching, RGB images for display, depth maps for 3D recovery.
        self.images = []
        self.images_rgb = []
        self.depth_maps = []

        # Camera intrinsic matrix K.
        self.k = np.array(
            [[640, 0, 640],
             [0, 480, 480],
             [0, 0, 1]],
            dtype=np.float32,
        )

        # Preload all frames once; this simplifies downstream indexing logic.
        self.read_frame()
        # Remove progress text after loading completes.
        print("\r" + " " * 20 + "\r", end="")

    def read_frame(self):
        # Keep depth/image lists aligned by loading in deterministic order.
        self._read_depth()
        self._read_image()

    def _read_image(self):
        for i in range(1, self.num_frames + 1):
            # Build zero-padded frame id: 1 -> 00001.
            zeroes = "0" * (5 - len(str(i)))
            im_name = f"{self.image_dir}/frame_{zeroes}{i}.png"
            # Grayscale for feature extraction and matching.
            self.images.append(cv.imread(im_name, flags=0))
            # RGB for matplotlib display (OpenCV default is BGR).
            self.images_rgb.append(cv.imread(im_name)[:, :, ::-1])
            progress = int((i + self.num_frames) / (self.num_frames * 2 - 1) * 100)
            print(f"Data loading: {progress}%", end="\r")

    def _read_depth(self):
        for i in range(1, self.num_frames + 1):
            # Depth files share naming pattern with RGB frames.
            zeroes = "0" * (5 - len(str(i)))
            depth_name = f"{self.depth_dir}/frame_{zeroes}{i}.dat"
            # Load depth map and convert to the expected scale for this project.
            depth = np.loadtxt(depth_name, delimiter=",", dtype=np.float64) * 1000.0
            self.depth_maps.append(depth)
            progress = int(i / (self.num_frames * 2 - 1) * 100)
            print(f"Data loading: {progress}%", end="\r")
