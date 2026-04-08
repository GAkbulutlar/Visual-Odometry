# Visual Odometry for Localization in Autonomous Driving

This repository implements a compact monocular visual odometry pipeline for autonomous driving scenes. The system estimates camera motion from image sequences and depth maps, then reconstructs and visualizes the vehicle trajectory over time.

The project is organized as a lightweight Python package under src and is designed to be easy to run, inspect, and extend.

## Quick Start

1. Create and activate the environment.
2. Place dataset files in `data/rgb` and `data/depth`.
3. Run the pipeline script.

```bash
conda env create -f environment.yml
conda activate visual-odometry
python scripts/run_vo.py
```

## Project Goals

- Estimate frame-to-frame camera pose from visual features.
- Build a global trajectory from relative pose estimates.
- Visualize feature correspondences and the final 2D and 3D path.
- Keep the code modular for experimentation.

## How It Works

The pipeline follows these stages:

1. Load synchronized RGB and depth frames.
2. Detect and describe ORB keypoints.
3. Match descriptors between consecutive frames.
4. Filter matches with ratio test and RANSAC geometry checks.
5. Back-project matched pixels with depth to form 3D-2D correspondences.
6. Estimate relative pose using PnP RANSAC.
7. Compose relative transforms into a global trajectory.
8. Plot trajectory and qualitative feature-match visualizations.

## Repository Structure

```text
Visual-Odometry/
  src/
    visual_odometry/
      __init__.py          # public package exports
      dataset.py           # frame loading and camera intrinsics
      matching.py          # ORB matching and pose estimation
      pipeline.py          # end-to-end trajectory pipeline
      visualization.py     # trajectory and match plotting
  scripts/
    run_vo.py              # entry point
  data/                    # local dataset (gitignored)
    rgb/
    depth/
  output/                  # generated images/gifs (gitignored)
  environment.yml
  pyrightconfig.json
```

## Setup

Use Conda:

```bash
conda env create -f environment.yml
conda activate visual-odometry
```

Or install dependencies manually:

```bash
pip install numpy opencv-python matplotlib jupyter
```

## Data Layout

Place your dataset locally in this structure:

```text
data/
  rgb/
    frame_00001.png
    ...
  depth/
    frame_00001.dat
    ...
```

Notes:

- RGB and depth filenames must stay index-aligned.
- Depth files are expected as comma-separated numeric grids.

## Run

```bash
python scripts/run_vo.py
```

## Outputs

The run produces:

- Printed pose/inlier diagnostics in the terminal.
- Robust feature-match visualization between selected frames.
- Combined trajectory plots in Z-X, Z-Y, Y-X, and 3D views.

If you save figures or animations, store them under output.

## Developer Notes

- Core package: src/visual_odometry
- Type checking is configured via pyrightconfig.json
- Large local artifacts are excluded from Git with .gitignore

## Limitations

- Relies on available depth maps for back-projection.
- Sensitive to texture-poor scenes and motion blur.
- Uses fixed camera intrinsics from the dataset handler.

## Next Improvements

1. Add CLI arguments for frame ranges and thresholds.
2. Add trajectory export to CSV or TUM format.
3. Integrate quantitative evaluation against ground truth.
4. Add unit tests for matching and transform composition.
