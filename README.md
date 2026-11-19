# Tree Reconstruction

Tools for reconstructing fruit trees from Helios2 depth and intensity captures,
and for manually annotating the calibration marker used to estimate camera poses.

The reconstruction pipeline detects and fits a known marker, refines frame
registration with ICP and a pose graph, merges the point clouds, and isolates and
filters the target tree. Optional non-rigid refinement produces a separate result
for moving branches. The annotation application provides manual corner selection,
pose estimation, projected-model review, and JSON/CSV export.

## Installation

### Docker

The supplied image includes Python 3.10, Open3D, OpenCV, NumPy, SciPy, and PySide6.
From the repository root:

```bash
bash docker/build.sh
```

Before starting the container, update the host data mount in
[`docker/run.sh`](docker/run.sh) to your dataset location. For example, replace
its existing data volume with:

```bash
--volume /media/simundic/T9/pears_dataset:/data \
```

Then start the container:

```bash
bash docker/run.sh
```

The repository is available at `/workspace` and the mounted data at `/data`.
Run the commands below from `/workspace` inside the container. GUI tools require
an X11 or XWayland desktop on the host and `xhost` for the supplied runner.
Rebuild the image and recreate the container after changing dependencies.

### Local annotation installation

To use only the annotation application without Docker, install Python 3.10+ and:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r marker_annotation/requirements.txt
```

Run annotation commands from the repository root and use host dataset paths.
For the full reconstruction environment, use Docker.

## Input data

Captures use the following layout:

```text
pears_original/
└── <side>/
    └── <capture>/
        └── marker_captures/
            └── depth/
                ├── 0_depth.png
                ├── 0_intensity.png
                ├── 0_camera_params.yaml
                └── ...
```

Each frame has a shared ID. Depth PNGs store axial Z in millimetres, with zero
indicating invalid depth. Intensity and depth are treated as pixel-aligned for
the pears dataset. Calibration YAML contains the image dimensions, flattened
3×3 camera matrix `k`, distortion coefficients `d`, and distortion model.

Reconstruction also requires the matching CAD mesh. Place
`helios2DepthMarker.ply` in `data/`, or configure its actual path. Dataset and CAD
files are not installed by the Docker build. The annotation application includes
a named corner table extracted from this mesh.

## Reconstruction

Edit [`marker_based_reconstruction/cfg/marker_based_reconstruction.yaml`](marker_based_reconstruction/cfg/marker_based_reconstruction.yaml).
For the example Docker mount above, set:

```yaml
dataset_dir: /data/pears_original
frame_dir: /data/pears_original/A/tree_0_V_0000/marker_captures/depth
output_dir: /workspace/reconstruction_output
model_path: /workspace/data/helios2DepthMarker.ply
model_units_to_m: 0.001
```

The same file controls marker detection, ICP, tree isolation, filtering,
visualization, and optional non-rigid refinement. To select a different config,
set `TREE_CFG=/path/to/config.yaml` before running.

Automatic marker detection requires a separately running SAM 3 HTTP service.
The container can reach a host service through `host.docker.internal`:

```bash
export SAM3_URL=http://host.docker.internal:8000
python -m marker_based_reconstruction.src.sam3_client
```

Reconstruct the capture selected by `frame_dir`:

```bash
python -m marker_based_reconstruction.src.marker_based_reconstruction
```

Or process all captures below `dataset_dir`:

```bash
python -m marker_based_reconstruction.src.reconstruct_dataset
```

Results are written under `<output_dir>/<side>/<capture>/`:

| Output | Contents |
|---|---|
| `<capture>_merged.ply` | Registered, isolated, filtered tree |
| `<capture>_merged.frames.txt` | Contributing frame paths |
| `<capture>_merged.poses.npz` | Camera poses, marker poses, and intrinsics |
| `<capture>_merged_marker_removed.ply` | Full scene with marker and pole removed, when enabled |
| `<capture>_merged_nonrigid.ply` | Separate non-rigid result, when enabled |

Additional intermediate clouds and displacement metadata depend on the enabled
options. See the [reconstruction documentation](marker_based_reconstruction/README.md)
for configuration and output details.

## Marker annotation

Inside the container, start the desktop application with:

```bash
python -m marker_annotation
```

The start screen lets you browse for a dataset folder and session file, choose the
marker profile, and enter your reviewer ID. Choices are remembered and preselected
on the next launch. Existing sessions resume saved annotations and the last frame.

Set **Dataset folder** to `/data/pears_original`. In **Include folder names**,
use `A; marker_captures` for only marker captures from side A. Semicolons mean
AND; commas mean OR within a group: `A,B; marker_captures` includes marker
captures from either side. Leave the filter blank for all frames. Folder names
match exactly; this dataset uses `marker_captures` (plural).
Change these settings later with **Workspace…**. Filters hide frames without
deleting saved records. Preferences live in `annotation_sessions/preferences.json`
on the host-mounted repository.

Select a corner, click its image location, estimate the pose, inspect the model
projection and residuals, then accept or flag the frame. At least four visible,
well-spread corners are required; include corners at different CAD heights to
reduce ambiguity. Click only visible corners and leave the rest unmarked; there
are no occlusion or visibility controls.

The **Pears** profile loads the real marker corner table, assumes pixel alignment,
and uses millimetre axial depth. Sessions autosave and restore through the same
session path. Export JSON and CSV from the application. Start an independent
repeat to annotate a frame again without seeing previous observations.
Annotation runs independently of SAM 3 and reconstruction; its records are not
automatically consumed by the reconstruction pipeline.

For a bundled synthetic example:

```bash
python -m marker_annotation \
  --dataset marker_annotation/example \
  --geometry marker_annotation/example/geometry.json \
  --session /workspace/annotation_sessions/synthetic.json
```

See the [annotation documentation](marker_annotation/README.md) for shortcuts,
custom geometry, depth checks, and review conventions.

## Coordinate conventions

Distances and translations are expressed in metres after loading. Camera axes
are +X right, +Y down, and +Z forward. Marker coordinates retain the CAD origin
and axes. `T_A_B` maps points from frame A to frame B:
`p_B = T_A_B @ p_A`. Thus `T_marker_cam` maps marker coordinates into a camera.
Reconstruction uses the first accepted camera as its reference frame.

Annotation poses use `X_camera = Rodrigues(rvec) @ X_marker + tvec_m`.
Clicks are stored in original image pixels with a top-left origin, independent
of display zoom. Numerical fit quality and manual acceptance are separate.

## Tests

Run the focused annotation tests from the repository root or `/workspace`:

```bash
pip install pytest
QT_QPA_PLATFORM=offscreen python -m pytest marker_annotation/tests -q
```

The tests cover coordinate mapping, persistence, pose recovery, degenerate
configurations, depth checks, and acceptance invalidation. The CAD test requires
`data/helios2DepthMarker.ply`.

## Repository layout

| Directory | Purpose |
|---|---|
| `marker_based_reconstruction/` | Reconstruction pipeline and configuration |
| `marker_annotation/` | Desktop annotation application, example, and tests |
| `dataset_analysis/` | Dataset validation and filtering utilities |
| `docker/` | Container build and run scripts |
| `data/` | Local CAD and other data files; excluded from version control |
