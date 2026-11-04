# Marker-based tree reconstruction

This package turns a set of depth and intensity frames taken around a fruit tree
into one registered, isolated, and filtered 3-D point cloud. A known calibration
marker visible in every frame provides the initial camera poses; overlapping scene
geometry then refines those poses before the point clouds are merged.

```text
depth + intensity frames
        ↓
marker detection and 3-D model fitting
        ↓
initial camera poses from the shared marker
        ↓
pairwise ICP and pose-graph optimization
        ↓
point-cloud merge → target-tree isolation → visibility filtering
        ↓
PLY reconstruction + frame and pose metadata

optional branch: rigid clouds → non-rigid branch-tip refinement → separate PLY
```

## Project layout

| Path | Purpose |
|---|---|
| [`cfg/marker_based_reconstruction.yaml`](cfg/marker_based_reconstruction.yaml) | Dataset paths, visualization switches, and pipeline parameters |
| [`src/marker_based_reconstruction.py`](src/marker_based_reconstruction.py) | Main single-capture reconstruction pipeline |
| [`src/reconstruct_dataset.py`](src/reconstruct_dataset.py) | Batch driver for all captures below `dataset_dir` |
| [`src/depth_regions.py`](src/depth_regions.py) | Frame loading and rectangular depth-region proposals |
| [`src/marker_detection_sam3.py`](src/marker_detection_sam3.py) | SAM 3 mask refinement and marker candidate selection |
| [`src/marker_registration.py`](src/marker_registration.py) | 3-D marker-model fitting and pose validation |
| [`src/sam3_client.py`](src/sam3_client.py) | HTTP client for the host-side SAM 3 service |
| [`src/annotate_marker_ground_truth.py`](src/annotate_marker_ground_truth.py) | Interactive manual marker annotation |
| [`src/viz_utils.py`](src/viz_utils.py) | Shared 2-D and 3-D visualization helpers |

## Requirements and setup

The supplied Docker image contains Python 3.10, Open3D, OpenCV, SciPy,
scikit-learn, and the other Python and native dependencies used by the pipeline.
From the repository root, build and start it with:

```bash
./docker/build.sh
./docker/run.sh
```

[`docker/run.sh`](../docker/run.sh) mounts the repository at `/workspace`, mounts
the host data directory at `/data`, forwards X11 for visualizations, and makes the
host available inside the container as `host.docker.internal`.

SAM 3 is not run in this CPU container. It must be running as an HTTP service on
the host. The client first tries `host.docker.internal`, then the standard Docker
bridge gateway. Set `SAM3_URL` when the service is somewhere else:

```bash
export SAM3_URL=http://host.docker.internal:8000
```

From `/workspace` in the container, verify that the service is reachable:

```bash
python -m marker_based_reconstruction.src.sam3_client
```

Add `--smoke` to send two synthetic segmentation requests as well as checking the
health endpoint.

## Expected input layout

Each capture ends in a `marker_captures/depth` directory. Every frame is a group
of three files with the same ID:

```text
<dataset_dir>/
└── <side>/
    └── <capture>/
        └── marker_captures/
            └── depth/
                ├── 0_depth.png
                ├── 0_intensity.png
                ├── 0_camera_params.yaml
                ├── 1_depth.png
                ├── 1_intensity.png
                └── 1_camera_params.yaml
```

| File | Contents |
|---|---|
| `<id>_depth.png` | Depth image; stored values are converted from millimetres to metres |
| `<id>_intensity.png` | Grayscale or color intensity image aligned with the depth image |
| `<id>_camera_params.yaml` | Camera `width`, `height`, and flattened 3 × 3 intrinsic matrix `k` |

All frames in a capture must observe the same physical tree and calibration
marker. The marker model configured by `model_path` must match that marker.

## Configuration

The default configuration is
[`cfg/marker_based_reconstruction.yaml`](cfg/marker_based_reconstruction.yaml).
To use another YAML file without editing the default:

```bash
export TREE_CFG=/path/to/my_reconstruction.yaml
```

The settings are grouped by purpose:

| Group | Controls |
|---|---|
| Top-level paths | `dataset_dir`, `frame_dir`, `output_dir`, and `model_path` |
| Marker model | Conversion of model coordinates to metres with `model_units_to_m` |
| `sam3` | Mask selection, prompt coverage, candidate limits, and optional step images |
| `pose_graph` | ICP resolution, robust loss, accepted edges, and global optimization |
| `tree_isolation` | DBSCAN clustering, target selection, and splitting fused neighboring trees |
| `visibility_filter` | Multi-view confirmations and free-space contradictions |
| `support_filter` | Optional simpler multi-frame proximity filter |
| `marker_removal` | Remove the marker and pole before isolation, filtering, and non-rigid refinement |
| `non_rigid` | Optional deformation-graph refinement of weakly supported branch tips |
| Visualization | SAM 3, isolation, and final-reconstruction windows |
| Batch behavior | Whether existing reconstructions are overwritten |

The visibility filter takes precedence if both filtering methods are enabled.
Parameter comments and units are documented beside each value in the YAML file.

## Running the project

Run commands from `/workspace` inside the container. Module execution is
preferred because it gives Python an unambiguous package path.

### One capture

Set `frame_dir` in the configuration to a capture's `marker_captures/depth`
directory, then run:

```bash
python -m marker_based_reconstruction.src.marker_based_reconstruction
```

This calls
[`reconstruct_capture()`](src/marker_based_reconstruction.py), which processes
that capture and writes its outputs below `output_dir`.

Set `non_rigid.enabled: true` to additionally refine movable branch tips and
write a separate non-rigid result. The normal rigid result is still produced
first and is not replaced.

### Complete dataset

Set `dataset_dir`, `output_dir`, and `overwrite`, then run:

```bash
python -m marker_based_reconstruction.src.reconstruct_dataset
```

The batch driver searches recursively for `marker_captures/depth` directories.
A failed capture is reported without stopping the remaining captures.

## Pipeline walkthrough

This section is a summary. See
[`PIPELINE_WALKTHROUGH.md`](PIPELINE_WALKTHROUGH.md) for the full data flow,
selection rules, fallbacks, and non-rigid refinement details.

### 1. Discover and load frames

[`discover_frame_ids()`](src/marker_based_reconstruction.py) finds matching depth
frames. [`load_frame()`](src/depth_regions.py) loads depth, intensity, and camera
intrinsics and converts depth values to metres.

### 2. Propose marker regions in 2-D

[`sweep_region_proposals()`](src/depth_regions.py) segments the frame using depth
and intensity continuity, reconnects fragmented regions, scores rectangle-like
shapes, and removes duplicate candidates.

### 3. Refine candidates with SAM 3

[`run_pipeline()`](src/marker_detection_sam3.py) sends each promising region to
the host SAM 3 service through [`sam3_client.py`](src/sam3_client.py). SAM 3
returns a cleaner marker mask for 3-D verification.

### 4. Fit the 3-D marker model

[`verify_candidates()`](src/marker_registration.py) lifts candidate pixels into a
point cloud and fits the known marker model with feature-based registration and
ICP. A candidate is accepted only when both the visible panel surface and its
raised relief match with sufficiently low error. Frames without an accepted
marker are skipped.

The returned transform is `T_marker_cam`: it maps marker-model coordinates into
that frame's camera coordinate system.

### 5. Build initial camera poses

[`build_initial_global_poses()`](src/marker_based_reconstruction.py) selects the
first accepted frame as the reference and chains every camera pose through the
shared marker:

```text
T_cam_i_cam_ref = T_marker_cam_ref @ inverse(T_marker_cam_i)
```

Throughout the project, `T_A_B` maps points from coordinate frame A into frame B:
`p_B = T_A_B @ p_A`.

### 6. Refine registration

[`register_to_reference()`](src/marker_based_reconstruction.py) downsamples the
clouds, estimates normals, and runs point-to-plane ICP between nearby frame pairs.
A robust Tukey loss reduces the influence of branches that moved between frames.
Only sufficiently accurate pairwise constraints enter the anchored pose graph,
which jointly optimizes all camera poses. If no usable constraints survive or
optimization fails, the pipeline keeps the marker-only poses.

### 7. Merge the point clouds

The optimized transforms place every full-resolution cloud in the reference
camera frame. [`merge_clouds()`](src/marker_based_reconstruction.py) combines
them; each source frame retains a distinct color.

### 8. Isolate the target tree

[`isolate_target_tree()`](src/marker_based_reconstruction.py) separates connected
components with DBSCAN and selects the substantial cluster toward which the
cameras point most directly. If a neighboring tree is physically fused to the
selected cluster, an along-row density valley is used to split it off.

When `marker_removal.enabled` is true,
[`remove_marker_and_pole()`](src/marker_based_reconstruction.py) removes a padded
marker-model box and a narrow column below the panel immediately after rigid
registration. Tree isolation, filtering, and non-rigid refinement all use these
cleaned per-frame clouds. The pipeline also saves the cleaned full scene before
isolation or filtering as `<capture>_merged_marker_removed.ply`.

### 9. Filter ghosts and noise

By default, [`filter_by_visibility()`](src/marker_based_reconstruction.py)
reprojects merged points into every camera. It keeps points confirmed by enough
frames and removes weakly confirmed points when several cameras observed free
space at their locations. This targets floating noise and ghosts from moving
branches without requiring every real branch to appear in multiple views.

[`filter_by_frame_support()`](src/marker_based_reconstruction.py) is a simpler
alternative that retains points supported by nearby observations from multiple
frames.

### 10. Save the reconstruction

[`save_reconstruction()`](src/marker_based_reconstruction.py) writes the final
cloud and its metadata files using the layout described below.

If `non_rigid.enabled` is true, the rigidly registered per-frame clouds also pass
through [`refine_clouds()`](src/non_rigid_refinement.py). Sparse, linear branch
ends with weak cross-frame support can move toward a multi-frame consensus while
stable and lower-scene geometry remains anchored. The deformed clouds then pass
through the same target isolation and filtering stages and are saved separately.

## Outputs

For an input capture at `<dataset_dir>/<side>/<capture>/...`, files are written to:

```text
<output_dir>/<side>/<capture>/
```

| File | Purpose |
|---|---|
| `<capture>_merged.ply` | Final isolated and filtered point cloud |
| `<capture>_merged.frames.txt` | Contributing depth-frame paths in reconstruction order; the reference frame is first |
| `<capture>_merged.poses.npz` | Frame IDs, final camera-to-reference poses, raw marker poses, and camera intrinsics |
| `<capture>_merged_scene.ply` | Scene before tree isolation; marker and pole are already absent when removal is enabled |
| `<capture>_merged_prefilter.ply` | Isolated tree before filtering; written only when a filter runs |
| `<capture>_merged_marker_removed.ply` | Full rigid scene with only the marker and its support pole removed |
| `<capture>_merged_nonrigid.ply` | Separately isolated and filtered non-rigid result; written only when enabled |
| `<capture>_merged_nonrigid_displacements.npz` | Per-point displacement vectors, magnitudes, mobility, and frame offsets |

The `.poses.npz` arrays are aligned with the order in `.frames.txt`, allowing
later reprojection or evaluation without repeating marker detection.
The marker-removed PLY has its own `.frames.txt` and `.poses.npz` sidecars.
The non-rigid branch also receives its own `.frames.txt`, `.poses.npz`, and any
applicable `_scene.ply` or `_prefilter.ply` sidecars.

## Visualization and debugging

Three configuration switches open blocking windows:

| Setting | View |
|---|---|
| `visualize_sam3` | Per-frame marker proposals, SAM 3 masks, and accepted fits |
| `visualize_isolation` | Scene clusters, camera aim, fused-tree split, and along-row density |
| `visualize_reconstruction` | Final registered per-frame point clouds in Open3D |

Close a window to let the pipeline continue. Disable these options for unattended
batch processing or environments without a display.

For VS Code, use the module-based entries in
[`../.vscode/launch.json`](../.vscode/launch.json):

- **Marker reconstruction: single capture**
- **Marker reconstruction: dataset**

Avoid the generic **Python: Current File** configuration for
`marker_based_reconstruction.py`: the file has the same name as its parent
package, so file-based debugger launchers can resolve the file instead of the
package and cause circular-import errors.
