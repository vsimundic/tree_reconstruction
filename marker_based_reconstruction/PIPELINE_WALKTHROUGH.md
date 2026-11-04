# Reconstruction pipeline walkthrough

This document describes the production path from a capture directory to the
saved rigid and optional non-rigid point clouds. It follows the order used by
[`reconstruct_capture()`](src/marker_based_reconstruction.py).

## Pipeline state

The main objects passed between stages are:

| Name | Type | Contents |
|---|---|---|
| `frame_ids` | `list[str]` | Frame basenames, sorted numerically when possible |
| `detections` | `dict[frame_id, dict]` | Accepted marker pose, depth, intensity, and intrinsics for each usable frame |
| `poses` | `dict[frame_id, 4x4 array]` | Initial camera-to-reference transforms derived from the marker |
| `clouds` | `dict[frame_id, PointCloud]` | Full-resolution per-frame clouds, transformed into the reference frame after registration |
| `final_poses` | `dict[frame_id, 4x4 array]` | Optimized camera-to-reference transforms |

Transform names use `T_A_B`: the transform maps a point from frame A to frame B.

```text
p_B = T_A_B @ p_A
```

The first frame with an accepted marker detection defines the reference camera
frame. Its pose is the identity and node 0 remains fixed during pose-graph
optimization.

## 1. Entry points and capture discovery

There are two production entry points:

- [`src/marker_based_reconstruction.py`](src/marker_based_reconstruction.py)
  processes `CFG.frame_dir`.
- [`src/reconstruct_dataset.py`](src/reconstruct_dataset.py) searches below
  `CFG.dataset_dir` for directories ending in `marker_captures/depth` and calls
  the same single-capture function for each one.

[`discover_frame_ids()`](src/marker_based_reconstruction.py) finds files matching
`*_depth.png`. It strips the suffix and sorts the IDs numerically if every ID can
be converted to an integer; otherwise it uses lexical order. The intensity and
camera-parameter filenames are derived from the resulting ID.

The batch driver checks the expected output path before processing a capture. If
the file exists and `overwrite` is false, that capture is skipped. Exceptions are
caught per capture so one failure does not stop the batch.

## 2. Frame loading

[`load_frame()`](src/depth_regions.py) reads three files for each frame:

```text
<id>_depth.png
<id>_intensity.png
<id>_camera_params.yaml
```

The camera YAML supplies `width`, `height`, and the flattened 3 × 3 intrinsic
matrix `k`. The width and height are parsed, while projection code uses the
matrix itself.

Depth values are multiplied by `0.001`, so all geometry after loading is in
metres. A color intensity image is converted to grayscale. Two intensity arrays
are retained:

- `intensity`: floating-point source values used by segmentation;
- `intensity8`: contrast-adjusted 8-bit values used by SAM 3 and debug images.

## 3. Depth-region proposals

Marker detection starts in [`sweep_region_proposals()`](src/depth_regions.py).
This stage produces image regions that might contain the calibration panel. It
does not estimate a pose.

### Boundary construction

[`discontinuities()`](src/depth_regions.py) marks pixels at:

- depth steps between valid four-connected neighbors;
- optional intensity steps;
- valid/invalid depth borders;
- depth steps found across short invalid-depth gaps after nearest-neighbor fill.

The boundary map is dilated slightly. Valid pixels outside the boundary map are
then split into connected components. Morphological opening removes small
artifacts but closing is not used, since it would reconnect surfaces across the
detected depth boundaries.

### Parameter sweep

No single depth/intensity threshold covers all views. The detector runs several
threshold combinations, each with dropout-border cutting enabled and disabled.
The combinations are defined by `SWEEP` and `SWEEP_DROPOUT_CUTS` in
[`depth_regions.py`](src/depth_regions.py).

Raised shapes on the panel can split it into nearby components. When enabled,
[`merge_segments()`](src/depth_regions.py) joins components that are close in the
image and have compatible mean depth.

Components outside the configured area range are removed. Near-identical masks
from different sweep variants are deduplicated by mask IoU. The largest remaining
regions are considered first, up to `sam3.max_regions_per_frame`.

## 4. SAM 3 mask refinement

[`run_pipeline()`](src/marker_detection_sam3.py) first checks the SAM 3 health
endpoint. Failure to contact the service stops marker detection; there is no
depth-only fallback.

For each region proposal:

1. The component centroid is used as the SAM 3 point prompt.
2. SAM 3 may return several masks.
3. [`_pick_best_mask()`](src/marker_detection_sam3.py) selects the best result
   using model score and overlap with the source region.
4. Masks that cover too little of the source region or grow too far beyond it
   are rejected using `sam3.min_prompt_cover` and `sam3.max_area_ratio`.
5. The selected mask contour is scored again for rectangular shape.
6. Center-distance non-maximum suppression removes duplicate hypotheses.

The detector retains at most `TOP_K` hypotheses for 3-D fitting. The optional
13-panel diagnostic output records the source images, segmentation, prompts,
SAM masks, rescored hypotheses, registration results, and best model projection.

## 5. Marker-model registration

[`verify_candidates()`](src/marker_registration.py) checks whether a 2-D
hypothesis is the physical marker rather than another rectangular surface.

### Model preparation

[`load_model_cloud()`](src/marker_registration.py) performs this work once and
caches the result:

1. Read the mesh from `model_path`.
2. Scale it by `model_units_to_m`.
3. Compute vertex normals.
4. Sample the mesh into a point cloud.
5. Remove bottom-facing model points that cannot be observed by the camera.
6. Downsample the cloud and compute FPFH features.
7. Record which model points belong to the raised relief.

### Candidate preparation and fitting

Each candidate bounding box is padded, lifted through the depth image and camera
intrinsics, and converted to a local point cloud. Candidates with too few usable
points are discarded.

The pose fit has two stages:

1. FPFH feature matching and RANSAC estimate a coarse rigid transform.
2. Point-to-plane ICP refines that transform.

RANSAC is stochastic, so each candidate can be attempted several times. The best
attempt is selected by the product of surface fitness and relief fitness.

### Acceptance

[`visible_fitness()`](src/marker_registration.py) evaluates only model points
whose normals face the camera. It calculates:

- `fitness`: matched fraction of all camera-facing model points;
- `relief_fitness`: matched fraction of camera-facing raised-detail points;
- `rmse`: ICP inlier error.

The relief check rejects flat surfaces that roughly match the panel outline. An
accepted candidate must pass all three thresholds: `MIN_FITNESS`,
`MIN_RELIEF_FITNESS`, and `MAX_RMSE_M`.

Candidate results are sorted by `fitness * relief_fitness`. Fits with nearby 3-D
marker centers are deduplicated. [`detect_marker()`](src/marker_detection_sam3.py)
returns the top result only if it is accepted. Its transform is
`T_marker_cam`, from marker-model coordinates to that frame's camera coordinates.

[`detect_all_markers()`](src/marker_based_reconstruction.py) stores accepted
frames in insertion order. A frame without an accepted marker is logged and does
not take part in reconstruction. If no frame is accepted, registration raises an
error.

## 6. Initial camera poses

[`build_initial_global_poses()`](src/marker_based_reconstruction.py) chooses the
first accepted frame as the reference. All frames observe the same stationary
marker, so each camera-to-reference transform is:

```text
T_cam_i_cam_ref = T_marker_cam_ref @ inverse(T_marker_cam_i)
```

These poses are used as initial estimates. Marker-fitting error is reduced in the
next stage using scene geometry.

[`make_initial_clouds()`](src/marker_based_reconstruction.py) also lifts every
valid depth pixel into a full-resolution point cloud. Each frame receives a
distinct uniform color. The colors are retained through merging and written to
the PLY files, which preserves the source-frame identity of each point.

## 7. Rigid registration refinement

Rigid refinement is implemented by
[`register_to_reference()`](src/marker_based_reconstruction.py).

### ICP inputs

[`preprocess_for_icp()`](src/marker_based_reconstruction.py) voxel-downsamples
each frame cloud and estimates normals. Full-resolution clouds are kept unchanged
until final transforms are available.

[`build_pose_graph()`](src/marker_based_reconstruction.py) considers pairs whose
frame-index distance is at most `pose_graph.pair_neighbor_span`. The marker poses
provide the initial relative transform for each pair.

[`pairwise_icp_registration()`](src/marker_based_reconstruction.py) runs:

1. coarse point-to-plane ICP with a wider correspondence distance;
2. fine point-to-plane ICP with a narrower distance;
3. a Tukey robust loss during the fine pass when `icp_tukey_k_m` is nonzero.

The robust loss reduces the weight of geometry that changed between captures,
including wind-driven branch movement. A pair is rejected if its fitness is below
`edge_min_fitness` or its RMSE exceeds `edge_max_rmse_m`.

Consecutive accepted pairs become certain pose-graph edges. Accepted pairs that
span more than one frame become uncertain loop-closure edges.

### Global optimization and fallback

[`optimize_pose_graph()`](src/marker_based_reconstruction.py) runs
Levenberg-Marquardt global optimization with node 0 fixed. The optimizer may prune weak
uncertain edges according to `prune_threshold`.

If no ICP edges survive, the full-resolution clouds are transformed with the
marker-derived poses. If global optimization throws an exception, the same
marker-only fallback is used. Otherwise, the optimized node poses transform the
full-resolution clouds into the reference frame.

The result of this stage is:

- one full-resolution cloud per accepted frame, already in the reference frame;
- one final camera-to-reference pose per accepted frame.

## 8. Marker removal and rigid output

Marker removal runs after rigid registration and before every downstream geometry
stage. The marker detections and poses remain available as metadata, but the
registered per-frame clouds passed to later stages no longer contain marker or
pole points.

When `marker_removal.enabled` is true,
[`remove_marker_and_pole()`](src/marker_based_reconstruction.py) creates cleaned
copies of the registered rigid clouds. All scene geometry is retained except
points inside the removal mask.

The mask is evaluated in marker-model coordinates for every accepted marker
pose. Taking the union across poses covers small differences left after rigid
registration. It contains two parts:

- the marker mesh bounding box expanded by `panel_padding_m`;
- a circular pole column centered below the panel, with `pole_radius_m` and
  `pole_length_m` controlling its size.

The vertical panel axis is selected by comparing its local X and Y axes with the
scene-up direction estimated from the cameras. The marker model's local Z axis is
the panel normal.

The cleaned full scene is written to `*_merged_marker_removed.ply` before tree
isolation and point filtering. The same cleaned per-frame clouds then feed both:

- [`save_reconstruction()`](src/marker_based_reconstruction.py), which produces
  the isolated and filtered rigid output;
- [`refine_clouds()`](src/non_rigid_refinement.py), when non-rigid refinement is
  enabled.

The rigid output is saved before optional non-rigid refinement. The non-rigid
branch does not replace the rigid files.

## 9. Target-tree isolation

[`isolate_target_tree()`](src/marker_based_reconstruction.py) works on all rigidly
registered points while retaining their per-frame grouping.

### Connected components

Open3D DBSCAN uses `tree_isolation.cluster_eps_m` and
`tree_isolation.cluster_min_points`. Clusters smaller than
`min_cluster_frac` times the largest cluster are excluded from target selection.
This removes small objects and most of the marker cloud from consideration.

For each remaining cluster, the code computes the mean angle between:

- each camera's optical axis (`R @ [0, 0, 1]`); and
- the direction from that camera center to the cluster centroid.

The cluster with the smallest mean angle is selected. Selection is based on
camera aim, not distance to the marker.

### Fused neighboring trees

DBSCAN cannot separate trees whose branches or support structures touch. When
`split_fused_trees` is enabled, [`_split_fused_tree()`](src/marker_based_reconstruction.py)
projects the selected cluster onto its widest horizontal axis and builds a
smoothed one-dimensional density histogram.

Deep density valleys split the cluster into segments. The segment best aligned
with the camera optical axes is retained. A split is ignored if the cloud is too
narrow, no qualified valley exists, or the proposed removed segment is smaller
than `split_min_lobe_frac`.

The keep mask is then divided back into per-frame clouds. This is required by the
multi-view filters that follow.

## 10. Visibility and support filtering

The visibility filter is selected first when both filters are enabled.

### Visibility filter

[`filter_by_visibility()`](src/marker_based_reconstruction.py) computes two
counts for every merged point.

**Confirmations** are frames with a point inside `confirm_radius_m` of the test
point in reference-frame 3-D space.

**Contradictions** are computed by projecting the point into a source camera:

1. Invert the final camera-to-reference pose.
2. Transform the point back to camera coordinates.
3. Project it through the camera intrinsics.
4. Read the nearest valid observed depth in a `pixel_margin` neighborhood.
5. Count a contradiction if the test point lies at least `tol_m` in front of
   the observed surface.

A point behind an observed surface is treated as occluded, not contradicted. A
point outside the image or too close to the camera is also neutral.

The exact keep rule is:

```text
confirmations >= min_confirmations
OR
contradictions < min_contradictions
```

This removes a point only when it lacks sufficient positive support and several
views independently observed free space at its position.

### Support filter

[`filter_by_frame_support()`](src/marker_based_reconstruction.py) is used only
when visibility filtering is disabled. It retains a point when at least
`support_filter.min_frames` source clouds contain a point within
`support_filter.radius_m`.

This filter is cheaper and simpler, but it can remove valid geometry observed by
only one frame.

## 11. Optional non-rigid branch

When `non_rigid.enabled` is true,
[`refine_clouds()`](src/non_rigid_refinement.py) starts from the same rigidly
registered per-frame clouds used by the rigid branch. The rigid output has already
been saved at this point.

### Deformation graph

Each source-frame cloud is downsampled to a work cloud. A second voxel sampling
creates deformation nodes. Every original and work point receives Gaussian
weights over its nearest nodes. Neighboring nodes are connected for the
smoothness term.

The deformation model is translation-only at the nodes:

```text
deformed_point = original_point + blend_weights @ node_displacements
```

It does not change the final camera poses and does not estimate local rotations.

### Mobility mask

A node is allowed to move only where three conditions overlap:

- low support from other frame clouds;
- a locally linear shape, measured from neighborhood covariance eigenvalues;
- low local point density.

The three scores are multiplied into a mobility value in `[0, 1]`. Nodes in the
lowest `bottom_anchor_fraction` of the scene receive zero mobility. Dense or
well-supported geometry is also strongly anchored.

### Correspondences and solve

For each movable work point, the algorithm finds one nearest candidate in every
other frame. Candidates inside the current correspondence radius form a median
consensus position. A correspondence is used only if:

- enough distinct frames contributed;
- point mobility exceeds `min_point_mobility`;
- consensus spread is below the configured fraction of the search radius.

The sparse least-squares system contains three terms:

- movement toward the frame-balanced consensus;
- smoothness between connected deformation nodes;
- a magnitude penalty that increases sharply as mobility approaches zero.

The solve runs once per coordinate axis. Node motion is capped by
`max_tip_displacement_m * mobility`.

Several passes run from wide to narrow correspondence radii. Each pass uses a
snapshot of all frame clouds, computes updates independently, and then applies
all updates together. This avoids making the result depend on frame iteration
order.

After deformation, the non-rigid per-frame clouds pass through the same target
isolation and visibility/support filtering code as the rigid branch.

## 12. Files written

[`output_path_for()`](src/marker_based_reconstruction.py) maps an input capture
to:

```text
<output_dir>/<side>/<capture>/<capture>_merged.ply
```

The rigid branch writes:

| File | Condition | Contents |
|---|---|---|
| `*_merged.ply` | Always | Final isolated and filtered rigid cloud |
| `*_merged.frames.txt` | Always | Input depth paths in contributing-frame order |
| `*_merged.poses.npz` | Poses available | Frame IDs, final poses, raw marker poses, and intrinsics |
| `*_merged_scene.ply` | Isolation removed points | Scene before isolation; marker and pole are absent when removal is enabled |
| `*_merged_prefilter.ply` | A filter ran | Isolated cloud before filtering |
| `*_merged_marker_removed.ply` | Marker removal enabled | Full rigid scene with only the marker and support pole removed |

With non-rigid refinement enabled, the equivalent files use the
`*_merged_nonrigid` stem. In addition,
`*_merged_nonrigid_displacements.npz` stores:

- `frame_ids`;
- `frame_offsets` into the concatenated point arrays;
- `displacement_vectors_m`;
- scalar `displacement_m`;
- `branch_tip_mobility`.

The displacement arrays describe the non-rigid input before target isolation and
filtering. Marker and pole points are absent when removal is enabled.
`frame_offsets[i]:frame_offsets[i + 1]` selects the points belonging to
`frame_ids[i]`.

## 13. Visualization points

Visualization is optional and blocking.

- `visualize_sam3` displays the 13 marker-detection panels and Open3D marker fit
  for every processed frame.
- `visualize_isolation` displays DBSCAN clusters and camera axes, the retained
  and removed portions of a fused cluster, and the along-row density profile.
- `visualize_reconstruction` displays the rigidly registered per-frame clouds.

The last view shows the rigid clouds even when non-rigid refinement is enabled.
The non-rigid result is written to disk but is not opened automatically.
