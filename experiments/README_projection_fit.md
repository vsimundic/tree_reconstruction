# Final-cloud projection-fit evaluation

`evaluate_projection_fit.py` compares only two completed point clouds:

- the rigid `*_merged_scene.ply`; and
- the matching `*_merged_scene_nonrigid.ply`.

It does not score marker detection, ICP edges, deformation nodes, or other
intermediate results.

## How the score works

For each source camera, both final clouds are projected with the same saved pose
and intrinsics. A z-buffer keeps the closest projected surface. The default
evaluation excludes the colour block contributed by the camera being evaluated.
This avoids a trivial match caused by projecting a frame's own raw points back
into that frame.

The projected surface is compared with either:

- binary masks from `MASK_DIR`, named `<frame_id>_mask.png`; or
- the valid-depth pixels in each raw depth image when `MASK_DIR` is `None`.

Binary tree masks give true tree-silhouette IoU. The valid-depth fallback measures
fit to all observed surfaces and is appropriate for the full merged-scene cloud.
The JSON report records which source was used.

The reported measurements are:

- strict silhouette IoU, precision, recall, and F1;
- 3-pixel tolerant F1 and dilated IoU;
- depth-aware IoU at 10, 20, and 40 mm;
- median and 90th-percentile absolute depth error.

## Run

Edit the variables at the top of the script and run it from the repository root:

```bash
python src/tree_reconstruction/experiments/evaluate_projection_fit.py
```

There are no command-line arguments. If left as `None`, the non-rigid cloud,
pose sidecar, frame manifest, and output directory are inferred from `RIGID_PLY`.

## Outputs

The output directory contains:

- `rigid/frame_<id>.png`: rigid-only overlays and measurements;
- `nonrigid/frame_<id>.png`: non-rigid-only overlays and measurements;
- `comparison/frame_<id>.png`: complete side-by-side comparison;
- `rigid_all_frames.png` and `nonrigid_all_frames.png`;
- `rigid_vs_nonrigid_all_frames.png`;
- `metrics_comparison.png`;
- `projection_fit_metrics.csv`;
- `projection_fit_report.json`.

In the overlap images, green is a matching pixel, red is an extra projected
surface, and blue is an observed surface missing from the projection.
