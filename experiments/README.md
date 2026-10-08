# Non-rigid tree-registration experiment

`non_rigid_registration.py` is an optional post-processing experiment. It reads
the full `*_merged_scene.ply` reconstruction and does not import or change the
active tree reconstruction pipeline.

Set `INPUT_PLY` in the configuration variables near the top of the script. You
can also change `POSES_PATH`, `OUTPUT_DIR`, or any tuning value there. Then run it
without arguments from the repository root:

```bash
python src/tree_reconstruction/experiments/non_rigid_registration.py
```

The script automatically pairs `*_merged_scene.ply` with its
`*_merged.poses.npz` sidecar. The scene PLY must still have the uniform per-frame
colours written by `tree_reconstruction.py`.

The experiment anchors stable geometry in every frame. A deformation node can
move only when it has low agreement across frames, looks locally like a thin line,
and has the low density expected near a branch end. Eligible tips are aligned to
the median position supplied by the other frames in coarse-to-fine passes. The
trunk, marker surfaces, poles, dense branch junctions, and lower scene stay fixed.
`MAX_TIP_DISPLACEMENT_M` provides a hard safety limit.

By default, results are written to `non_rigid_comparison/` beside the input:

- `*_merged_scene_nonrigid.ply`: the refined full-scene point cloud;
- `*_nonrigid_displacements.npz`: the movement of every point in metres;
- `*_nonrigid_report.json`: settings and before/after measurements;
- `*_comparison_front.png`: rigid and non-rigid front views;
- `*_comparison_side.png`: rigid and non-rigid side views;
- `*_movement_front.png`: movement magnitude shown on the refined scene;
- `*_mobility_front.png`: points the branch-tip detector allowed to move.

This is deliberately separate from the production pipeline. Inspect the PNGs
and report before deciding whether a result is geometrically trustworthy.
