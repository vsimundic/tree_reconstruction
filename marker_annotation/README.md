# Marker reference annotation

A local PySide6 desktop application. No backend or reconstruction dependencies.

From the repository root, with Python 3.10+:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r marker_annotation/requirements.txt
python -m marker_annotation
```

The start screen provides folder and session browsing, a marker profile or custom geometry, a folder filter, and a reviewer ID. These choices are remembered in `annotation_sessions/preferences.json` in the repository, including inside Docker. **Workspace…** reopens setup. An existing session restores its saved geometry and conventions; the folder and filter select the visible frames.

Choose a capture directory directly to work on it alone. To restrict a dataset to marker images, enter `marker_captures` in **Include folder names**. Comma-separated names include either folder; an empty filter includes everything. Names match exactly against image parent directories, including ancestors, so selecting `marker_captures/depth` directly also works. This only changes the displayed frame list; existing records outside it remain saved. The last frame is restored when reopening a session.

The session file is atomically autosaved after edits and on navigation/close. Reuse `--session` or Restore session to resume. JSON export includes all independent records; CSV summarizes their review status and fit. Keep the session on a writable local filesystem. Image data are never rewritten.

## Real pears capture profile

```sh
python -m marker_annotation --profile pears --dataset /media/simundic/T9/pears_dataset/pears_original --session /tmp/pears-annotations.json
```

This explicitly enables the user-authorized same-sensor pixel alignment assumption, axial Z depth at 0.001 m/unit, and the supplied physical CAD corner table `helios_corners.json`. Alignment is assumed, not independently calibrated. Without this profile, unknown conventions remain disabled. Existing restored sessions retain their saved conventions.

The table is extracted from `data/helios2DepthMarker.ply`, retaining its origin and axes. The plate spans X/Y = 0..230 mm and Z = 0..5 mm; relief reaches Z = 60 mm. Units follow the repository model scale of 0.001. Only 14 top-surface corners are selectable: 4 plate corners at Z = 5 mm, 4 cube corners at Z = 60 mm, and 6 L-shape corners at Z = 60 mm. Existing IDs are retained, including gaps in L numbering, to keep saved clicks attached to their physical points. Each corner stores its original `cad_vertex_index`. Saved sessions migrate once to the top-surface schema; removed lower/underside observations and affected fits are archived, and affected accepted records require a new estimate and review. Click only visible top corners; leave any other corners unmarked. Crease edges omit triangulation diagonals. Regenerate with:

```sh
python -m marker_annotation.import_cad data/helios2DepthMarker.ply marker_annotation/helios_corners.json
```

Depth was checked against paired row-major original point clouds for A/tree_0_V_0000 marker frames 0, 3, 7 (99,213 / 91,411 / 47,879 valid pixels). `PNG / 1000` differs from cloud Z by at most 0.75 mm; median absolute differences are 0.50 / 0.50 / 0.25 mm. Comparing to radial distance instead gives medians 126 / 129 / 83 mm. Zero pixels are omitted from the paired clouds. This supports axial Z in millimetres for this export, consistent with existing loader code. It is not a device-specific accuracy guarantee for every capture.

LUCID also documents the C channel as Z: [3D pixel formats](https://support.thinklucid.com/knowledgebase/pixel-formats-3d/). Raw device scale/offset conversion may vary, so no additional device offset is applied to these already exported PNGs.

## Input formats

The application recursively finds `<id>_intensity.png`, optional `<id>_depth.png`, and `<id>_camera_params.yaml` alongside them. This matches the supplied pears dataset. Calibration YAML uses ROS-style `k` (nine row-major camera matrix values), `d` (OpenCV distortion coefficients), and `distortion_model: plumb_bob`. Images must be raw pixels compatible with this calibration. Rectified images need their corresponding rectified camera matrix and zero distortion. Missing calibration permits annotation but prevents pose estimation.

The repository's previous annotator uses a 0.001 depth scale. **Outside the explicit pears profile, this application does not assume that scale, axial Z versus range, or intensity/depth alignment.** Enter a scale only after confirming axial Z semantics and invalid values (currently zero/nonfinite). Check pixel alignment only when calibration supports it. Unconfirmed cross-image observations are excluded with an explicit reason; a click in another modality replaces that corner's previous click. For independent modality annotation use separate repeat records. There is no invented registration between cameras.

The real mesh is now supplied in `data/helios2DepthMarker.ply` and its stable corner table is included. For another marker, supply a JSON table with actual CAD coordinates and stable IDs:

```json
{
  "name": "Physical marker revision",
  "units": "m",
  "axes": "Describe the CAD origin and positive axes here",
  "corners": {
    "P0": {"group": "plate", "xyz": [0, 0, 0]},
    "L0": {"group": "L-shape", "xyz": [0.02, 0.03, 0.04]},
    "C0": {"group": "cube", "xyz": [0.1, 0.1, 0.05]}
  },
  "edges": [["P0", "L0"], ["L0", "C0"]]
}
```

The illustrative table above is not enough to solve a pose. Supply at least four well-spread actual corners. Calibration and geometry definitions, as well as their original paths, are embedded in saved sessions.

## Synthetic demonstration

`example/` is an explicitly fictional plate/L/cube corner model with three rendered intensity frames and entirely invalid depth. It is not the physical pears marker. Regenerate with `python -m marker_annotation.synthetic`.

```sh
python -m marker_annotation --dataset marker_annotation/example --geometry marker_annotation/example/geometry.json --session /tmp/synthetic-session.json
```

Select a corner in the diagram or list, click it, estimate, inspect residuals, then accept or flag ambiguous. Only click corners you can see. Leave all other corners unmarked; no visibility classification is required. New independent repeat starts a blank record for the current frame and hides earlier clicks and fits; exported records remain separate. Repeat records are completed by Accept or Flag ambiguous. No automatic detections are imported.

## Interaction

- Wheel: zoom about cursor; drag: pan; Fit image or F: fit.
- Click: place selected corner; Shift-drag: reposition it.
- Delete: remove click; Ctrl+Z / Ctrl+Shift+Z: undo/redo observation edits.
- Left/Right: previous/next frame; Up/Down in corner list: select corner.
- E: estimate; A: accept. Text entry retains normal text editing shortcuts.
- Corners are either clicked or unmarked. Delete removes a click; undo restores it.
- Diagram: use −90° / +90°, type any clockwise angle, toggle Mirror (left/right after rotation), or Reset. Orientation is remembered in the session and applies to the whole diagram. This affects only the diagram; IDs, CAD coordinates, image clicks, poses, and acceptance are unchanged.

All annotations use original image pixels. Depth normalization creates a separate display image; raw measurements are preserved. Invalid depth is muted purple. The selected corner's depth check reports invalid patches and discontinuities.

## Pose and review conventions

`X_camera = Rodrigues(rvec) @ X_marker + tvec_m`. Translation and CAD coordinates are metres. Camera axes are OpenCV optical axes: +X right, +Y down, +Z forward; CAD axes/origin come from the geometry file. Pixel coordinates have top-left origin, +u right, +v down. The diagram is a CAD plan view (+X right, +Y up) showing all 14 top corners together, not a camera view. For the real marker, the cube and L-shape top corners have Z = 60 mm and the plate top corners have Z = 5 mm. Lower and underside corners are excluded. Selecting a corner highlights it in the shared view; its actual Z height remains in the details and corner list. Spaced, clickable ID labels connect to exact corner locations; the list and selected-corner details also show height and click status.

Visible clicked corners in the active image (or confirmed aligned images) enter the fit. Unmarked corners are omitted. Legacy non-visible observations are preserved for compatibility, remain excluded from estimation, and are not shown as image clicks; a new click replaces one with a visible observation. Other excluded observations retain explicit reasons. There is no silent robust rejection. Planar sets use IPPE; nonplanar sets use SQPnP. Collinear/coincident sets and behind-camera solutions are rejected. Competing candidates within 0.5 px RMSE are marked ambiguous and cannot be accepted; add reliable nonplanar corners or flag for review. All candidates and solver thresholds are exported. The application conservatively reports depth consistency rather than automatically using depth to choose a competing solution.

Depth checks require confirmed units and alignment (or active depth view), a fully valid 5×5 patch, and no patch spread greater than 20 mm. Median axial depth is compared with predicted camera Z; edge depth is never silently used. Depth is optional and does not change the fit. Reprojection RMSE measures numerical fit only; acceptance is a separate explicit reviewer action. No fit threshold automatically accepts a record. Editing observations invalidates the pose and acceptance; undo does not reinstate acceptance. Changing depth/alignment settings or geometry also invalidates fits.

## Verification

```sh
pip install pytest
QT_QPA_PLATFORM=offscreen python -m pytest marker_annotation/tests -q
```

Tests cover original-pixel view transforms, atomic JSON round trips, synthetic nonplanar recovery, planar/degenerate handling, invalid depth/edges, and acceptance invalidation. Offscreen GUI launch and restoration were also checked against synthetic data; real 640×480 uint16 intensity/depth loading was checked on the supplied capture. Physical-marker accuracy still depends on correct clicks, calibration, and the stated alignment assumption.

## Docker

The repository Docker image includes the annotation dependencies. From a Linux
X11/XWayland desktop, run `bash docker/build.sh`, then `bash docker/run.sh`
with the correct host data mount. Inside the container, run:

```bash
python -m marker_annotation
```

Choose `/data/pears_original`, the Pears profile, and a session under `/workspace/annotation_sessions` in the start screen. Command-line options remain available for scripted launches. See the [project README](../README.md) for setup and dataset paths. Sessions
under `/workspace/annotation_sessions` persist in the host repository.
