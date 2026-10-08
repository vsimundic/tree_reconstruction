# Tree Reconstruction

Turn a handful of depth + intensity photos taken around a single fruit tree into
one clean 3-D point cloud of **just that tree**.

Everything described here is driven by
[`tree_reconstruction.py`](tree_reconstruction.py). The entry point is
`reconstruct_capture()`, which processes one capture from start to finish.

---

## The idea in one paragraph

We photograph a tree from several angles. Each photo (a "frame") is a depth image
plus an intensity image from the same depth camera. On its own, each frame is a
partial 3-D view of the tree from one side. To fuse them into a complete cloud we
need to know exactly where each camera was standing — its **pose**. We get that in
two steps. First, every frame contains the same physical **calibration marker**
placed in the scene, so detecting the marker gives us a rough pose for each frame.
Second, we sharpen those rough poses by matching the actual overlapping 3-D
geometry between frames. Once the poses are good, merging is just stacking the
clouds together. After merging we remove the neighboring trees and the noise, and
save the result.

---

## What goes in

A **capture** is one directory (the config calls it `frame_dir`), ending in
`.../marker_captures/depth`. Inside it, each frame has three files named by a
shared id:

| File | What it is |
|---|---|
| `<id>_depth.png` | Depth image (converted to metres when loaded). |
| `<id>_intensity.png` | Grayscale brightness image from the same camera. |
| `<id>_camera_params.yaml` | The camera's intrinsics `K` (focal length, principal point). |

All frames look at the **same tree** and the **same marker** — a flat panel with a
raised cube/L pattern on it. The raised pattern matters later: it is what tells a
real marker apart from a flat patch of ground.

## What comes out

Written under `<output_dir>/<side>/<capture>/`:

| File | What it is |
|---|---|
| `<capture>_merged.ply` | **The main output** — the single, isolated, filtered tree. |
| `<capture>_merged.frames.txt` | The depth-frame paths that contributed, in order (reference first). |
| `<capture>_merged.poses.npz` | Each frame's final pose, raw marker pose, and `K`, so you can reproject later without re-detecting anything. |
| `<capture>_merged_scene.ply` | Snapshot of the **full scene** with every tree — written only when isolation actually dropped something, for before/after comparison. |
| `<capture>_merged_prefilter.ply` | Snapshot of the isolated tree **before filtering** — written only when a filter ran. |

Each frame's points are painted a distinct color, so in the merged cloud you can
see which frame contributed which points.

---

## A note on how poses are written

Poses are 4×4 rigid transforms. The naming convention throughout the code is:

> `T_A_B` maps a point from frame **A** into frame **B**, i.e. `p_B = T_A_B @ p_A`.

So `T_marker_cam` is the marker's pose expressed in a frame's camera. The
**reference frame** is the first frame whose marker was accepted; every other
frame's pose is expressed relative to it.

---

## The pipeline, step by step

At a glance:

1. **Detect the marker** in every frame.
2. **Give each frame a rough pose** by chaining through the shared marker.
3. **Sharpen the poses** by matching overlapping 3-D geometry (ICP + pose graph).
4. **Merge** every frame into one cloud.
5. **Isolate the target tree** and drop the neighbors.
6. **Filter** out ghost points and noise.
7. **Save** the cloud and its sidecar files.

### 1. Detect the marker in every frame

Function: `detect_all_markers()`, which calls `detect_marker()` from
[`marker_detection_sam3.py`](marker_detection_sam3.py).

For each frame we need the marker's full 3-D pose, `T_marker_cam`. Finding it has
two halves:

- **Find the marker in 2-D.** The intensity/depth image is segmented into
  candidate flat rectangular regions. Each candidate's center is handed to
  **SAM 3** (an image-segmentation model running as a separate service — see
  [`sam3_client.py`](sam3_client.py)), which returns a clean mask of that shape.
  Masks are scored as rectangle hypotheses and near-duplicates are removed.
- **Confirm it in 3-D.** For each surviving 2-D candidate we lift its pixels to a
  small 3-D point cloud and fit the **known 3-D model of the marker** to it
  (a coarse feature-based alignment, then fine ICP). See
  [`marker_registration.py`](marker_registration.py).

A fit is **accepted** only if enough of the camera-facing marker surface matches
**and** enough of its raised relief matches, with a low enough error. Requiring
the relief is the trick that rejects impostors: flat ground can imitate the panel
but not the bumps, and noisy vegetation matches neither.

Frames with no accepted marker are simply skipped. The rest give us a
`T_marker_cam` each.

### 2. Give each frame a rough pose

Function: `build_initial_global_poses()`.

The first accepted frame becomes the **reference**; its pose is the identity
(it defines the coordinate system). Because every frame sees the *same* physical
marker, we can hop from one camera to another through the marker:

```
T_cam_i_cam_ref = T_marker_cam_ref @ inv(T_marker_cam_i)
```

This gives each frame a starting pose relative to the reference. It is only as good
as the marker detections — good enough to get the frames roughly on top of each
other, not yet good enough to merge cleanly.

At this point we also build a **full-resolution point cloud** for every frame
(`make_initial_clouds()`), each painted its own color.

### 3. Sharpen the poses

Function: `register_to_reference()`.

The rough marker poses are refined by looking at the actual tree geometry the
frames share.

- **Prepare** (`preprocess_for_icp()`): downsample each cloud and estimate surface
  normals, which point-to-plane ICP needs.
- **Match nearby frame pairs** (`build_pose_graph()` + `pairwise_icp_registration()`):
  for every pair of frames close together in the sequence, run **ICP** to refine
  their relative pose. ICP nudges one cloud until its surfaces sit on the other's.
  It runs a coarse pass, then a fine pass with a **robust weighting** so that
  branches that *moved* between shots (a gust of wind) are ignored and only the
  static structure drives the alignment. A pair is kept as a **constraint** only
  if the match is good enough (high overlap, low error); weak matches are dropped.
- **Solve everything at once** (`optimize_pose_graph()`): all the kept
  pairwise constraints go into a **pose graph** — think of the frames as nodes and
  each good match as a spring pulling two nodes into agreement. A global optimizer
  settles all the springs together, keeping the reference frame pinned in place.
  This spreads out small errors instead of letting them pile up along the chain.

**Fallbacks.** If no pairwise match survives, or the optimizer fails, the pipeline
falls back to the rough marker-only poses from step 2 rather than giving up.

The output is every cloud placed into the reference frame, plus the final pose of
each frame (kept because the visibility filter later needs to know where each
camera stood).

### 4. Merge

Function: `merge_clouds()`.

Now that every cloud is in the same coordinate frame, merging is just adding them
together into one point cloud.

### 5. Isolate the target tree

Function: `isolate_target_tree()`.

A capture often includes **more than one tree** — trees in a row stand close
together. We only want the one the cameras were actually pointed at. (The marker
can't tell us which one; it might be sitting at a neighbor's base.)

- **Cluster** the merged cloud into connected blobs (DBSCAN). Points closer than a
  set distance land in the same blob, so separate trees become separate clusters.
- **Pick the target**: among the substantial clusters (ignoring tiny ones like the
  marker's own points and stray specks), keep the one the cameras aim at most
  directly — the smallest average angle between each camera's viewing direction and
  the direction to the cluster's center.
- **Split fused neighbors** (`_split_fused_tree()`): if a neighbor tree is
  physically touching the target (overlapping branches, a shared trellis) they end
  up in one cluster. We look at how the points are spread out *along the row*: two
  trees show up as two humps with a valley between their trunks. If there's a deep
  enough valley, we cut there and keep the side the cameras face. A lone tree is a
  single hump, so it is never cut.

The result is a cloud of just the target tree. (If clustering finds nothing usable,
the cloud is left as-is.)

### 6. Filter out ghosts and noise

Two filters are available; the **visibility filter** takes precedence and is the
default.

**Visibility (free-space) filter** — `filter_by_visibility()`. This uses the fact
that we know where every camera stood and what its depth image saw. For each point
in the merged cloud we tally two things:

- **Confirmations**: how many frames actually observed a surface at that spot.
- **Contradictions**: how many frames looked *straight through* that spot into empty
  space behind it (their depth image says there was nothing there).

A point is **kept** if enough frames confirm it, **or** if too few frames
contradict it. In plain terms: a real branch that even a couple of views captured
survives no matter how many other views happened to miss it, while a point that
almost nobody saw *and* that many views see straight through — a ghost of a swayed
branch, or floating speckle — gets removed.

**Support filter** — `filter_by_frame_support()`. A simpler alternative: keep a
point only if several different frames put a point near it. It is off by default
because it can throw away a real branch that only one camera happened to see.

### 7. Save

Function: `save_reconstruction()`.

Writes the main `.ply`, the two sidecar files (the list of contributing frames and
the per-frame poses), and — for before/after inspection — the full-scene and
pre-filter snapshots described in [What comes out](#what-comes-out).

---

## Configuration

There are no command-line flags. Every knob lives in
[`cfg/default.yaml`](cfg/default.yaml): dataset paths, the marker model, the
ICP/pose-graph settings, tree isolation, and the two filters. Each setting has an
inline comment explaining it. Point the pipeline at a different config file with
the `TREE_CFG` environment variable.

## Running it

```bash
cd src/tree_reconstruction

# Reconstruct the single capture named by `frame_dir` in the config
python tree_reconstruction.py

# Reconstruct every capture found under `dataset_dir`
python reconstruct_pears.py
```

Three toggles in the config open blocking visualization windows so you can watch
what happened:

- `visualize_sam3` — the marker-detection steps, per frame.
- `visualize_isolation` — how the target tree was chosen and split from its
  neighbors, including the along-row density used for the cut.
- `visualize_reconstruction` — the final merged cloud in a 3-D viewer.

---

## Where each step lives

| Step | Key functions in [`tree_reconstruction.py`](tree_reconstruction.py) |
|---|---|
| Detect the marker | `detect_all_markers` (→ [`marker_detection_sam3.py`](marker_detection_sam3.py), [`marker_registration.py`](marker_registration.py)) |
| Rough poses from the marker | `build_initial_global_poses`, `make_initial_clouds` |
| Sharpen poses (ICP + pose graph) | `register_to_reference`, `build_pose_graph`, `pairwise_icp_registration`, `optimize_pose_graph` |
| Merge | `merge_clouds` |
| Isolate the target tree | `isolate_target_tree`, `_split_fused_tree` |
| Filter | `filter_by_visibility`, `filter_by_frame_support` |
| Save | `save_reconstruction`, `save_frame_manifest`, `save_frame_poses` |
| Orchestration | `reconstruct_capture` |

This is **Stage 1** of the wider PEAR project; see the
[top-level README](../../README.md) for how it fits with pruning-point detection.
