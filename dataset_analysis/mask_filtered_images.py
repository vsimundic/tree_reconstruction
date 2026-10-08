"""Regenerate depth/intensity/heatmap images for the *filtered* pear dataset.

The filtered dataset (`pears_filtered`) has grass/trash points removed from the
per-view point clouds, but lacks the corresponding depth/intensity images. This
script reconstructs those images by:

  1. Loading the original per-view cloud O (whose point order equals the
     row-major order of valid, non-zero depth pixels).
  2. Loading the filtered cloud F and flipping it 180 deg about x (x, -y, -z) to
     bring it back into O's camera frame.
  3. Flagging each O point whose nearest-neighbour distance to F exceeds a
     threshold as "removed" (grass/trash), and zeroing those points' pixels in
     the depth/intensity/heatmap images (the row-major correspondence between O
     and the valid depth pixels is what lets the removed points be projected
     onto the images).
  4. Writing the filtered cloud back out flipped into O's coordinate frame, so
     output point clouds share the original dataset's orientation.

Output mirrors the original folder layout, dropping the `Filtered__noGrass`
wrapper but keeping `reconstruction_Voxel002`.

Run in two phases:
  * VALIDATION -- scan the whole dataset, pairing original and filtered captures,
    and report every inconsistency (missing views/subsets/clouds/images, vertex
    vs valid-pixel count mismatches). Nothing is written during this phase.
  * PROCESSING -- mask + write only the captures that passed validation.

Set MODE below to "validate" (scan + report only, nothing written) or
"process" (validate, then -- after confirmation unless SKIP_CONFIRM is
True -- mask and write the consistent captures; inconsistent ones are left
out).
"""
from __future__ import annotations
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODE = "process"  # "validate" or "process"
ORIG_ROOT = "/workspace/data/pears"
FILTERED_ROOT = "/workspace/data/pears_filtered"
OUT_ROOT = "/workspace/data/pears_filtered_clean"
THRESHOLD = 0.01          # NN distance (m) above which a point counts as removed
LIMIT = 0                 # validate/process at most N views per group (0 = all)
REPORT_FILE = "/workspace/data/pears_filtered_clean_validation_report.txt"        # e.g. "/workspace/data/pears_filtered_clean_validation_report.txt"
SKIP_CONFIRM = False       # process mode: skip the interactive confirmation prompt

# Filtered clouds are flipped 180 deg about the x-axis relative to the originals.
FLIP = np.array([1.0, -1.0, -1.0])
SUBSETS = ("captures", "marker_captures")
IMG_KINDS = ("depth", "intensity", "heatmap")
AUX_FILES = ("intensity", "heatmap")  # depth is handled explicitly
RECON_DIR = "reconstruction_Voxel002"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ply_vertex_count(path: Path) -> int:
    """Read the vertex count from a PLY header without parsing the body."""
    with open(path, "rb") as f:
        for _ in range(60):
            line = f.readline()
            if not line:
                break
            s = line.strip().lower()
            if s.startswith(b"element vertex"):
                return int(s.split()[-1])
            if s == b"end_header":
                break
    raise ValueError(f"no vertex count in PLY header: {path}")


def count_valid_depth(depth_path: Path) -> int:
    return int((np.array(Image.open(depth_path)) > 0).sum())


def load_points(path: Path) -> np.ndarray:
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def flip_cloud(pcd: o3d.geometry.PointCloud, flip: np.ndarray) -> o3d.geometry.PointCloud:
    """Apply the (x, -y, -z) flip in place, bringing a filtered cloud into the
    original cloud's coordinate frame. FLIP is a proper 180 deg rotation
    (det=+1), so normals transform the same way as points."""
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * flip)
    if pcd.has_normals():
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) * flip)
    return pcd


# --------------------------------------------------------------------------- #
# Capture description + validation
# --------------------------------------------------------------------------- #
@dataclass
class Capture:
    group: str
    view: str
    sub: str
    idx: str
    orig_sub: Path   # pears/<grp>/<view>/<sub>
    filt_sub: Path   # pears_filtered/<grp>/<view>/Filtered__noGrass/<sub>
    out_sub: Path    # <out>/<grp>/<view>/<sub>

    @property
    def tag(self) -> str:
        return f"{self.group}/{self.view}/{self.sub}/{self.idx}"


@dataclass
class Report:
    ok: list = field(default_factory=list)                 # list[Capture]
    filtered_view_no_original: list = field(default_factory=list)
    subset_only_in_filtered: list = field(default_factory=list)
    cloud_only_in_original: list = field(default_factory=list)  # orig ply, no filtered ply
    cloud_only_in_filtered: list = field(default_factory=list)  # filtered ply, no orig ply
    missing_depth_image: list = field(default_factory=list)
    missing_aux_image: list = field(default_factory=list)       # intensity/heatmap/camera_params
    count_mismatch: list = field(default_factory=list)          # (tag, n_pts, n_valid_px)
    missing_reconstruction: list = field(default_factory=list)
    orig_only_views: list = field(default_factory=list)         # informational

    def n_blocking(self) -> int:
        return sum(len(getattr(self, k)) for k in (
            "filtered_view_no_original", "subset_only_in_filtered",
            "cloud_only_in_filtered", "missing_depth_image",
            "count_mismatch", "missing_reconstruction"))

    def n_inconsistent(self) -> int:
        """All flagged items except purely-informational original-only views."""
        return sum(len(getattr(self, k)) for k in (
            "filtered_view_no_original", "subset_only_in_filtered",
            "cloud_only_in_filtered", "cloud_only_in_original",
            "missing_depth_image", "missing_aux_image",
            "count_mismatch", "missing_reconstruction"))


def ply_indices(directory: Path) -> set:
    if not directory.exists():
        return set()
    return {p.stem for p in directory.glob("*.ply")}


def validate(orig_root: Path, filt_root: Path, out_root: Path, limit: int = 0) -> Report:
    rep = Report()
    filt_wrap = "Filtered__noGrass"

    for group in ("A", "B"):
        og, fg = orig_root / group, filt_root / group
        if not fg.exists():
            continue
        # Informational: original views that have no filtered counterpart.
        if og.exists():
            filt_names = {p.name for p in fg.iterdir() if p.is_dir()}
            for ov in sorted(og.iterdir()):
                if ov.is_dir() and ov.name not in filt_names:
                    rep.orig_only_views.append(f"{group}/{ov.name}")

        filt_views = sorted(p for p in fg.iterdir() if p.is_dir())
        if limit:
            filt_views = filt_views[:limit]

        for fv in filt_views:
            view = fv.name
            ov = og / view
            if not ov.exists():
                rep.filtered_view_no_original.append(f"{group}/{view}")
                continue

            for sub in SUBSETS:
                orig_sub = ov / sub
                filt_sub = fv / filt_wrap / sub
                out_sub = out_root / group / view / sub

                if not filt_sub.exists():
                    continue  # this subset simply wasn't filtered; nothing to produce
                if not orig_sub.exists():
                    rep.subset_only_in_filtered.append(f"{group}/{view}/{sub}")
                    continue

                orig_idx = ply_indices(orig_sub / "pointcloud")
                filt_idx = ply_indices(filt_sub)

                for idx in sorted(orig_idx - filt_idx):
                    rep.cloud_only_in_original.append(f"{group}/{view}/{sub}/{idx}")
                for idx in sorted(filt_idx - orig_idx):
                    rep.cloud_only_in_filtered.append(f"{group}/{view}/{sub}/{idx}")

                for idx in sorted(orig_idx & filt_idx):
                    cap = Capture(group, view, sub, idx, orig_sub, filt_sub, out_sub)
                    depth_path = orig_sub / "depth" / f"{idx}_depth.png"
                    if not depth_path.exists():
                        rep.missing_depth_image.append(cap.tag)
                        continue
                    for kind in AUX_FILES:
                        if not (orig_sub / "depth" / f"{idx}_{kind}.png").exists():
                            rep.missing_aux_image.append(f"{cap.tag}:{kind}")
                    if not (orig_sub / "depth" / f"{idx}_camera_params.yaml").exists():
                        rep.missing_aux_image.append(f"{cap.tag}:camera_params")

                    n_pts = ply_vertex_count(orig_sub / "pointcloud" / f"{idx}.ply")
                    n_px = count_valid_depth(depth_path)
                    if n_pts != n_px:
                        rep.count_mismatch.append((cap.tag, n_pts, n_px))
                        continue

                    rep.ok.append(cap)

                if not (filt_sub / RECON_DIR).exists():
                    rep.missing_reconstruction.append(f"{group}/{view}/{sub}")

    return rep


def build_report(rep: Report, full: bool = False) -> str:
    """Render the validation report. full=True lists every item (for a file)."""
    lines = ["=== VALIDATION REPORT ===",
             f"captures OK to process        : {len(rep.ok)}",
             f"inconsistent captures/items   : {rep.n_inconsistent()}",
             f"blocking inconsistencies      : {rep.n_blocking()}"]

    def show(title, items, fmt=str, info=False):
        if not items:
            return
        label = "info" if info else "!!  "
        lines.append(f"{label} {title}: {len(items)}")
        shown = items if full else items[:8]
        for it in shown:
            lines.append(f"        {fmt(it)}")
        if not full and len(items) > 8:
            lines.append(f"        ... and {len(items) - 8} more (use --report-file for the full list)")

    show("filtered views with no original", rep.filtered_view_no_original)
    show("subsets only in filtered", rep.subset_only_in_filtered)
    show("filtered cloud with no original cloud (cannot make image)", rep.cloud_only_in_filtered)
    show("original depth image missing", rep.missing_depth_image)
    show("vertex/valid-pixel count mismatch", rep.count_mismatch,
         fmt=lambda t: f"{t[0]}: {t[1]} pts vs {t[2]} valid px")
    show("missing reconstruction_Voxel002", rep.missing_reconstruction)
    show("missing intensity/heatmap/camera_params (non-blocking)", rep.missing_aux_image)
    show("original cloud with no filtered cloud (not produced)", rep.cloud_only_in_original)
    show("original views with no filtered counterpart (skipped)", rep.orig_only_views, info=True)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #
def removed_mask(orig_pts: np.ndarray, filt_pts: np.ndarray, threshold: float) -> np.ndarray:
    filt_aligned = filt_pts * FLIP
    dist, _ = cKDTree(filt_aligned).query(orig_pts, k=1)
    return dist > threshold


def process_capture(cap: Capture, threshold: float, stats: dict) -> None:
    depth_dir = cap.orig_sub / "depth"
    depth_path = depth_dir / f"{cap.idx}_depth.png"
    depth = np.array(Image.open(depth_path))
    rows, cols = np.nonzero(depth > 0)

    O = load_points(cap.orig_sub / "pointcloud" / f"{cap.idx}.ply")
    filt_pcd = o3d.io.read_point_cloud(str(cap.filt_sub / f"{cap.idx}.ply"))
    F = np.asarray(filt_pcd.points)

    # Row-major ordering sanity check (validated broadly; guard per-file anyway).
    z_err = np.abs(depth[rows, cols] / 1000.0 - O[:, 2])
    if np.median(z_err) > 0.005:
        stats["skipped_bad_ordering"].append(cap.tag)
        return

    rm = removed_mask(O, F, threshold)
    frac = float(rm.mean())
    if frac > 0.9:
        stats["skipped_high_removal"].append((cap.tag, round(frac, 3)))
        return
    stats["removed_fracs"].append(frac)
    rr, rc = rows[rm], cols[rm]

    out_depth_dir = cap.out_sub / "depth"
    out_depth_dir.mkdir(parents=True, exist_ok=True)
    for kind in IMG_KINDS:
        src = depth_dir / f"{cap.idx}_{kind}.png"
        if not src.exists():
            continue
        im = Image.open(src)
        arr = np.array(im)
        if arr.ndim == 3:
            arr[rr, rc, :] = 0
        else:
            arr[rr, rc] = 0
        Image.fromarray(arr, mode=im.mode).save(out_depth_dir / f"{cap.idx}_{kind}.png")

    cam = depth_dir / f"{cap.idx}_camera_params.yaml"
    if cam.exists():
        shutil.copy2(cam, out_depth_dir / f"{cap.idx}_camera_params.yaml")

    out_pc_dir = cap.out_sub / "pointcloud"
    out_pc_dir.mkdir(parents=True, exist_ok=True)
    flip_cloud(filt_pcd, FLIP)
    o3d.io.write_point_cloud(str(out_pc_dir / f"{cap.idx}.ply"), filt_pcd, write_ascii=False)
    stats["processed"] += 1


def copy_reconstructions(rep: Report) -> None:
    """Copy reconstruction_Voxel002 for every processed subset, once."""
    done = set()
    for cap in rep.ok:
        key = (cap.group, cap.view, cap.sub)
        if key in done:
            continue
        done.add(key)
        src = cap.filt_sub / RECON_DIR
        dst = cap.out_sub / RECON_DIR
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)


def process(rep: Report, threshold: float) -> dict:
    stats = {"processed": 0, "removed_fracs": [],
             "skipped_bad_ordering": [], "skipped_high_removal": []}
    total = len(rep.ok)
    print(f"Processing {total} captures (threshold={threshold} m)...")
    for i, cap in enumerate(rep.ok, 1):
        process_capture(cap, threshold, stats)
        if i % 200 == 0 or i == total:
            print(f"  {i}/{total} captures, {stats['processed']} written")
    return stats


def print_stats(stats: dict) -> None:
    print("\n=== PROCESSING SUMMARY ===")
    print(f"captures written            : {stats['processed']}")
    if stats["removed_fracs"]:
        rf = np.array(stats["removed_fracs"])
        print(f"removed fraction per capture: median {np.median(rf):.3f}  mean {rf.mean():.3f}  "
              f"min {rf.min():.3f}  max {rf.max():.3f}")
    for key in ("skipped_bad_ordering", "skipped_high_removal"):
        if stats[key]:
            print(f"{key:26s}: {len(stats[key])}  e.g. {stats[key][:3]}")


# --------------------------------------------------------------------------- #
def main():
    orig_root, filt_root, out_root = Path(ORIG_ROOT), Path(FILTERED_ROOT), Path(OUT_ROOT)

    print("Scanning dataset (no files written)...")
    rep = validate(orig_root, filt_root, out_root, limit=LIMIT)
    print("\n" + build_report(rep))

    if REPORT_FILE:
        Path(REPORT_FILE).write_text(build_report(rep, full=True) + "\n")
        print(f"\nFull report written to {REPORT_FILE}")

    if MODE == "validate":
        return

    # process mode
    if rep.n_inconsistent() and not SKIP_CONFIRM:
        print(f"\n{rep.n_inconsistent()} inconsistent item(s) found. These captures will be "
              f"LEFT OUT; {len(rep.ok)} consistent captures will be processed.")
        try:
            ans = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            print("Aborted. No files written.")
            return

    stats = process(rep, THRESHOLD)
    copy_reconstructions(rep)
    print_stats(stats)


if __name__ == "__main__":
    main()
