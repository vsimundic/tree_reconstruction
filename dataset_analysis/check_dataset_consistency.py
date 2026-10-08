"""Report-only scan of a dataset for pointcloud <-> image inconsistencies
within each view/subset (captures, marker_captures).

For every capture index found in either pointcloud/ or depth/, checks:
  - pointcloud/{idx}.ply exists
  - depth/{idx}_depth.png, {idx}_intensity.png, {idx}_heatmap.png,
    {idx}_camera_params.yaml exist
  - vertex count in the ply matches the count of valid (>0) pixels in the
    depth image

Nothing is modified. Writes the full report to a file (default:
<dataset>_consistency_report.txt next to the dataset root) and prints a
summary to stdout.

Usage:
    python check_dataset_consistency.py --root /workspace/data/pears
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

SUBSETS = ("captures", "marker_captures")
AUX = ("depth", "intensity", "heatmap", "camera_params")


def ply_vertex_count(path: Path) -> int:
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
    raise ValueError(f"no vertex count: {path}")


def count_valid_depth(path: Path) -> int:
    return int((np.array(Image.open(path)) > 0).sum())


def scan(root: Path):
    missing_ply = []          # image(s) exist, no matching ply
    missing_images = []       # ply exists, some image/yaml missing -> (tag, [missing kinds])
    count_mismatch = []       # (tag, n_pts, n_valid_px)
    ok = 0
    total_views = 0

    for group in ("A", "B"):
        g = root / group
        if not g.exists():
            continue
        for view in sorted(p for p in g.iterdir() if p.is_dir()):
            total_views += 1
            for sub in SUBSETS:
                sub_dir = view / sub
                pc_dir = sub_dir / "pointcloud"
                depth_dir = sub_dir / "depth"
                if not sub_dir.exists():
                    continue

                pc_idx = {p.stem for p in pc_dir.glob("*.ply")} if pc_dir.exists() else set()
                depth_idx = set()
                if depth_dir.exists():
                    for p in depth_dir.glob("*_depth.png"):
                        depth_idx.add(p.name[: -len("_depth.png")])

                tag_base = f"{group}/{view.name}/{sub}"

                for idx in sorted(depth_idx - pc_idx):
                    missing_ply.append(f"{tag_base}/{idx}")

                for idx in sorted(pc_idx):
                    missing_kinds = []
                    for kind in AUX:
                        ext = "yaml" if kind == "camera_params" else "png"
                        fpath = depth_dir / f"{idx}_{kind}.{ext}"
                        if not fpath.exists():
                            missing_kinds.append(kind)
                    if missing_kinds:
                        missing_images.append((f"{tag_base}/{idx}", missing_kinds))
                        continue  # can't count-check without the depth image

                    n_pts = ply_vertex_count(pc_dir / f"{idx}.ply")
                    n_px = count_valid_depth(depth_dir / f"{idx}_depth.png")
                    if n_pts != n_px:
                        count_mismatch.append((f"{tag_base}/{idx}", n_pts, n_px))
                    else:
                        ok += 1

    return {
        "total_views": total_views,
        "ok": ok,
        "missing_ply": missing_ply,
        "missing_images": missing_images,
        "count_mismatch": count_mismatch,
    }


def build_report(root: Path, res: dict) -> str:
    lines = [f"=== DATASET CONSISTENCY REPORT: {root} ===",
             f"views scanned                          : {res['total_views']}",
             f"captures OK (ply <-> images consistent): {res['ok']}",
             f"depth image with no matching ply       : {len(res['missing_ply'])}",
             f"ply with missing image/yaml file(s)    : {len(res['missing_images'])}",
             f"ply vertex count != valid pixel count  : {len(res['count_mismatch'])}",
             ""]

    def show(title, items, fmt=str):
        if not items:
            return
        lines.append(f"-- {title} ({len(items)}) --")
        for it in items:
            lines.append(f"    {fmt(it)}")
        lines.append("")

    show("depth image with no matching pointcloud ply", res["missing_ply"])
    show("pointcloud ply missing image/yaml file(s)", res["missing_images"],
         fmt=lambda t: f"{t[0]}  missing={t[1]}")
    show("vertex count vs valid-pixel count mismatch", res["count_mismatch"],
         fmt=lambda t: f"{t[0]}: {t[1]} pts vs {t[2]} valid px")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset root containing A/ and B/ (e.g. .../pears)")
    ap.add_argument("--report-file", default=None,
                    help="output path (default: <root>_consistency_report.txt)")
    args = ap.parse_args()

    root = Path(args.root)
    report_file = Path(args.report_file) if args.report_file else root.parent / f"{root.name}_consistency_report.txt"

    print(f"Scanning {root} ...")
    res = scan(root)
    report = build_report(root, res)

    report_file.write_text(report + "\n")

    print(f"views scanned                          : {res['total_views']}")
    print(f"captures OK                            : {res['ok']}")
    print(f"depth image with no matching ply       : {len(res['missing_ply'])}")
    print(f"ply with missing image/yaml file(s)    : {len(res['missing_images'])}")
    print(f"ply vertex count != valid pixel count  : {len(res['count_mismatch'])}")
    print(f"\nFull report written to {report_file}")


if __name__ == "__main__":
    main()
