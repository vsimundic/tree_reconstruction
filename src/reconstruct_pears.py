"""Batch driver for tree_reconstruction.reconstruct_capture: reconstruct every
capture found under config.dataset_dir. A failing capture is reported and the
batch continues. All settings come from cfg/default.yaml (see config.py).
"""

import glob
import os

import config
import tree_reconstruction as tr


def is_frame_dir(path):
    """True if `path` directly holds *_depth.png frames (a capture's frame dir)."""
    return os.path.isdir(path) and bool(glob.glob(os.path.join(path, "*_depth.png")))


def find_frame_dirs(path):
    """Capture frame dirs (.../marker_captures/depth) at or under `path`, which may
    be a frame dir, a single capture dir, or a dataset/side root."""
    if is_frame_dir(path):                                   # already a frame dir
        return [path]
    capture_frame_dir = os.path.join(path, "marker_captures", "depth")
    if is_frame_dir(capture_frame_dir):                      # a single capture dir
        return [capture_frame_dir]
    hits = glob.glob(os.path.join(path, "**", "marker_captures", "depth"),
                     recursive=True)                         # a dataset/side root
    return sorted(fd for fd in hits if is_frame_dir(fd))


def main():
    path = config.CFG.dataset_dir
    frame_dirs = find_frame_dirs(path)
    if not frame_dirs:
        raise SystemExit(f"no captures (…/marker_captures/depth) found under {path}")

    print(f"{len(frame_dirs)} capture(s) to reconstruct under {path}")
    ok = skipped = failed = 0
    for i, frame_dir in enumerate(frame_dirs, 1):
        out_path = tr.output_path_for(frame_dir)
        print(f"\n[{i}/{len(frame_dirs)}] {frame_dir}")
        if os.path.exists(out_path) and not config.CFG.overwrite:
            print(f"  already reconstructed -> {out_path} (set overwrite: true to redo)")
            skipped += 1
            continue
        try:
            tr.reconstruct_capture(
                frame_dir,
                visualize_sam3=config.CFG.visualize_sam3,
                visualize_reconstruction=config.CFG.visualize_reconstruction,
                visualize_isolation=config.CFG.visualize_isolation)
            ok += 1
        except Exception as e:  # one bad capture shouldn't stop the batch
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\ndone: {ok} reconstructed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
