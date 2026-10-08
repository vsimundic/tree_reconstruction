# Whole-image SAM comparison

`compare_sam_whole_image.py` is an experiment; it does not change the production
marker detector. It compares:

1. the current depth/intensity connected-region proposals followed by SAM point
   prompts; and
2. SAM automatic-style segmentation using a uniform prompt grid over the entire
   image, with no connected-region proposal stage.

Both options use the same rectangle scoring, NMS, and 3-D registration against
the known marker model. Manual marker positions are used only to measure and
draw the results, never as prompts or fallbacks.

Edit the variables near the top of the script, then run from the repository root:

```bash
python src/tree_reconstruction/experiments/compare_sam_whole_image.py
```

The output directory contains per-frame step images and side-by-side comparisons,
plus these overall files:

- `all_frames_2d_candidates.png`;
- `all_frames_3d_verdicts.png`;
- `summary_metrics.png`;
- `comparison_report.json`.

The whole-image method is intentionally expensive: with a 48 px grid on a
640 x 480 image it makes 130 SAM calls per frame. The prompt grid is independent
of depth and intensity content so it is a genuine alternative to connected
regions, not the same proposals under another name.
