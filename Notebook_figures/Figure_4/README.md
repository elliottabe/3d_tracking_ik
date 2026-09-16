# Figure 4

Builds the courtship Figure 4 from `ik_output_combined_v1_pose_v2_20260914`.

```bash
conda activate 3d_tracking_ik
pip install -e "../..[figures]"        # scikit-learn + mujoco_visualizer
export PYTHONPATH=$PWD/../../src MUJOCO_GL=egl

python make_figure4.py                 # all ten panels -> fig4.svg / fig4.png
python make_figure4.py --skip-assets   # panels B-G only; no GPU needed
python make_figure4.py --refresh       # ignore the cache and recompute
```

Needs a GPU node for panel H's MuJoCo renders. Never run it on the login node.
A cold run is about 12 minutes; with the cache warm, restyling is seconds.

## Files

| | |
|---|---|
| `make_figure4.py` | the script — inputs at the top, three stages |
| `fig4_analysis.py` | song / locomotion / pulse analysis, AST-extracted (82 defs) |
| `fig4_assets.py` | video crops, MuJoCo pair render, SAM3 resolution (39 defs) |
| `fig4_panels.py` | the 9 panel plotters + the spec→plotter adapters |
| `fig4_layout.py` | panel rects, styles and letters, generated from `fig4.json` |
| `floor_happy_house.xml` | the render arena (the shared `floor.xml` with its geom enabled) |
| `cache/` | the computed panel arrays and rasters, one h5 per input |

`fig4_analysis.py` and `fig4_assets.py` are **extracted, not rewritten** — every
function body is byte-identical to the source. A reimplementation could change
the science with no figure revealing it. If you edit either, verify it against a
dataset whose answer is already known; an extraction can silently drop a branch
that fires on only some bouts, and the figure will still look plausible.

## Panels

| | Panel | Source |
|---|---|---|
| A | video strip ×4 | mp4 + DLT calibration + SAM3 masks + kp3d |
| B | sine in-phase | exemplar bout |
| C | wing z + song shading | exemplar bout |
| D | wing phase polar | pooled |
| E | wing angle density | pooled |
| F | Pslow / Pfast | pooled (PCA + GMM) |
| G | body height | pooled + the free-running h5 |
| H | MuJoCo renders ×4 | qpos + body model + arena |
| I | male vs target pitch | qpos + SAM3-triangulated female COM |
| J | pitch-alignment violin | pooled over every bout with masks |

