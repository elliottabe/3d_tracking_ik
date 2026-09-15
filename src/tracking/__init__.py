"""Mask-free multi-camera fly tracking: video -> 3D keypoints -> articulated IK.

Stage packages: `io` (ordering contract, recordings, video, bout tables),
`geometry` (calibration and triangulation), `detector` (MVQ lifter and the
coarse/fine passes), `viz` (renders).
"""

__version__ = "0.1.0"
