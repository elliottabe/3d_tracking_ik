"""Courtship analysis for Figure 4, extracted from the source repo.

An AST trace of what the figure actually executes across
`3d_tracking_dataset/utils/`: 82 definitions, bodies byte-identical to the
source. Extracted rather than rewritten because a reimplementation could
change the science with no figure revealing it.

Provenance (source lines -> extracted):

    song_analysis      1634 -> 1456   wing angles, pulse/sine detection
    keypoint_filter    1404 ->  312   wing-tip identity repair, despiking
    courtship_loader    778 ->  364   the per-pair driver
    pair_validity       424 ->  281   per-fly and colocation validity
    mvq_pitch_alignment 387 ->  336   panel J's per-bout pitch alignment
    io_dict_to_hdf5     290 ->   86   h5 -> nested dict
    locomotion          286 ->  221   centroid velocity, COM height, walking
    pulse_types         205 ->  127   Pslow/Pfast PCA + GMM
    sam3_female_com     178 ->  149   female COM from SAM3 masks
    pulse_type_cache    153 ->  116   pooled pulse-type labelling
    sex_id              140 ->  116   male/female from song + body length
    stac_data_utils     787 ->    7   bout key ordering

`sex_id` is kept even though the combined h5 carries `info/sex`: it still
runs, and `make_figure4.py` checks its verdict against the h5 rather than
trusting either one silently.
"""
from __future__ import annotations

# The union of what the source modules imported at module level. jax and
# omegaconf are deliberately absent: the traced functions reach them only on
# branches this figure does not take (`enable_jax`, DictConfig configs), and
# importing jax here would pull a GPU context into a plotting script.
import dataclasses
import fnmatch
import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import (butter, filtfilt, find_peaks, hilbert, savgol_filter,
                          spectrogram as sp_spectrogram)
from scipy.signal.windows import dpss
from scipy.stats import f as _f_dist
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture



# ======================================================================
# from utils/io_dict_to_hdf5.py
# ======================================================================


def convert_dict_to_list_if_appropriate(d):
    """Convert a dictionary with string integer keys back to a list if appropriate"""
    if not isinstance(d, dict):
        return d
    
    # Check if all keys can be converted to integers
    try:
        keys_as_ints = [int(k) for k in d.keys()]
        if sorted(keys_as_ints) == list(range(len(keys_as_ints))):
            # Keys are consecutive integers starting from 0, convert to list
            result = [None] * len(d)
            for k, v in d.items():
                result[int(k)] = v
            return result
    except (ValueError, TypeError):
        pass
    
    return d

def recursively_convert_appropriate_dicts_to_lists(data):
    """Recursively convert dictionaries that should be lists back to lists"""
    if isinstance(data, dict):
        # First, recursively process all values
        processed_data = {k: recursively_convert_appropriate_dicts_to_lists(v) for k, v in data.items()}
        # Then check if this dictionary should be converted to a list
        return convert_dict_to_list_if_appropriate(processed_data)
    elif isinstance(data, list):
        return [recursively_convert_appropriate_dicts_to_lists(item) for item in data]
    else:
        return data

def load(filename, ASLIST=False, enable_jax=False, auto_convert_lists=True):
    """
    Default: load a hdf5 file (saved with io_dict_to_hdf5.save function above) as a hierarchical
    python dictionary (as described in the doc_string of io_dict_to_hdf5.save).
    
    Parameters:
    - ASLIST: if True, loads the top level as a list (requires integer convertible keys)
    - enable_jax: if True, converts numeric data to JAX arrays while preserving strings
    - auto_convert_lists: if True, automatically detects and converts dictionaries that 
      were originally lists back to lists (based on consecutive integer string keys)
    
    Both ASLIST and enable_jax can be used together - JAX conversion will be applied to appropriate data types
    while maintaining list structure for containers.
    """
    with h5py.File(filename, 'r') as h5file:
        out = recursively_load_dict_contents_from_group(h5file, '/', enable_jax=enable_jax)
        
        # Apply automatic list conversion if requested
        if auto_convert_lists:
            out = recursively_convert_appropriate_dicts_to_lists(out)
        
        # Apply top-level ASLIST conversion if requested
        if ASLIST:
            outl = [None for l in range(len(out.keys()))]
            for key, item in out.items():
                outl[int(key)] = item
            out = outl
            
        return out

def recursively_load_dict_contents_from_group(h5file, path, enable_jax=False):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            data = item[()]
            
            # Handle string/bytes conversion
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            elif isinstance(data, np.ndarray) and data.dtype.kind in ['S', 'U']:
                # Handle numpy string arrays
                if data.dtype.kind == 'S':  # byte strings
                    data = np.array([s.decode('utf-8') if isinstance(s, bytes) else s for s in data.flat]).reshape(data.shape)
                # Unicode strings (dtype.kind == 'U') are already handled correctly
            
            # Only convert to JAX if enable_jax is True AND it's not string data
            if enable_jax and isinstance(data, np.ndarray) and data.dtype.kind not in ['S', 'U']:
                import jax.numpy as jnp   # imported here: see this file's header
                ans[key] = jnp.asarray(data)
            elif enable_jax and isinstance(data, (int, float)) and not isinstance(data, (str, bytes)):
                import jax.numpy as jnp
                ans[key] = jnp.asarray(data)
            else:
                ans[key] = data
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/', enable_jax=enable_jax)
    return ans

# ======================================================================
# from utils/stac_data_utils.py
# ======================================================================


def sorted_bout_keys(keys) -> list:
    """Bout keys in NUMERIC order (bout_2 < bout_10 < bout_100), regardless
    of zero-pad width. Non-numeric suffixes sort after, alphabetically."""
    def _key(k):
        m = re.search(r'(\d+)\s*$', k)
        return (0, int(m.group(1)), k) if m else (1, 0, k)
    return sorted(keys, key=_key)

# ======================================================================
# from utils/keypoint_filter.py
# ======================================================================


def despike_isolated_spikes(
    arr: np.ndarray,
    threshold_factor: float = 10.0,
    max_iterations: int = 1,
    verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """Remove tracking glitches via velocity-reversal detection.

    Each pass finds frames where the jump in is large, the jump out is
    large, and the two jumps have opposite signs (immediate reversal).
    Flagged frames are replaced with the average of their neighbours.

    With ``max_iterations=1`` (default), only true single-frame spikes are
    fixed — safe for signals with fast oscillations like male wing song.

    With ``max_iterations>1``, multi-frame glitches are peeled from the
    outside in: a 3-frame spike becomes a 2-frame spike after pass 1,
    then a 1-frame spike after pass 2, fully fixed by pass 3.  Use higher
    values only for signals where multi-frame tracking errors are expected
    and real fast oscillations are absent (e.g. non-singing flies).

    Works on arrays of any shape whose first axis is time:
    ``(T,)``, ``(T, D)``, ``(T, N, 3)``, etc.

    Parameters
    ----------
    arr : ndarray
        Input array.  First axis is time.
    threshold_factor : float
        A frame is flagged when its inward *and* outward velocity both
        exceed ``threshold_factor × median(|diff|)`` for that signal.
    max_iterations : int
        Number of passes.  1 = single-frame only (conservative, safe for
        song).  Higher values peel multi-frame glitches layer by layer.
    verbose : bool
        Print a summary line to stdout.

    Returns
    -------
    (cleaned, n_fixed) : tuple[ndarray, int]
        *cleaned* has the same shape/dtype as *arr*.  *n_fixed* is the total
        number of spike frames replaced across all signals.
    """
    if arr.ndim == 0 or arr.shape[0] < 3:
        return arr.copy(), 0

    T = arr.shape[0]
    orig_shape = arr.shape
    # Flatten to (T, C) so we can iterate columns
    flat = arr.reshape(T, -1).astype(np.float64, copy=True)
    C = flat.shape[1]
    total_fixed = 0

    for c in range(C):
        x = flat[:, c]

        # Compute threshold once from the original signal's velocity scale
        v0 = np.diff(x)
        abs_v0 = np.abs(v0)
        finite = np.isfinite(abs_v0)
        if finite.sum() < 3:
            continue
        med_v = float(np.median(abs_v0[finite]))
        if med_v < 1e-12:
            continue
        thresh = threshold_factor * med_v

        for _iteration in range(max_iterations):
            v = np.diff(x)
            abs_v = np.abs(v)

            big = abs_v > thresh
            big_in  = big[:-1]
            big_out = big[1:]
            reversal = (v[:-1] * v[1:]) < 0
            spike = big_in & big_out & reversal

            idx = np.nonzero(spike)[0] + 1
            if idx.size == 0:
                break

            n_this = 0
            for t in idx:
                left, right = x[t - 1], x[t + 1]
                if np.isfinite(left) and np.isfinite(right):
                    x[t] = (left + right) / 2.0
                    n_this += 1
            total_fixed += n_this
            if n_this == 0:
                break

    out = flat.reshape(orig_shape)
    if verbose and total_fixed:
        tag = "single-frame" if max_iterations == 1 else f"up to {max_iterations} passes"
        print(f"  [isolated-spike] fixed {total_fixed} spike frames ({tag}, "
              f">{threshold_factor}\u00d7 median velocity, with reversal)")
    return out, total_fixed

def repair_wing_tip_identity_swaps(
    kp0: np.ndarray,
    kp1: np.ndarray,
    kp_names: Sequence[str],
    wing_kps: Tuple[str, str] = ('WingL_V13', 'WingR_V13'),
    threshold_mm: float = 0.10,
    max_flicker_frames: int = 30,
    max_iterations: int = 3,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Fix short-run wing-tip identity swaps between two tracked flies.

    During close-contact courtship the JARVIS multi-animal tracker occasionally
    assigns fly0's wing-tip to fly1 (or vice versa) for a short contiguous run
    of frames ("flicker").  Per-fly despike can't remove this because each
    fly's trajectory contains a real wing-tip position — just the wrong one.

    Each wing is handled independently.  For every transition ``i`` (between
    frames ``i`` and ``i+1``) we compare two hypotheses for this single wing:

    * ``keep``  — the two tracks continue as-is
    * ``swap``  — fly0's and fly1's wing identities switch at frame ``i+1``

    Transitions where ``cost(keep) − cost(swap) > threshold_mm`` are marked
    as flip events.  The per-transition cost is label-symmetric under a
    global swap, so absolute parity can't be recovered from cost alone.
    To avoid any global-parity ambiguity, we only act on **short** runs
    between consecutive flip events: when two flip events at ``t_a < t_b``
    are separated by ``t_b − t_a <= max_flicker_frames`` frames, the region
    ``[t_a+1, t_b]`` is treated as a flicker and its wing identities are
    exchanged.  Long regions are left untouched — if the tracker held a
    state for hundreds of frames, we trust that state.

    Only the two wing-tip keypoints are modified; all other keypoints pass
    through unchanged.  ``max_iterations`` is retained for symmetry with
    :func:`despike_isolated_spikes`; iterations exit early once no further
    flips qualify.

    Parameters
    ----------
    kp0, kp1 : ndarray, shape (T, N, 3)
        Per-fly keypoints, aligned in time.
    kp_names : sequence of str
        Keypoint names; must contain both entries of ``wing_kps``.
    wing_kps : (str, str)
        Names of the left and right wing-tip keypoints to repair.
    threshold_mm : float
        Minimum cost reduction (in position units, typically mm) required
        for a transition to count as a flip event.
    max_flicker_frames : int
        Maximum length (in frames) of a swapped run to repair.  Runs longer
        than this are assumed to be persistent states and left alone.
    max_iterations : int
        Maximum repair passes.  Exits early once a pass produces no flips.
    verbose : bool
        Print a summary line.

    Returns
    -------
    (kp0_out, kp1_out, n_frames_swapped) : tuple[ndarray, ndarray, int]
        Repaired keypoint arrays and total number of frame-level wing
        identity swaps applied across both wings and all iterations.
    """
    if kp0.shape != kp1.shape:
        raise ValueError(
            f"kp0/kp1 shape mismatch: {kp0.shape} vs {kp1.shape}")
    if kp0.ndim != 3 or kp0.shape[-1] != 3:
        raise ValueError(
            f"expected (T, N, 3) arrays, got kp0.shape={kp0.shape}")
    T = kp0.shape[0]
    if T < 3:
        return kp0.copy(), kp1.copy(), 0

    try:
        i_tips = [kp_names.index(n) for n in wing_kps]
    except ValueError as e:
        raise ValueError(
            f"wing keypoints {wing_kps} not in kp_names") from e

    out0 = kp0.astype(np.float64, copy=True)
    out1 = kp1.astype(np.float64, copy=True)
    total_swaps = 0

    def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(a - b, axis=-1)
        return np.where(np.isfinite(d), d, np.inf)

    def _short_run_assignment(parity: np.ndarray) -> np.ndarray:
        """True in every contiguous ``parity==True`` run of length
        ``<= max_flicker_frames``; False elsewhere."""
        assign = np.zeros(T, dtype=bool)
        in_run = False
        run_start = 0
        for t in range(T):
            if parity[t] and not in_run:
                in_run = True
                run_start = t
            elif not parity[t] and in_run:
                if t - run_start <= max_flicker_frames:
                    assign[run_start:t] = True
                in_run = False
        if in_run and (T - run_start) <= max_flicker_frames:
            assign[run_start:T] = True
        return assign

    def _assignment_for_wing(tip_idx: int) -> Tuple[np.ndarray, int]:
        """Return (per-frame assignment bool, n_frames_swapped) for one wing."""
        tip0 = out0[:, tip_idx]
        tip1 = out1[:, tip_idx]
        keep = _dist(tip0[1:], tip0[:-1]) + _dist(tip1[1:], tip1[:-1])
        swap = _dist(tip1[1:], tip0[:-1]) + _dist(tip0[1:], tip1[:-1])
        flip = (keep - swap) > threshold_mm
        if not flip.any():
            return np.zeros(T, dtype=bool), 0
        # Absolute parity is unrecoverable from cost alone, so we evaluate
        # both interpretations and take whichever places more short runs —
        # i.e., covers more plausible flicker frames.  The short-run length
        # cap ensures neither parity ever proposes flipping a long region.
        parity0 = np.concatenate(([False], np.cumsum(flip) % 2 == 1))
        assign0 = _short_run_assignment(parity0)
        assign1 = _short_run_assignment(~parity0)
        chosen = assign0 if assign0.sum() >= assign1.sum() else assign1
        return chosen, int(chosen.sum())

    for _it in range(max_iterations):
        n_this = 0
        for tip_idx in i_tips:
            assign, n = _assignment_for_wing(tip_idx)
            if n == 0:
                continue
            tmp = out0[assign, tip_idx].copy()
            out0[assign, tip_idx] = out1[assign, tip_idx]
            out1[assign, tip_idx] = tmp
            n_this += n
        if n_this == 0:
            break
        total_swaps += n_this

    if verbose and total_swaps:
        print(f"  [wing-tip swap repair] swapped {total_swaps} frame-wing "
              f"identities (>{threshold_mm:.3f} mm per-transition gain, "
              f"runs \u2264 {max_flicker_frames} frames, "
              f"{'/'.join(wing_kps)})")
    return out0.astype(kp0.dtype, copy=False), out1.astype(kp1.dtype, copy=False), total_swaps

def medfilt_despike(
    arr: np.ndarray,
    kernel: int = 7,
    threshold_factor: float = 10.0,
    max_replace_frac: float = 0.10,
    verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """Replace frames that deviate from a local median by more than a
    velocity-based threshold.

    Designed for non-singing flies where multi-frame tracking excursions
    are common (keypoint drifts to wrong feature for several frames).
    NOT safe for fast oscillatory signals like male wing song — the median
    filter will flatten real wing beats.

    Parameters
    ----------
    arr : ndarray
        Input array, shape ``(T,)``, ``(T, D)``, or ``(T, N, 3)``.
    kernel : int
        Median filter kernel size (must be odd).
    threshold_factor : float
        A frame is replaced when ``|signal - medfilt| > threshold_factor
        × median(|diff(signal)|)``.  Uses the same velocity scale as
        :func:`despike_isolated_spikes`.
    max_replace_frac : float
        Safety cap: skip a signal column if more than this fraction of
        frames would be replaced (avoids destroying good data).
    verbose : bool
        Print summary.

    Returns
    -------
    (cleaned, n_fixed) : tuple[ndarray, int]
    """
    from scipy.signal import medfilt as _medfilt

    if arr.ndim == 0 or arr.shape[0] < kernel:
        return arr.copy(), 0

    T = arr.shape[0]
    orig_shape = arr.shape
    flat = arr.reshape(T, -1).astype(np.float64, copy=True)
    C = flat.shape[1]
    total_fixed = 0

    for c in range(C):
        sig = flat[:, c]
        med = _medfilt(sig, kernel_size=kernel)
        dev = np.abs(sig - med)

        v = np.abs(np.diff(sig))
        vf = v[np.isfinite(v)]
        if len(vf) < 5:
            continue
        med_v = float(np.median(vf))
        if med_v < 1e-12:
            continue
        thresh = threshold_factor * med_v

        bad = dev > thresh
        n_bad = int(bad.sum())
        if n_bad > 0 and n_bad < T * max_replace_frac:
            sig[bad] = med[bad]
            total_fixed += n_bad

    out = flat.reshape(orig_shape)
    if verbose and total_fixed:
        print(f"  [medfilt-despike] replaced {total_fixed} frames "
              f"(kernel={kernel}, >{threshold_factor}\u00d7 median velocity)")
    return out, total_fixed

# ======================================================================
# from utils/pair_validity.py
# ======================================================================


@dataclass
class PairValidityConfig:
    enabled: bool = True
    critical_kp_patterns: Sequence[str] = field(
        default_factory=lambda: [
            "Scutellum",
            "Scutum",
            "*_ThxCx",
            "*_FeTi",
        ]
    )
    ground_kp_patterns: Sequence[str] = field(
        default_factory=lambda: ["*_TaTip", "*_TaT3"]
    )
    ground_epsilon_mm: float = 0.05
    floor_percentile: float = 5.0
    swap_guard_frames: int = 5
    min_paired_frames: int = 30
    min_solo_frames: int = 30
    # Identity-collapse detector. When > 0, frames where the two flies'
    # ``colocation_centroid_kp`` are closer than this threshold are marked
    # invalid for BOTH flies (see ``compute_colocation_mask``). 0 disables
    # the check. For courtship, ~1.0 mm (half a Drosophila body length) is
    # a reasonable floor.
    min_pair_separation_mm: float = 0.0
    colocation_centroid_kp: str = "Scutellum"

def _match_indices(kp_names: Sequence[str], patterns: Sequence[str]) -> List[int]:
    idx = []
    for i, name in enumerate(kp_names):
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            idx.append(i)
    return idx

def _filter_valid(kp: np.ndarray, critical_idx: Sequence[int]) -> np.ndarray:
    """A frame is filter-valid iff every critical keypoint is finite."""
    if len(critical_idx) == 0:
        return np.ones(kp.shape[0], dtype=bool)
    sub = kp[:, list(critical_idx), :]
    return np.all(np.isfinite(sub), axis=(1, 2))

def _ground_valid(
    kp: np.ndarray, ground_idx: Sequence[int], percentile: float, epsilon: float
) -> Tuple[np.ndarray, float]:
    """Per-frame: at least one ground keypoint is within epsilon of the floor.

    Floor z is estimated as the `percentile`-th percentile of the ground
    keypoints' z values across the bout (ignoring NaNs).
    """
    if len(ground_idx) == 0:
        return np.ones(kp.shape[0], dtype=bool), float("nan")
    zs = kp[:, list(ground_idx), 2]  # (T, K)
    finite_zs = zs[np.isfinite(zs)]
    if finite_zs.size == 0:
        return np.zeros(kp.shape[0], dtype=bool), float("nan")
    floor_z = float(np.percentile(finite_zs, percentile))
    # min over ground kps per frame; nan propagates → invalid
    with np.errstate(invalid="ignore"):
        min_z = np.nanmin(zs, axis=1)
    grounded = np.isfinite(min_z) & (min_z <= floor_z + epsilon)
    return grounded, floor_z

def _guard_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a True mask by `radius` frames on each side."""
    if radius <= 0 or not mask.any():
        return mask
    T = mask.shape[0]
    out = mask.copy()
    for shift in range(1, radius + 1):
        out[shift:] |= mask[:-shift]
        out[:-shift] |= mask[shift:]
    return out

def compute_colocation_mask(
    fly0_kp: np.ndarray,
    fly1_kp: np.ndarray,
    kp_names: Sequence[str],
    min_separation_mm: float,
    centroid_kp: str = "Scutellum",
) -> np.ndarray:
    """Per-frame mask flagging identity-collapse frames in a paired bout.

    A frame is True iff the two flies' ``centroid_kp`` positions are
    measurable and closer than ``min_separation_mm``. The JARVIS multi-peak
    tracker occasionally produces two output tracks that both lock onto the
    same physical animal (e.g. when one fly is occluded); that failure mode
    shows up as an inter-fly centroid distance that collapses to ~0.

    NaN handling: frames with any NaN in either centroid are returned as
    False — we only invalidate frames we can actually measure. Downstream
    validity logic will still drop NaN frames via ``_filter_valid``.

    Args:
        fly0_kp, fly1_kp: (T, N, 3) keypoints in a *shared* world frame.
            Passing egocentric arrays is useless because each fly is at the
            origin in its own frame.
        kp_names: ordered keypoint names (length N, shared by both flies).
        min_separation_mm: inter-centroid distance threshold. 0 disables
            the check and returns an all-False mask.
        centroid_kp: keypoint name to use as the body centroid. Must be in
            ``kp_names``. If missing, the function returns an all-False
            mask (fail-open).

    Returns:
        (T,) bool ndarray. True on collapse frames.
    """
    T = fly0_kp.shape[0]
    if min_separation_mm <= 0 or T == 0:
        return np.zeros(T, dtype=bool)
    if fly0_kp.shape != fly1_kp.shape:
        return np.zeros(T, dtype=bool)
    try:
        idx = list(kp_names).index(centroid_kp)
    except ValueError:
        return np.zeros(T, dtype=bool)
    c0 = fly0_kp[:, idx, :]
    c1 = fly1_kp[:, idx, :]
    diff = c0 - c1
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(diff, axis=1)
    mask = np.isfinite(dist) & (dist < float(min_separation_mm))
    return mask

def compute_pair_validity(
    fly0_kp: np.ndarray,
    fly1_kp: np.ndarray,
    kp_names: Sequence[str],
    cfg: Optional[PairValidityConfig] = None,
    swap_state: Optional[np.ndarray] = None,
    edge_nan_mask_fly0: Optional[np.ndarray] = None,
    edge_nan_mask_fly1: Optional[np.ndarray] = None,
) -> dict:
    """Compute per-frame validity masks for a paired bout.

    Parameters
    ----------
    fly0_kp, fly1_kp : ndarray (T, N, 3)
        Filtered keypoints for each fly over the same frame range.
    kp_names : list[str]
        Keypoint names (length N, shared by both flies).
    cfg : PairValidityConfig, optional
    swap_state : ndarray (T,) bool, optional
        Cumulative relink state from utils.identity_relink.relink_pair —
        True where the assignment was flipped relative to the input. The
        keypoints passed in are assumed to already be relink-corrected, so
        only *toggle events* in this state mark uncertain frames; toggles
        are dilated by `cfg.swap_guard_frames` on each side.
    edge_nan_mask_fly0, edge_nan_mask_fly1 : ndarray (T, N) bool, optional
        Per-fly edge-NaN masks carried over from preprocessing. Frames where
        ANY keypoint was in a leading/trailing NaN run (whether filled by
        bounded extrapolation or left NaN) are marked not-valid for that fly.
        Callers in the joint-relink pathway should pass these *after* applying
        any swap_state-driven row exchange so the masks remain aligned with
        the post-swap keypoints.

    Returns
    -------
    dict with keys:
        valid_fly0, valid_fly1, valid_both : (T,) bool
        pair_state : (T,) uint8    (0 none, 1 fly0, 2 fly1, 3 both)
        bout_pair_class : str      'paired'|'fly0_only'|'fly1_only'|'mixed'|'empty'
        floor_z_fly0, floor_z_fly1 : float
        identity_valid : (T,) bool
        pair_colocated : (T,) bool  — True on frames where the two flies'
            centroid keypoints are within ``cfg.min_pair_separation_mm``
            (identity-collapse detector; False everywhere when disabled).
        n_colocated : int
    """
    if cfg is None:
        cfg = PairValidityConfig()
    assert fly0_kp.shape == fly1_kp.shape, (
        f"shape mismatch {fly0_kp.shape} vs {fly1_kp.shape}"
    )
    T = fly0_kp.shape[0]

    critical_idx = _match_indices(kp_names, cfg.critical_kp_patterns)
    ground_idx = _match_indices(kp_names, cfg.ground_kp_patterns)

    filt0 = _filter_valid(fly0_kp, critical_idx)
    filt1 = _filter_valid(fly1_kp, critical_idx)

    def _edge_bad(mask, fly_kp):
        if mask is None:
            return None
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != (T, fly_kp.shape[1]):
            return None
        if critical_idx:
            return arr[:, list(critical_idx)].any(axis=1)
        return arr.any(axis=1)

    edge_bad0 = _edge_bad(edge_nan_mask_fly0, fly0_kp)
    edge_bad1 = _edge_bad(edge_nan_mask_fly1, fly1_kp)
    if edge_bad0 is not None:
        filt0 = filt0 & ~edge_bad0
    if edge_bad1 is not None:
        filt1 = filt1 & ~edge_bad1

    grd0, floor0 = _ground_valid(
        fly0_kp, ground_idx, cfg.floor_percentile, cfg.ground_epsilon_mm
    )
    grd1, floor1 = _ground_valid(
        fly1_kp, ground_idx, cfg.floor_percentile, cfg.ground_epsilon_mm
    )

    if swap_state is not None:
        swap = np.asarray(swap_state, dtype=bool)
        if swap.shape[0] != T:
            raise ValueError(
                f"swap_state length {swap.shape[0]} != bout length {T}"
            )
        # Only the toggle events are uncertain — frames deep inside a stable
        # (un)swapped segment are reliable because the relink correction has
        # already been applied to the keypoints.
        toggles = np.zeros(T, dtype=bool)
        if T >= 2:
            toggles[1:] = swap[1:] != swap[:-1]
        guarded = _guard_dilate(toggles, cfg.swap_guard_frames)
        identity_valid = ~guarded
    else:
        identity_valid = np.ones(T, dtype=bool)

    # Identity-collapse detector. When enabled (min_pair_separation_mm > 0),
    # frames where the two flies' centroid keypoints are closer than the
    # threshold are treated as invalid for *both* flies — we have no way to
    # tell which track was real, so the safe policy is to drop both.
    pair_colocated = compute_colocation_mask(
        fly0_kp,
        fly1_kp,
        kp_names,
        min_separation_mm=cfg.min_pair_separation_mm,
        centroid_kp=cfg.colocation_centroid_kp,
    )
    not_colocated = ~pair_colocated

    valid_fly0 = filt0 & grd0 & identity_valid & not_colocated
    valid_fly1 = filt1 & grd1 & identity_valid & not_colocated
    valid_both = valid_fly0 & valid_fly1

    pair_state = (
        valid_fly0.astype(np.uint8) | (valid_fly1.astype(np.uint8) << 1)
    )

    n_both = int(valid_both.sum())
    n_fly0 = int(valid_fly0.sum())
    n_fly1 = int(valid_fly1.sum())
    bout_pair_class = classify_bout(
        n_both, n_fly0, n_fly1, cfg.min_paired_frames, cfg.min_solo_frames
    )

    return dict(
        valid_fly0=valid_fly0,
        valid_fly1=valid_fly1,
        valid_both=valid_both,
        pair_state=pair_state,
        identity_valid=identity_valid,
        pair_colocated=pair_colocated,
        bout_pair_class=bout_pair_class,
        floor_z_fly0=floor0,
        floor_z_fly1=floor1,
        n_frames=T,
        n_valid_both=n_both,
        n_valid_fly0=n_fly0,
        n_valid_fly1=n_fly1,
        n_colocated=int(pair_colocated.sum()),
    )

def classify_bout(
    n_both: int,
    n_fly0: int,
    n_fly1: int,
    min_paired: int,
    min_solo: int,
) -> str:
    has_pair = n_both >= min_paired
    has_f0 = n_fly0 >= min_solo
    has_f1 = n_fly1 >= min_solo
    if has_pair and (n_fly0 - n_both >= min_solo or n_fly1 - n_both >= min_solo):
        return "mixed"
    if has_pair:
        return "paired"
    if has_f0 and not has_f1:
        return "fly0_only"
    if has_f1 and not has_f0:
        return "fly1_only"
    if has_f0 and has_f1:
        # both have solo frames but never paired enough → mixed solo
        return "mixed"
    return "empty"

# ======================================================================
# from utils/song_analysis.py
# ======================================================================


_REQUIRED_KP = (
    "Scutellum",
    "WingL_base",
    "WingR_base",
    "Antenna_Base",
    "WingL_V12",
    "WingL_V13",
    "WingR_V12",
    "WingR_V13",
)

@dataclass
class SongAnalysisConfig:
    """All tunable parameters for the song detector.

    Defaults reproduce
    ``notebooks/Courtship_Song_Analysis.ipynb`` cell 20 exactly so
    results from the paper notebook match the sandbox. The comments
    note which frame-rate / unit assumptions each value makes.
    """

    # Frame rate -------------------------------------------------------------
    fs: float = 800.0  # Hz — Johnson-lab courtship camera rate

    # Pipeline selection -----------------------------------------------------
    # Which detector(s) to run inside ``analyze_fly_song``:
    #   "legacy" — only the FFT-spectrogram + bilateral derivative-peak
    #              detector. ``frame_labels`` / ``summary`` come from it and
    #              are mirrored under ``*_legacy``; no ``*_new`` keys.
    #   "new"    — only the Butterworth+Hilbert pulse + multitaper F-test
    #              sine detector. ``frame_labels`` / ``summary`` come from
    #              it and are mirrored under ``*_new``; no ``*_legacy`` keys.
    #   "both"   — run both in parallel. Primary keys still come from the
    #              legacy detector (backward compatible); ``*_new`` is the
    #              experimental detector.
    pipeline: str = "legacy"

    # ------------------------------------------------------------------ #
    # New pulse detector (Butterworth bandpass + Hilbert envelope, per paper)
    # ------------------------------------------------------------------ #
    pulse_band_hz: Tuple[float, float] = (200.0, 380.0)
    pulse_filter_order: int = 4
    pulse_envelope_lp_hz: float = 25.0
    pulse_noise_floor_percentile: float = 50.0  # median of bandpassed |x|
    pulse_threshold_mult: float = 3.0           # T = mult * noise_floor.
                                                #  Lowered from 4.0 after
                                                #  hand-validation on bout 37:
                                                #  a 12-pulse train at 32 ms
                                                #  IPI had envelope peaks
                                                #  right at 4×median of the
                                                #  bandpassed signal, so
                                                #  only 4/12 survived. 3× is
                                                #  a stable noise floor the
                                                #  envelope rarely touches
                                                #  outside real pulses.
    pulse_min_dist_ms: float = 15.0
    pulse_dedupe_window_ms: float = 10.0        # keep-largest window
    pulse_isolation_window_ms: float = 120.0    # drop if no neighbor within
    pulse_train_max_median_ipi_ms: float = 80.0  # drop trains above this
    pulse_train_min_n: int = 2
    pulse_mask_half_width_ms: float = 12.0
    # Spectral-ratio gate: reject candidate peaks where the local
    # pulse-band (200-380 Hz) envelope is weaker than this fraction of the
    # local sine-band (80-200 Hz) envelope. Kills false pulses produced by
    # sine-harmonic leakage on all-sine bouts where the noise-floor
    # estimate collapses. 0.5 preserves real pulses (bout 37: 12/12) while
    # eliminating sine-leakage false pulses (bout 61: 7→0).
    # Set to 0 to disable.
    pulse_spectral_ratio_min: float = 0.5

    # ------------------------------------------------------------------ #
    # New sine detector (multitaper F-test, per paper)
    # ------------------------------------------------------------------ #
    sine_band_hz: Tuple[float, float] = (80.0, 200.0)
    sine_filter_order: int = 4
    sine_window_ms: float = 100.0               # → 80 samples @ 800 Hz
    sine_hop_ms: float = 10.0                   # → 8 samples
    sine_test_band_hz: Tuple[float, float] = (90.0, 175.0)
    sine_dpss_NW: float = 3.0
    sine_dpss_K: int = 5                        # ≈ 2*NW - 1
    sine_f_test_p: float = 0.05                 # Raw per-window F-test p
                                                #  threshold (no Bonferroni
                                                #  correction). Chaining + the
                                                #  80 ms min-segment gate
                                                #  already require 8+
                                                #  consecutive freq-consistent
                                                #  windows, which provides
                                                #  strong segment-level
                                                #  control. Bonferroni on top
                                                #  of that was over-conservative
                                                #  because courtship sine is
                                                #  modulated (F~6-10, not
                                                #  pure-line-huge) — applying
                                                #  the correction dropped ~2/3
                                                #  of real sine windows on
                                                #  bouts 23/27.
    sine_noise_floor_percentile: float = 25.0   # per-window power gate uses
                                                #  this percentile of window
                                                #  band-power as floor
    sine_noise_floor_mult: float = 0.5          # gate = mult * pct(power).
                                                #  0.5 * pct25 only drops the
                                                #  lowest-energy windows; 1.0
                                                #  * median drops half.
    sine_freq_tolerance: float = 0.20           # ±20% for chaining
    sine_min_segment_ms: float = 80.0
    sine_pulse_overlap_max: float = 0.5         # reject windows with more
                                                #  than this fraction inside
                                                #  the pulse mask

    # ------------------------------------------------------------------ #
    # Legacy FFT/peak detector (kept for parallel comparison output)
    # ------------------------------------------------------------------ #
    # Spectrogram windowing (legacy only) -----------------------------------
    fft_nperseg: int = 48  # ~60 ms, freq res ~16.7 Hz
    fft_hop: int = 8       # ~10 ms
    fft_song_band: Tuple[float, float] = (80.0, 400.0)
    fft_sine_band: Tuple[float, float] = (100.0, 180.0)
    fft_pulse_band: Tuple[float, float] = (140.0, 400.0)

    # FFT classifier thresholds ---------------------------------------------
    song_power_threshold: float = 2e-8
    pulse_peak_freq_min: float = 140.0
    pulse_freq_ratio_min: float = 0.35
    min_segment_frames: int = 16

    # Legacy peak-based pulse detection (bilateral max |dZ/dt|) -------------
    use_peak_pulse: bool = True
    pulse_detect_height: float = 25.0       # min |dZ/dt| (data-units / s)
    pulse_detect_prominence: float = 15.0
    pulse_detect_min_dist_ms: float = 15.0
    pulse_detect_half_width_ms: float = 12.0
    # Max gap between consecutive detected peaks that still counts as the
    # same pulse train. Normal D. melanogaster inter-pulse intervals are
    # ~35 ms; a single missed peak doubles that to ~70 ms, two missed peaks
    # gives ~105 ms, three gives ~140 ms. 135 ms tolerates up to 3 dropped
    # peaks while staying below the shortest observed sine-straddling gap
    # (148.8 ms in pair bout 23 — last pulse before and first pulse after a
    # sine passage). True inter-train silences are typically >250 ms, so
    # 135 ms stays well below that floor.
    pulse_train_max_gap_ms: float = 135.0
    # Minimum ratio of peak height to local baseline |dZ/dt|. True pulses
    # stand ≥3-4× above the inter-pulse baseline; individual sine-carrier
    # cycles only rise ~1.5-2.5× above the sustained carrier energy, so
    # this cleanly rejects false pulses inside strong sine song.
    # Set to 0 to disable the filter.
    pulse_baseline_ratio_min: float = 3.0
    # ± window (ms) used to compute the local |dZ/dt| baseline at each
    # candidate peak. Must be wider than one inter-pulse interval (~35 ms)
    # so a pulse train's baseline is dominated by inter-pulse silence
    # rather than the pulses themselves.
    pulse_baseline_window_ms: float = 40.0
    # ± window (ms) around the peak itself to EXCLUDE from the baseline
    # computation (removes the peak's own width from the median).
    pulse_baseline_exclude_ms: float = 5.0

    # Per-pulse feature extraction (Clemens 2018 adapted to 800 Hz) ---------
    # 25 ms total window → 21 samples at 800 Hz (±10 frames + center).
    pulse_window_ms: float = 25.0
    # Min pulses in a bout before per-bout Pslow/Pfast fractions are trusted
    # as summary statistics (per-pulse features are still extracted either
    # way; this just gates downstream aggregation).
    pulse_feature_min_n: int = 40
    # Spectral center-of-mass threshold from Clemens (e^-1 of peak
    # magnitude). Frequencies whose |X(f)| exceeds this fraction of the
    # spectrum's maximum contribute to the carrier-frequency estimate.
    pulse_spectral_thresh: float = 0.3679

    # Wing activity detection (gating + dominant-wing choice) ---------------
    wing_activity_window: int = 100     # frames for smoothed |dZ/dt|
    wing_activity_threshold: float = 1.0
    singing_min_bout_frames: int = 80
    singing_max_gap_frames: int = 200

    # Tips used for the spectrogram (paired: one left + one right) ----------
    left_tip: str = "WingL_V13"
    right_tip: str = "WingR_V13"

    # Default wing joint indices in qpos (may be overridden per-model) -----
    qpos_wing_yaw_L: int = 7
    qpos_wing_roll_L: int = 8
    qpos_wing_pitch_L: int = 9
    qpos_wing_yaw_R: int = 10
    qpos_wing_roll_R: int = 11
    qpos_wing_pitch_R: int = 12

    def ms_to_frames(self, ms: float) -> int:
        return max(1, int(round(ms * 1e-3 * self.fs)))

def resolve_kp_indices(kp_names: Sequence[str]) -> Dict[str, int]:
    """Return a ``{name: index}`` dict for the keypoints the detector needs.

    Raises ``KeyError`` if any required keypoint is missing, so callers fail
    loudly instead of silently computing garbage.
    """
    names = list(kp_names)
    out: Dict[str, int] = {}
    missing: List[str] = []
    for kp in _REQUIRED_KP:
        try:
            out[kp] = names.index(kp)
        except ValueError:
            missing.append(kp)
    if missing:
        raise KeyError(
            f"song_analysis: missing required keypoint(s) {missing}. "
            f"Available: {names}"
        )
    return out

def compute_wing_extension_angles(
    xpos_ego: np.ndarray,
    kp_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Wing extension angle (deg) from egocentric keypoints.

    For each wing, returns the angle between the body axis
    (Scutellum → Antenna_Base) and the wing vector
    (wing base → midpoint of V12, V13).
    """
    scut = kp_idx["Scutellum"]
    ant = kp_idx["Antenna_Base"]
    wl_b = kp_idx["WingL_base"]
    wr_b = kp_idx["WingR_base"]
    wl12 = kp_idx["WingL_V12"]
    wl13 = kp_idx["WingL_V13"]
    wr12 = kp_idx["WingR_V12"]
    wr13 = kp_idx["WingR_V13"]

    body_axis = xpos_ego[:, ant] - xpos_ego[:, scut]
    body_axis /= np.linalg.norm(body_axis, axis=1, keepdims=True) + 1e-12

    tip_L = 0.5 * (xpos_ego[:, wl12] + xpos_ego[:, wl13])
    vec_L = tip_L - xpos_ego[:, wl_b]
    vec_L /= np.linalg.norm(vec_L, axis=1, keepdims=True) + 1e-12

    tip_R = 0.5 * (xpos_ego[:, wr12] + xpos_ego[:, wr13])
    vec_R = tip_R - xpos_ego[:, wr_b]
    vec_R /= np.linalg.norm(vec_R, axis=1, keepdims=True) + 1e-12

    dot_L = np.clip(np.sum(body_axis * vec_L, axis=1), -1, 1)
    dot_R = np.clip(np.sum(body_axis * vec_R, axis=1), -1, 1)
    return np.degrees(np.arccos(dot_L)), np.degrees(np.arccos(dot_R))

def compute_wing_horizontal_angles(
    xpos_ego: np.ndarray,
    kp_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Wing horizontal-plane angle (deg) from egocentric keypoints.

    Builds a body frame from Scutellum→Antenna_Base (fore-aft) and the
    left/right wing bases (lateral), projects each wing vector into the
    horizontal plane (perpendicular to the derived dorsal axis), and
    returns the unsigned angle between the projection and the body
    fore-aft line. 0° ≈ wing parallel to body axis (rest); 90° ≈ wing
    perpendicular to the body axis (fully extended).
    """
    scut = kp_idx["Scutellum"]
    ant = kp_idx["Antenna_Base"]
    wl_b = kp_idx["WingL_base"]
    wr_b = kp_idx["WingR_base"]
    wl12 = kp_idx["WingL_V12"]
    wl13 = kp_idx["WingL_V13"]
    wr12 = kp_idx["WingR_V12"]
    wr13 = kp_idx["WingR_V13"]

    fwd = xpos_ego[:, ant] - xpos_ego[:, scut]
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-12

    lat_raw = xpos_ego[:, wr_b] - xpos_ego[:, wl_b]
    dorsal = np.cross(fwd, lat_raw)
    dorsal /= np.linalg.norm(dorsal, axis=1, keepdims=True) + 1e-12
    lat = np.cross(dorsal, fwd)
    lat /= np.linalg.norm(lat, axis=1, keepdims=True) + 1e-12

    def _angle(t12: int, t13: int, base: int) -> np.ndarray:
        tip = 0.5 * (xpos_ego[:, t12] + xpos_ego[:, t13])
        vec = tip - xpos_ego[:, base]
        vec_h = vec - np.sum(vec * dorsal, axis=1, keepdims=True) * dorsal
        x = np.sum(vec_h * fwd, axis=1)
        y = np.sum(vec_h * lat, axis=1)
        return np.degrees(np.arctan2(np.abs(y), np.abs(x)))

    return _angle(wl12, wl13, wl_b), _angle(wr12, wr13, wr_b)

def extract_wing_joint_angles(
    qpos: np.ndarray, cfg: SongAnalysisConfig
) -> Dict[str, np.ndarray]:
    """Return the six wing DOFs from qpos as a dict (radians)."""
    return {
        "yaw_L":   qpos[:, cfg.qpos_wing_yaw_L],
        "roll_L":  qpos[:, cfg.qpos_wing_roll_L],
        "pitch_L": qpos[:, cfg.qpos_wing_pitch_L],
        "yaw_R":   qpos[:, cfg.qpos_wing_yaw_R],
        "roll_R":  qpos[:, cfg.qpos_wing_roll_R],
        "pitch_R": qpos[:, cfg.qpos_wing_pitch_R],
    }

def compute_wing_tip_signal(
    kp_world: np.ndarray, kp_idx: Dict[str, int]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract wing tip (x, y, z) traces from a (T, N, 3) keypoint array.

    Uses RAW world-frame keypoints (``kp_data``) rather than the IK-
    reconstructed ``xpos_egocentric`` because the IK output misses real
    strokes in some bouts (noted in cell 21 of the sandbox). Absolute
    position doesn't matter for FFT power or peak detection — only the
    oscillation shape.
    """
    tip_map = {
        "WingL_V12": kp_idx["WingL_V12"],
        "WingL_V13": kp_idx["WingL_V13"],
        "WingR_V12": kp_idx["WingR_V12"],
        "WingR_V13": kp_idx["WingR_V13"],
    }
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, i in tip_map.items():
        out[name] = {
            "x": np.asarray(kp_world[:, i, 0]),
            "y": np.asarray(kp_world[:, i, 1]),
            "z": np.asarray(kp_world[:, i, 2]),
        }
    return out

def _wing_activity(
    wing_data: Dict[str, Dict[str, np.ndarray]],
    fs: float,
    window: int,
) -> Dict[str, np.ndarray]:
    """Windowed mean ``|dZ/dt|`` per wing tip."""
    out: Dict[str, np.ndarray] = {}
    for tip in ("WingL_V12", "WingL_V13", "WingR_V12", "WingR_V13"):
        z = wing_data[tip]["z"]
        dz = np.abs(np.diff(z, prepend=z[0]) * fs)
        out[tip] = uniform_filter1d(dz, size=window)
    return out

def detect_singing_frames(
    wing_data: Dict[str, Dict[str, np.ndarray]],
    cfg: SongAnalysisConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], str]:
    """Mark frames with active wing oscillation and pick the dominant wing.

    Returns ``(is_singing, activities, dominant_wing)`` where
    ``dominant_wing`` is ``'L'`` or ``'R'``.
    """
    activities = _wing_activity(
        wing_data, cfg.fs, cfg.wing_activity_window
    )
    thr = cfg.wing_activity_threshold

    left_active = (
        (activities["WingL_V12"] > thr) & (activities["WingL_V13"] > thr)
    )
    right_active = (
        (activities["WingR_V12"] > thr) & (activities["WingR_V13"] > thr)
    )
    is_singing = left_active | right_active
    T = len(is_singing)

    # Bridge short gaps.
    changes = np.diff(is_singing.astype(int))
    ons = np.where(changes == 1)[0] + 1
    offs = np.where(changes == -1)[0] + 1
    if is_singing[0]:
        ons = np.concatenate([[0], ons])
    if is_singing[-1]:
        offs = np.concatenate([offs, [T]])
    n = min(len(ons), len(offs))
    ons, offs = ons[:n], offs[:n]
    for i in range(len(ons) - 1):
        if ons[i + 1] - offs[i] <= cfg.singing_max_gap_frames:
            is_singing[offs[i]:ons[i + 1]] = True

    # Drop short segments.
    changes = np.diff(is_singing.astype(int))
    ons = np.where(changes == 1)[0] + 1
    offs = np.where(changes == -1)[0] + 1
    if is_singing[0]:
        ons = np.concatenate([[0], ons])
    if is_singing[-1]:
        offs = np.concatenate([offs, [T]])
    n = min(len(ons), len(offs))
    ons, offs = ons[:n], offs[:n]
    for on, off in zip(ons, offs):
        if off - on < cfg.singing_min_bout_frames:
            is_singing[on:off] = False

    left_power = (
        float(np.mean(activities["WingL_V12"][is_singing]))
        if is_singing.any() else 0.0
    )
    right_power = (
        float(np.mean(activities["WingR_V12"][is_singing]))
        if is_singing.any() else 0.0
    )
    dominant_wing = "L" if left_power > right_power else "R"
    return is_singing, activities, dominant_wing

def _interp_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if not np.isnan(x).any():
        return x
    idx = np.arange(len(x))
    good = ~np.isnan(x)
    if good.sum() == 0:
        return np.zeros_like(x)
    return np.interp(idx, idx[good], x[good])

def _band_power(Sxx: np.ndarray, f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (f >= lo) & (f <= hi)
    return Sxx[mask, :].sum(axis=0)

def _peak_freq(Sxx: np.ndarray, f: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (f >= lo) & (f <= hi)
    if mask.sum() == 0:
        return np.zeros(Sxx.shape[1])
    sub = Sxx[mask, :]
    f_sub = f[mask]
    return f_sub[np.argmax(sub, axis=0)]

def _merge_short_segments(labels: np.ndarray, min_len: int) -> np.ndarray:
    if len(labels) == 0:
        return labels
    out = labels.copy()
    changes = np.where(out[1:] != out[:-1])[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(out)]])
    for s_i, e_i in zip(starts, ends):
        if e_i - s_i >= min_len:
            continue
        left_label = out[s_i - 1] if s_i > 0 else None
        right_label = out[e_i] if e_i < len(out) else None
        if left_label is not None:
            out[s_i:e_i] = left_label
        elif right_label is not None:
            out[s_i:e_i] = right_label
    return out

def _butter_bandpass(
    low_hz: float, high_hz: float, fs: float, order: int
) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    return butter(order, [low, high], btype="bandpass")

def _butter_lowpass(
    cutoff_hz: float, fs: float, order: int
) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    wn = min(cutoff_hz / nyq, 0.999)
    return butter(order, wn, btype="lowpass")

def _bilateral_pulse_mask_butterworth(
    z_L: np.ndarray, z_R: np.ndarray, cfg: SongAnalysisConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Paper-accurate pulse detector.

    Pipeline (adapted from FlySongSegmenter):

    1. 4th-order zero-phase Butterworth bandpass at ``pulse_band_hz``
       applied to each tip Z trace.
    2. Hilbert amplitude envelope per side, then bilateral ``max``.
    3. Zero-phase 4th-order Butterworth low-pass at ``pulse_envelope_lp_hz``
       to smooth the envelope.
    4. Noise floor from the bandpassed bilateral signal's
       ``pulse_noise_floor_percentile`` value; threshold
       ``T = pulse_threshold_mult * noise_floor``.
    5. ``find_peaks`` on the smoothed envelope with ``height=T`` and
       minimum inter-peak distance ``pulse_min_dist_ms``.
    6. Dedupe: within ``pulse_dedupe_window_ms`` keep only the
       largest-amplitude peak.
    7. Isolation gate: drop peaks with no neighbor within
       ``pulse_isolation_window_ms``.
    8. Per-train IPI gate: group peaks into trains; drop trains whose
       median IPI exceeds ``pulse_train_max_median_ipi_ms`` or that
       contain fewer than ``pulse_train_min_n`` peaks.
    9. Paint a per-frame mask ``±pulse_mask_half_width_ms`` around each
       surviving peak and fill intra-train gaps.

    Returns ``(mask, kept_peak_frames)``.
    """
    z_L = _interp_nan(np.asarray(z_L, dtype=float))
    z_R = _interp_nan(np.asarray(z_R, dtype=float))
    n = len(z_L)
    if n < 16:
        return np.zeros(n, dtype=bool), np.array([], dtype=int)

    b_bp, a_bp = _butter_bandpass(
        cfg.pulse_band_hz[0],
        cfg.pulse_band_hz[1],
        cfg.fs,
        cfg.pulse_filter_order,
    )
    b_lp, a_lp = _butter_lowpass(
        cfg.pulse_envelope_lp_hz, cfg.fs, cfg.pulse_filter_order
    )

    bp_L = filtfilt(b_bp, a_bp, z_L)
    bp_R = filtfilt(b_bp, a_bp, z_R)
    env_L = np.abs(hilbert(bp_L))
    env_R = np.abs(hilbert(bp_R))
    env = np.maximum(env_L, env_R)
    env_lp = filtfilt(b_lp, a_lp, env)

    # Sine-band envelope (80-200 Hz) for the spectral-ratio gate below.
    # Computed alongside the pulse-band envelope so we can compare the two
    # at each candidate peak and reject sine-harmonic leakage.
    if cfg.pulse_spectral_ratio_min > 0:
        b_bp_s, a_bp_s = _butter_bandpass(
            cfg.sine_band_hz[0],
            cfg.sine_band_hz[1],
            cfg.fs,
            cfg.pulse_filter_order,
        )
        bp_L_s = filtfilt(b_bp_s, a_bp_s, z_L)
        bp_R_s = filtfilt(b_bp_s, a_bp_s, z_R)
        env_sine = np.maximum(np.abs(hilbert(bp_L_s)), np.abs(hilbert(bp_R_s)))
    else:
        env_sine = None

    bp_max = np.maximum(np.abs(bp_L), np.abs(bp_R))
    noise_floor = float(
        np.percentile(bp_max, cfg.pulse_noise_floor_percentile)
    )
    if not np.isfinite(noise_floor) or noise_floor <= 0:
        noise_floor = float(np.median(np.abs(bp_max))) + 1e-12
    threshold = cfg.pulse_threshold_mult * noise_floor

    min_dist = cfg.ms_to_frames(cfg.pulse_min_dist_ms)
    peaks, _ = find_peaks(env_lp, height=threshold, distance=min_dist)

    # Dedupe within a 10 ms window (keep the largest-amplitude peak).
    if len(peaks):
        dedupe_w = max(1, cfg.ms_to_frames(cfg.pulse_dedupe_window_ms))
        amps = env_lp[peaks]
        keep_mask = np.ones(len(peaks), dtype=bool)
        order = np.argsort(-amps)  # descending amplitude
        occupied = np.zeros(n, dtype=bool)
        for rank in order:
            p = peaks[rank]
            lo = max(0, p - dedupe_w // 2)
            hi = min(n, p + dedupe_w // 2 + 1)
            if occupied[lo:hi].any():
                keep_mask[rank] = False
            else:
                occupied[lo:hi] = True
        peaks = peaks[keep_mask]
        peaks.sort()

    # Spectral-ratio gate: reject peaks whose local pulse-band envelope is
    # weaker than ``pulse_spectral_ratio_min`` × local sine-band envelope.
    # Real pulses have a sharp broadband transient that dominates the
    # 200-380 Hz band; sine-harmonic leakage into the pulse band is always
    # accompanied by much larger 80-200 Hz energy, so the ratio < 1.
    if len(peaks) and env_sine is not None:
        half_w = max(1, cfg.ms_to_frames(cfg.pulse_mask_half_width_ms))
        kept = []
        for p in peaks:
            lo = max(0, p - half_w)
            hi = min(n, p + half_w + 1)
            env_p_local = float(env[lo:hi].mean())
            env_s_local = float(env_sine[lo:hi].mean())
            if env_s_local <= 0:
                kept.append(p)
                continue
            if env_p_local >= cfg.pulse_spectral_ratio_min * env_s_local:
                kept.append(p)
        peaks = np.asarray(kept, dtype=int)

    # Isolation gate: drop peaks with no neighbor within 120 ms.
    if len(peaks) > 1:
        iso_w = cfg.ms_to_frames(cfg.pulse_isolation_window_ms)
        diffs = np.diff(peaks)
        near_prev = np.concatenate([[np.inf], diffs]) <= iso_w
        near_next = np.concatenate([diffs, [np.inf]]) <= iso_w
        peaks = peaks[near_prev | near_next]
    elif len(peaks) == 1:
        peaks = np.array([], dtype=int)  # a lone peak is never a train

    # Group into trains (break on gap > isolation window) and apply the
    # median-IPI gate.
    kept: List[int] = []
    if len(peaks):
        iso_w = cfg.ms_to_frames(cfg.pulse_isolation_window_ms)
        max_median_ipi = cfg.ms_to_frames(cfg.pulse_train_max_median_ipi_ms)
        train_start = 0
        for i in range(1, len(peaks) + 1):
            is_last = i == len(peaks)
            if is_last or (peaks[i] - peaks[i - 1]) > iso_w:
                train = peaks[train_start:i]
                if len(train) >= cfg.pulse_train_min_n:
                    median_ipi = float(np.median(np.diff(train)))
                    if median_ipi <= max_median_ipi:
                        kept.extend(int(p) for p in train)
                train_start = i
    peaks = np.asarray(kept, dtype=int)

    mask = np.zeros(n, dtype=bool)
    if len(peaks):
        half_w = cfg.ms_to_frames(cfg.pulse_mask_half_width_ms)
        for p in peaks:
            mask[max(0, p - half_w):min(n, p + half_w + 1)] = True
        # Fill intra-train gaps so the mask is contiguous within a train.
        iso_w = cfg.ms_to_frames(cfg.pulse_isolation_window_ms)
        for a, b in zip(peaks[:-1], peaks[1:]):
            if (b - a) <= iso_w:
                mask[a:b + 1] = True

    return mask, peaks

def _thomson_f_test(
    x: np.ndarray, tapers: np.ndarray, K: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thomson (1982) line-component F-test.

    Parameters
    ----------
    x : (N,) signal.
    tapers : (K, N) DPSS tapers (unit-energy).
    K : number of tapers.

    Returns
    -------
    freqs     : (N//2 + 1,) rFFT frequency bins, unit fraction of ``fs``.
    F_stat    : (len(freqs),) F-statistic per frequency bin.
    amp       : (len(freqs),) complex line-component amplitude estimate.
    """
    N = len(x)
    # X_k(f): rFFT of tapered signal for each taper.
    tapered = tapers * x[None, :]                   # (K, N)
    Xk = np.fft.rfft(tapered, axis=1)               # (K, Nf)
    # DC sum of each taper: only even-symmetric tapers (k=0,2,4,...) have
    # non-zero DC, odd tapers cancel out — but we include all to stay
    # general and they contribute ~0.
    U = tapers.sum(axis=1)                          # (K,)
    U_sq_sum = float((U * U).sum())
    if U_sq_sum <= 0:
        nf = Xk.shape[1]
        return (
            np.fft.rfftfreq(N, d=1.0),
            np.zeros(nf),
            np.zeros(nf, dtype=complex),
        )
    # μ(f) = Σ_k X_k(f) · U_k / Σ_k U_k²
    mu = (Xk * U[:, None]).sum(axis=0) / U_sq_sum   # (Nf,)
    # Residuals: X_k(f) - μ(f) · U_k
    resid = Xk - mu[None, :] * U[:, None]           # (K, Nf)
    resid_sq = (np.abs(resid) ** 2).sum(axis=0)     # (Nf,)
    num = (K - 1) * (np.abs(mu) ** 2) * U_sq_sum
    denom = np.where(resid_sq > 0, resid_sq, np.inf)
    F_stat = num / denom
    freqs = np.fft.rfftfreq(N, d=1.0)
    return freqs, F_stat, mu

def _classify_sine_multitaper(
    z: np.ndarray,
    pulse_mask: np.ndarray,
    cfg: SongAnalysisConfig,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Paper-accurate sine detector (Thomson multitaper F-test).

    Returns ``(frame_labels, window_features)`` where frame_labels contain
    only ``{"pulse", "sine", "quiet"}`` (pulse frames come from the
    provided ``pulse_mask``).
    """
    z = _interp_nan(np.asarray(z, dtype=float))
    n = len(z)
    pulse_mask = np.asarray(pulse_mask, dtype=bool)
    if pulse_mask.shape[0] != n:
        pulse_mask = np.zeros(n, dtype=bool)

    empty_features = {
        "window_centers": np.array([], dtype=int),
        "window_freq": np.array([], dtype=float),
        "window_F": np.array([], dtype=float),
        "window_p": np.array([], dtype=float),
        "window_is_sine": np.array([], dtype=bool),
        "sine_segments": np.zeros((0, 2), dtype=int),
        "pulse_event_frames": np.array([], dtype=int),
    }

    if n < 32:
        frame_labels = np.full(n, "quiet", dtype=object)
        frame_labels[pulse_mask] = "pulse"
        return frame_labels, empty_features

    # Pulse pre-mask: replace pulse samples with linear interp so the
    # bandpass filter isn't driven by pulse energy.
    z_masked = z.copy()
    if pulse_mask.any() and not pulse_mask.all():
        idx_all = np.arange(n)
        good = ~pulse_mask
        z_masked[pulse_mask] = np.interp(
            idx_all[pulse_mask], idx_all[good], z[good]
        )

    b_bp, a_bp = _butter_bandpass(
        cfg.sine_band_hz[0],
        cfg.sine_band_hz[1],
        cfg.fs,
        cfg.sine_filter_order,
    )
    z_bp = filtfilt(b_bp, a_bp, z_masked)

    W = cfg.ms_to_frames(cfg.sine_window_ms)
    hop = max(1, cfg.ms_to_frames(cfg.sine_hop_ms))
    if W < 8 or W > n:
        frame_labels = np.full(n, "quiet", dtype=object)
        frame_labels[pulse_mask] = "pulse"
        return frame_labels, empty_features

    NW = cfg.sine_dpss_NW
    K = max(2, int(cfg.sine_dpss_K))
    tapers = dpss(W, NW, Kmax=K)  # (K, W), unit-energy
    p_threshold = cfg.sine_f_test_p
    dof2 = 2 * K - 2

    starts = np.arange(0, n - W + 1, hop, dtype=int)
    n_win = len(starts)
    if n_win == 0:
        frame_labels = np.full(n, "quiet", dtype=object)
        frame_labels[pulse_mask] = "pulse"
        return frame_labels, empty_features

    freq_fraction = np.fft.rfftfreq(W, d=1.0)
    freq_hz = freq_fraction * cfg.fs
    test_lo, test_hi = cfg.sine_test_band_hz
    test_band = (freq_hz >= test_lo) & (freq_hz <= test_hi)
    if not test_band.any():
        frame_labels = np.full(n, "quiet", dtype=object)
        frame_labels[pulse_mask] = "pulse"
        return frame_labels, empty_features

    window_centers = starts + W // 2
    window_freq = np.zeros(n_win, dtype=float)
    window_F = np.zeros(n_win, dtype=float)
    window_p = np.ones(n_win, dtype=float)
    window_pulse_frac = np.zeros(n_win, dtype=float)
    window_power = np.zeros(n_win, dtype=float)

    for i, s in enumerate(starts):
        w_mask = pulse_mask[s:s + W]
        window_pulse_frac[i] = float(w_mask.mean())
        seg = z_bp[s:s + W]
        # Simple band-power proxy (mean-squared bandpassed amplitude).
        window_power[i] = float(np.mean(seg * seg))
        _, F_stat, _ = _thomson_f_test(seg, tapers, K)
        if F_stat.shape[0] != freq_hz.shape[0]:
            continue
        F_band = F_stat[test_band]
        f_band = freq_hz[test_band]
        if F_band.size == 0:
            continue
        idx_max = int(np.argmax(F_band))
        window_F[i] = float(F_band[idx_max])
        window_freq[i] = float(f_band[idx_max])
        window_p[i] = float(_f_dist.sf(window_F[i], 2, dof2))

    # Raw per-window F-test p threshold. We deliberately skip the Bonferroni
    # correction across test-band bins because the downstream chaining gate
    # (need ≥ ``sine_min_segment_ms / hop`` freq-consistent windows in a row
    # within ±``sine_freq_tolerance``) already provides segment-level
    # multiple-comparison control. Courtship sine is modulated, producing
    # moderate F stats (~5-15) with raw p ~0.02-0.08 per window; Bonferroni
    # at n_bins ≈ 9 dropped the effective threshold to ~0.0056, which
    # rejected ~2/3 of real sine windows on hand-validated bouts 23/27.
    p_threshold_bonf = p_threshold

    # Per-window power gate: mult × percentile of band-power across windows.
    # Median (50th pct) is too aggressive when a bout has loud pulse regions
    # whose interp-filtered residue inflates the median — genuine sine can
    # end up below it. A lower percentile (e.g. 25th) tracks the quiet-window
    # floor more faithfully.
    nz_power = window_power[window_power > 0]
    floor_power = (
        float(np.percentile(nz_power, cfg.sine_noise_floor_percentile))
        if nz_power.size > 0
        else 0.0
    )
    power_gate = cfg.sine_noise_floor_mult * floor_power
    is_candidate = (
        (window_p < p_threshold_bonf)
        & (window_pulse_frac <= cfg.sine_pulse_overlap_max)
        & (window_power >= power_gate)
    )

    # Chain consecutive candidate windows with ±freq_tol frequency
    # consistency. When a candidate's frequency doesn't match the running
    # reference, we drop THIS window (don't close + reopen) so that noise
    # regions with scattered F-test hits at random frequencies don't
    # produce spurious short segments.
    segments: List[Tuple[int, int, float, int]] = []
    seg_open = False
    seg_start = 0
    seg_end = 0
    seg_freqs: List[float] = []
    freq_tol = cfg.sine_freq_tolerance
    max_gap_windows = 2

    def _close_segment() -> None:
        nonlocal seg_open, seg_freqs
        if seg_open and seg_freqs:
            segments.append(
                (seg_start, seg_end, float(np.median(seg_freqs)), len(seg_freqs))
            )
        seg_open = False
        seg_freqs = []

    prev_cand_idx = -10 ** 9
    for i in range(n_win):
        if not is_candidate[i]:
            if seg_open and (i - prev_cand_idx) > max_gap_windows:
                _close_segment()
            continue
        f_i = window_freq[i]
        s_frame = starts[i]
        e_frame = starts[i] + W
        if not seg_open:
            seg_open = True
            seg_start = s_frame
            seg_end = e_frame
            seg_freqs = [f_i]
            prev_cand_idx = i
            continue
        # Gap timeout: close out the old segment and start a new one.
        if (i - prev_cand_idx) > max_gap_windows:
            _close_segment()
            seg_open = True
            seg_start = s_frame
            seg_end = e_frame
            seg_freqs = [f_i]
            prev_cand_idx = i
            continue
        f_ref = float(np.median(seg_freqs))
        if abs(f_i - f_ref) <= freq_tol * f_ref:
            seg_end = e_frame
            seg_freqs.append(f_i)
            prev_cand_idx = i
        # else: frequency inconsistent; skip this window without closing.
    _close_segment()

    # Length gate + minimum-candidate gate: reject segments shorter than
    # ``sine_min_segment_ms`` in frames OR segments whose span is padded
    # out mostly by the window width itself (need enough matching windows
    # to demonstrate a sustained signal).
    min_len = cfg.ms_to_frames(cfg.sine_min_segment_ms)
    min_cand_windows = max(2, min_len // max(1, hop))
    segments = [
        (s, e, f)
        for (s, e, f, n_cand) in segments
        if (e - s) >= min_len and n_cand >= min_cand_windows
    ]

    frame_labels = np.full(n, "quiet", dtype=object)
    for s, e, _ in segments:
        frame_labels[s:e] = "sine"
    # Pulse takes precedence over any overlapping sine.
    frame_labels[pulse_mask] = "pulse"

    seg_array = (
        np.asarray([[s, e] for (s, e, _) in segments], dtype=int)
        if segments
        else np.zeros((0, 2), dtype=int)
    )

    features = {
        "window_centers": window_centers,
        "window_freq": window_freq,
        "window_F": window_F,
        "window_p": window_p,
        "window_is_sine": is_candidate,
        "sine_segments": seg_array,
        "pulse_event_frames": np.array([], dtype=int),
    }
    return frame_labels, features

def _classify_one_side_fft_legacy(
    z: np.ndarray, cfg: SongAnalysisConfig
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """FFT classifier (primary) for one wing's tip Z trace. Returns
    ``(frame_labels, window_features)``. Pulse vs sine is determined by
    the peak frequency inside the song band AND the pulse/song power
    ratio. Non-singing windows fall through to ``quiet``."""
    z = _interp_nan(np.asarray(z))
    n_frames = len(z)
    nperseg_eff = min(cfg.fft_nperseg, n_frames)
    noverlap = max(0, nperseg_eff - cfg.fft_hop)
    if n_frames < 2:
        return (
            np.full(n_frames, "quiet", dtype=object),
            {
                "window_centers": np.array([], dtype=int),
                "song_power": np.array([]),
                "pulse_power": np.array([]),
                "peak_freq": np.array([]),
                "pulse_freq_ratio": np.array([]),
                "pulse_event_frames": np.array([], dtype=int),
            },
        )

    f_axis, t_axis, Sxx = sp_spectrogram(
        z, fs=cfg.fs, nperseg=nperseg_eff, noverlap=noverlap, detrend="linear"
    )
    n_windows = len(t_axis)

    song_p = _band_power(Sxx, f_axis, *cfg.fft_song_band)
    pulse_p = _band_power(Sxx, f_axis, *cfg.fft_pulse_band)
    peak_f = _peak_freq(Sxx, f_axis, *cfg.fft_song_band)

    with np.errstate(divide="ignore", invalid="ignore"):
        pulse_freq_ratio = np.where(song_p > 0, pulse_p / song_p, 0.0)

    window_labels = np.full(n_windows, "quiet", dtype=object)
    for i in range(n_windows):
        is_singing = song_p[i] >= cfg.song_power_threshold
        if is_singing:
            if (
                peak_f[i] >= cfg.pulse_peak_freq_min
                and pulse_freq_ratio[i] >= cfg.pulse_freq_ratio_min
            ):
                window_labels[i] = "pulse"
            else:
                window_labels[i] = "sine"

    frame_labels = np.full(n_frames, "quiet", dtype=object)
    window_centers = (t_axis * cfg.fs).astype(int)
    for i in range(n_windows):
        w_start = max(0, window_centers[i] - cfg.fft_hop // 2)
        w_end = min(n_frames, window_centers[i] + cfg.fft_hop // 2)
        frame_labels[w_start:w_end] = window_labels[i]
    if n_windows > 0:
        frame_labels[: max(0, window_centers[0])] = window_labels[0]
        frame_labels[min(n_frames, window_centers[-1]):] = window_labels[-1]

    frame_labels = _merge_short_segments(frame_labels, cfg.min_segment_frames)

    return frame_labels, {
        "window_centers": window_centers,
        "song_power": song_p,
        "pulse_power": pulse_p,
        "peak_freq": peak_f,
        "pulse_freq_ratio": pulse_freq_ratio,
        "pulse_event_frames": np.array([], dtype=int),
    }

def _bilateral_pulse_mask_legacy(
    z_L: np.ndarray, z_R: np.ndarray, cfg: SongAnalysisConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Derivative-peak pulse detector on ``max(|dZ/dt|_L, |dZ/dt|_R)``.
    Returns a per-frame mask (with intra-train gaps filled) and the peak
    frame indices themselves. This is the primary pulse detector."""
    sig_L = np.abs(np.diff(z_L, prepend=z_L[0]) * cfg.fs)
    sig_R = np.abs(np.diff(z_R, prepend=z_R[0]) * cfg.fs)
    dz_max = np.maximum(sig_L, sig_R)

    min_dist = cfg.ms_to_frames(cfg.pulse_detect_min_dist_ms)
    half_w = cfg.ms_to_frames(cfg.pulse_detect_half_width_ms)
    peaks, _ = find_peaks(
        dz_max,
        height=cfg.pulse_detect_height,
        prominence=cfg.pulse_detect_prominence,
        distance=min_dist,
    )

    # Relative peak-to-baseline gate. For a true pulse the inter-pulse
    # |dZ/dt| sits near the noise floor, so peak/baseline is large (≥4).
    # For a strong sine carrier the baseline *is* the signal energy, so
    # peak/baseline collapses toward ~1.5-2.5 and these candidates get
    # dropped here before we paint the pulse mask.
    if len(peaks) and cfg.pulse_baseline_ratio_min > 0:
        w_base = cfg.ms_to_frames(cfg.pulse_baseline_window_ms)
        ex = cfg.ms_to_frames(cfg.pulse_baseline_exclude_ms)
        kept = []
        for p in peaks:
            lo = max(0, p - w_base)
            hi = min(len(dz_max), p + w_base + 1)
            local = dz_max[lo:hi]
            center_lo = max(0, (p - ex) - lo)
            center_hi = min(len(local), (p + ex + 1) - lo)
            local = np.concatenate([local[:center_lo], local[center_hi:]])
            if len(local) == 0:
                kept.append(p)
                continue
            baseline = float(np.median(local))
            ratio = dz_max[p] / max(baseline, 1e-9)
            if ratio >= cfg.pulse_baseline_ratio_min:
                kept.append(p)
        peaks = np.asarray(kept, dtype=int)

    mask = np.zeros(len(dz_max), dtype=bool)
    for pk in peaks:
        mask[max(0, pk - half_w):min(len(dz_max), pk + half_w + 1)] = True

    # Bridge gaps between consecutive pulses in a single train.
    max_gap = cfg.ms_to_frames(cfg.pulse_train_max_gap_ms)
    for a, b in zip(peaks[:-1], peaks[1:]):
        if (b - a) <= max_gap:
            mask[a:b + 1] = True

    return mask, peaks

def extract_pulse_waveforms(
    wing_z: np.ndarray,
    peak_frames: np.ndarray,
    cfg: SongAnalysisConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice z-scored, sign-aligned pulse windows around each peak.

    Parameters
    ----------
    wing_z : (T,) float
        Tip Z trace (the same one the detector ran on). NaNs are
        interpolated with :func:`_interp_nan` first so a single missing
        frame next to a pulse does not drop the window.
    peak_frames : (N,) int
        Peak indices from ``window_features['pulse_event_frames']``.
    cfg : SongAnalysisConfig
        Supplies ``fs`` and ``pulse_window_ms``.

    Returns
    -------
    waveforms : (M, W) float32
        One row per surviving peak. Each row is linearly detrended,
        z-scored to unit std, and sign-flipped so its center sample is
        non-negative (Clemens alignment convention). ``W =
        2 * ms_to_frames(pulse_window_ms / 2) + 1``.
    kept_frames : (M,) int
        Subset of ``peak_frames`` whose full ±half_w window fits inside
        the bout. Callers that need to re-index other per-peak arrays
        should use this.
    """
    peak_frames = np.asarray(peak_frames, dtype=int)
    half_w = cfg.ms_to_frames(cfg.pulse_window_ms / 2.0)
    W = 2 * half_w + 1
    if peak_frames.size == 0:
        return np.zeros((0, W), dtype=np.float32), peak_frames

    z = _interp_nan(np.asarray(wing_z, dtype=float))
    T = len(z)

    inside = (peak_frames - half_w >= 0) & (peak_frames + half_w + 1 <= T)
    kept_frames = peak_frames[inside]
    if kept_frames.size == 0:
        return np.zeros((0, W), dtype=np.float32), kept_frames

    # Gather windows in one shot.
    offsets = np.arange(-half_w, half_w + 1)
    idx = kept_frames[:, None] + offsets[None, :]
    waves = z[idx].astype(np.float32)

    # Per-window detrend (remove linear baseline over the window).
    x = np.arange(W, dtype=np.float32)
    x_mean = x.mean()
    x_centered = x - x_mean
    denom = float((x_centered * x_centered).sum()) + 1e-12
    slope = (waves * x_centered).sum(axis=1, keepdims=True) / denom
    intercept = waves.mean(axis=1, keepdims=True) - slope * x_mean
    waves = waves - (slope * x + intercept)

    # z-score (per window).
    std = waves.std(axis=1, keepdims=True)
    std = np.where(std < 1e-9, 1.0, std)
    waves = waves / std

    # Sign-align: make the center sample non-negative.
    center = waves[:, half_w]
    flip = np.where(center < 0, -1.0, 1.0).astype(np.float32)
    waves = waves * flip[:, None]

    return waves, kept_frames

def compute_pulse_symmetry(waveforms: np.ndarray) -> np.ndarray:
    """Per-pulse symmetry index s (Clemens 2018 methods).

        s = (a · flip(b)) / (||a|| ||b||)

    where ``a`` is the first half and ``b`` the second half of the
    aligned window. Pslow ≈ +0.3, Pfast ≈ −0.2. Returns (N,) float.
    """
    waves = np.asarray(waveforms, dtype=float)
    if waves.ndim != 2 or waves.shape[0] == 0:
        return np.zeros(0, dtype=float)
    W = waves.shape[1]
    half = W // 2
    a = waves[:, :half]
    # skip the center sample when W is odd so |a| == |b|
    b_start = W - half
    b = waves[:, b_start:]
    b_flip = b[:, ::-1]
    num = (a * b_flip).sum(axis=1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na * nb
    out = np.zeros_like(num)
    good = denom > 1e-12
    out[good] = num[good] / denom[good]
    return out

def compute_pulse_carrier_freq(
    waveforms: np.ndarray,
    fs: float,
    thresh: float = 0.3679,
) -> np.ndarray:
    """Per-pulse spectral center-of-mass frequency (Hz).

    For each aligned pulse window, compute the rFFT magnitude, threshold
    at ``thresh * max(|X(f)|)``, and return the energy-weighted mean
    frequency across the surviving bins. Falls back to argmax on any
    degenerate spectrum. Returns (N,) float.
    """
    waves = np.asarray(waveforms, dtype=float)
    if waves.ndim != 2 or waves.shape[0] == 0:
        return np.zeros(0, dtype=float)
    N, W = waves.shape
    freqs = np.fft.rfftfreq(W, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(waves, axis=1))
    peak_mag = mag.max(axis=1, keepdims=True)
    mask = mag >= (thresh * peak_mag)
    weighted = mag * mask
    wsum = weighted.sum(axis=1)
    out = np.zeros(N, dtype=float)
    good = wsum > 1e-12
    out[good] = (weighted[good] * freqs[None, :]).sum(axis=1) / wsum[good]
    # Fallback to argmax for any degenerate rows.
    bad = ~good
    if bad.any():
        out[bad] = freqs[mag[bad].argmax(axis=1)]
    return out

def compute_pulse_wing_angle(
    angle_dominant: Optional[np.ndarray],
    peak_frames: np.ndarray,
    cfg: SongAnalysisConfig,
) -> Optional[np.ndarray]:
    """Mean extended-wing angle (deg) over ± ``pulse_window_ms``/2 around
    each peak. Returns (N,) float or ``None`` if ``angle_dominant`` is
    ``None`` (e.g. bout without egocentric positions)."""
    if angle_dominant is None:
        return None
    peak_frames = np.asarray(peak_frames, dtype=int)
    if peak_frames.size == 0:
        return np.zeros(0, dtype=float)
    ang = np.asarray(angle_dominant, dtype=float)
    T = len(ang)
    half_w = cfg.ms_to_frames(cfg.pulse_window_ms / 2.0)
    out = np.full(peak_frames.shape, np.nan, dtype=float)
    for i, p in enumerate(peak_frames):
        lo = max(0, p - half_w)
        hi = min(T, p + half_w + 1)
        seg = ang[lo:hi]
        if seg.size:
            out[i] = float(np.nanmean(np.abs(seg)))
    return out

def classify_song_both_sides(
    wing_data: Dict[str, Dict[str, np.ndarray]],
    cfg: SongAnalysisConfig,
) -> Dict[str, Dict[str, Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]]]:
    """Run the requested detector pipeline(s) on L/R wings.

    Honours ``cfg.pipeline``:
        "legacy" — only the FFT + derivative-peak detector
        "new"    — only the Butterworth + multitaper detector
        "both"   — both pipelines

    Skipped pipelines return ``None`` in their slot.

    Returned shape:
        {"L": {"legacy": (labels, features) or None,
               "new":    (labels, features) or None},
         "R": {...}}
    """
    if cfg.pipeline not in ("legacy", "new", "both"):
        raise ValueError(
            f"cfg.pipeline must be 'legacy', 'new', or 'both'; "
            f"got {cfg.pipeline!r}"
        )

    z_L = _interp_nan(np.asarray(wing_data[cfg.left_tip]["z"]))
    z_R = _interp_nan(np.asarray(wing_data[cfg.right_tip]["z"]))

    out: Dict[
        str, Dict[str, Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]]
    ] = {
        "L": {"legacy": None, "new": None},
        "R": {"legacy": None, "new": None},
    }

    if cfg.pipeline in ("new", "both"):
        # NEW pipeline: Butterworth + Hilbert pulse, multitaper sine.
        mask_new, peaks_new = _bilateral_pulse_mask_butterworth(z_L, z_R, cfg)
        labels_L_new, feats_L_new = _classify_sine_multitaper(z_L, mask_new, cfg)
        labels_R_new, feats_R_new = _classify_sine_multitaper(z_R, mask_new, cfg)
        feats_L_new["pulse_event_frames"] = peaks_new
        feats_R_new["pulse_event_frames"] = peaks_new
        out["L"]["new"] = (labels_L_new, feats_L_new)
        out["R"]["new"] = (labels_R_new, feats_R_new)

    if cfg.pipeline in ("legacy", "both"):
        # LEGACY pipeline: FFT classifier + bilateral peak pulse mask.
        labels_L_leg, feats_L_leg = _classify_one_side_fft_legacy(z_L, cfg)
        labels_R_leg, feats_R_leg = _classify_one_side_fft_legacy(z_R, cfg)
        if cfg.use_peak_pulse and len(z_L) > 1:
            mask_leg, peaks_leg = _bilateral_pulse_mask_legacy(z_L, z_R, cfg)
            labels_L_leg = labels_L_leg.copy()
            labels_R_leg = labels_R_leg.copy()
            labels_L_leg[labels_L_leg == "pulse"] = "sine"
            labels_R_leg[labels_R_leg == "pulse"] = "sine"
            labels_L_leg[mask_leg] = "pulse"
            labels_R_leg[mask_leg] = "pulse"
            feats_L_leg["pulse_event_frames"] = peaks_leg
            feats_R_leg["pulse_event_frames"] = peaks_leg
        out["L"]["legacy"] = (labels_L_leg, feats_L_leg)
        out["R"]["legacy"] = (labels_R_leg, feats_R_leg)

    return out

def segments_from_labels(
    frame_labels: np.ndarray, fs: float
) -> List[Dict]:
    """Run-length encode per-frame labels into a list of segment dicts."""
    T = len(frame_labels)
    out: List[Dict] = []
    if T == 0:
        return out
    current = frame_labels[0]
    seg_start = 0
    for i in range(1, T):
        if frame_labels[i] != current:
            out.append({
                "start": seg_start,
                "end": i - 1,
                "type": current,
                "n_frames": i - seg_start,
                "duration_s": (i - seg_start) / fs,
            })
            current = frame_labels[i]
            seg_start = i
    out.append({
        "start": seg_start,
        "end": T - 1,
        "type": current,
        "n_frames": T - seg_start,
        "duration_s": (T - seg_start) / fs,
    })
    return out

def _side_summary(
    frame_labels: np.ndarray, fs: float, valid_mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    T = len(frame_labels)
    if valid_mask is None:
        valid_mask = np.ones(T, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
    eff = int(valid_mask.sum())
    if eff == 0:
        return {
            "n_frames": 0, "duration_s": 0.0,
            "frac_pulse": 0.0, "frac_sine": 0.0,
            "frac_quiet": 0.0, "song_fraction": 0.0,
        }
    lbl = frame_labels[valid_mask]
    n_p = int((lbl == "pulse").sum())
    n_s = int((lbl == "sine").sum())
    n_q = int((lbl == "quiet").sum())
    return {
        "n_frames": eff,
        "duration_s": eff / fs,
        "frac_pulse": n_p / eff,
        "frac_sine": n_s / eff,
        "frac_quiet": n_q / eff,
        "song_fraction": (n_p + n_s) / eff,
    }

def analyze_fly_song(
    kp_world: np.ndarray,
    xpos_ego: Optional[np.ndarray],
    qpos: Optional[np.ndarray],
    kp_names: Sequence[str],
    cfg: Optional[SongAnalysisConfig] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict:
    """Full per-fly per-bout song analysis.

    Parameters
    ----------
    kp_world : (T, N, 3) or (T, N*3) float array
        RAW world-frame keypoints — used for the oscillation signal.
    xpos_ego : (T, N, 3) or None
        Egocentric positions — used only for wing extension angles
        (kinematic feature, not for classification). Pass ``None`` to
        skip.
    qpos : (T, D) or None
        Joint angles — used only for extracting wing DOFs for plotting.
        Pass ``None`` to skip.
    kp_names : sequence of keypoint name strings matching the N axis.
    cfg : SongAnalysisConfig or None
        Detector configuration. Defaults reproduce the sandbox
        behaviour.
    valid_mask : (T,) bool or None
        If provided, per-frame validity mask (e.g. ``valid_fly0 &
        ~pair_colocated``). Summaries are computed over valid frames
        only, but frame_labels cover all T frames so the caller can
        still plot them.

    Returns a dict with:
        ``wing_data``          — per-tip xyz traces
        ``sides``              — {'L': {frame_labels, segments, summary,
                                          window_features, pulse_features,
                                          [*_legacy aliases], [*_new aliases]},
                                  'R': {...}}.
                                  ``cfg.pipeline`` controls which detector
                                  fills the primary (unsuffixed) keys:
                                  "legacy" → FFT + derivative-peak, no
                                  ``*_new`` keys; "new" → Butterworth +
                                  multitaper, no ``*_legacy`` keys; "both"
                                  → primary stays legacy and ``*_new`` is
                                  populated alongside.
        ``dominant_wing``      — 'L' or 'R'
        ``is_singing``         — (T,) bool gate
        ``activities``         — per-tip windowed |dZ/dt|
        ``angle_L, angle_R``   — (T,) wing extension angles (deg) if
                                  xpos_ego was provided
        ``joints``             — dict of wing DOFs if qpos was provided
        ``summary``            — dominant-wing song_fraction + bout
                                  metadata, from whichever pipeline
                                  ``cfg.pipeline`` selects as primary
        ``summary_legacy``     — present only if the legacy detector ran
        ``summary_new``        — present only if the new detector ran
    """
    if cfg is None:
        cfg = SongAnalysisConfig()
    kp_idx = resolve_kp_indices(kp_names)

    kp_world = np.asarray(kp_world)
    if kp_world.ndim == 2:
        kp_world = kp_world.reshape(kp_world.shape[0], -1, 3)
    T = kp_world.shape[0]

    wing_data = compute_wing_tip_signal(kp_world, kp_idx)
    is_singing, activities, dominant_wing = detect_singing_frames(wing_data, cfg)

    # Run the detector pipeline(s) selected by ``cfg.pipeline`` and route
    # the chosen one to the unsuffixed primary keys. ``*_legacy`` /
    # ``*_new`` aliases are populated only for pipelines that actually ran.
    # When ``cfg.pipeline == "both"`` the legacy detector keeps the primary
    # slot (backward compatible).
    side_results = classify_song_both_sides(wing_data, cfg)
    primary_key = "new" if cfg.pipeline == "new" else "legacy"
    sides: Dict[str, Dict] = {}
    for sname, streams in side_results.items():
        labels_p, feats_p = streams[primary_key]
        segs_p = segments_from_labels(labels_p, cfg.fs)
        summ_p = _side_summary(labels_p, cfg.fs, valid_mask)
        side_d: Dict = {
            "frame_labels": labels_p,
            "window_features": feats_p,
            "segments": segs_p,
            "summary": summ_p,
        }
        if streams["legacy"] is not None:
            if primary_key == "legacy":
                side_d["frame_labels_legacy"] = labels_p
                side_d["window_features_legacy"] = feats_p
                side_d["segments_legacy"] = segs_p
                side_d["summary_legacy"] = summ_p
            else:
                l_labels, l_feats = streams["legacy"]
                side_d["frame_labels_legacy"] = l_labels
                side_d["window_features_legacy"] = l_feats
                side_d["segments_legacy"] = segments_from_labels(l_labels, cfg.fs)
                side_d["summary_legacy"] = _side_summary(l_labels, cfg.fs, valid_mask)
        if streams["new"] is not None:
            if primary_key == "new":
                side_d["frame_labels_new"] = labels_p
                side_d["window_features_new"] = feats_p
                side_d["segments_new"] = segs_p
                side_d["summary_new"] = summ_p
            else:
                n_labels, n_feats = streams["new"]
                side_d["frame_labels_new"] = n_labels
                side_d["window_features_new"] = n_feats
                side_d["segments_new"] = segments_from_labels(n_labels, cfg.fs)
                side_d["summary_new"] = _side_summary(n_labels, cfg.fs, valid_mask)
        sides[sname] = side_d

    # Wing extension angles (geometric feature, for plotting) -------------
    angle_L = angle_R = None
    horiz_angle_L = horiz_angle_R = None
    if xpos_ego is not None:
        xpe = np.asarray(xpos_ego)
        if xpe.ndim == 2:
            xpe = xpe.reshape(xpe.shape[0], -1, 3)
        angle_L, angle_R = compute_wing_extension_angles(xpe, kp_idx)
        horiz_angle_L, horiz_angle_R = compute_wing_horizontal_angles(xpe, kp_idx)

    # Per-pulse features (Clemens 2018, adapted) --------------------------
    # The bilateral peak detector stores the same peak frames under both
    # 'L' and 'R' sides; we still compute waveform-based features per side
    # because the tip-Z oscillation shape differs between wings.
    side_tip = {"L": cfg.left_tip, "R": cfg.right_tip}
    side_angle = {"L": angle_L, "R": angle_R}
    for sname in ("L", "R"):
        feats = sides[sname]["window_features"]
        peak_frames = np.asarray(
            feats.get("pulse_event_frames", np.array([], dtype=int)),
            dtype=int,
        )
        wing_z = wing_data[side_tip[sname]]["z"]
        waveforms, kept = extract_pulse_waveforms(wing_z, peak_frames, cfg)
        symmetry = compute_pulse_symmetry(waveforms)
        carrier_hz = compute_pulse_carrier_freq(
            waveforms, cfg.fs, thresh=cfg.pulse_spectral_thresh
        )
        wing_angle = compute_pulse_wing_angle(side_angle[sname], kept, cfg)
        ipi_ms = (
            np.diff(kept).astype(float) * 1000.0 / cfg.fs
            if kept.size > 1 else np.array([], dtype=float)
        )
        sides[sname]["pulse_features"] = {
            "peak_frames": kept,
            "waveforms": waveforms,
            "symmetry": symmetry,
            "carrier_hz": carrier_hz,
            "wing_angle": wing_angle,
            "ipi_ms": ipi_ms,
        }

    joints = None
    if qpos is not None:
        qp = np.asarray(qpos)
        if qp.shape[1] > max(cfg.qpos_wing_pitch_R, 0):
            joints = extract_wing_joint_angles(qp, cfg)

    dom_summary = sides[dominant_wing]["summary"]
    primary_summary = {
        "n_frames": T,
        "duration_s": T / cfg.fs,
        "dominant_wing": dominant_wing,
        "song_fraction": dom_summary["song_fraction"],
        "frac_pulse": dom_summary["frac_pulse"],
        "frac_sine": dom_summary["frac_sine"],
        "frac_quiet": dom_summary["frac_quiet"],
        "valid_n_frames": dom_summary["n_frames"],
    }

    def _bout_summary(side_summary: Dict[str, float]) -> Dict[str, float]:
        return {
            "n_frames": T,
            "duration_s": T / cfg.fs,
            "dominant_wing": dominant_wing,
            "song_fraction": side_summary["song_fraction"],
            "frac_pulse": side_summary["frac_pulse"],
            "frac_sine": side_summary["frac_sine"],
            "frac_quiet": side_summary["frac_quiet"],
            "valid_n_frames": side_summary["n_frames"],
        }

    out: Dict = {
        "wing_data": wing_data,
        "sides": sides,
        "dominant_wing": dominant_wing,
        "is_singing": is_singing,
        "activities": activities,
        "angle_L": angle_L,
        "angle_R": angle_R,
        "horiz_angle_L": horiz_angle_L,
        "horiz_angle_R": horiz_angle_R,
        "joints": joints,
        "summary": primary_summary,
    }
    if "summary_legacy" in sides[dominant_wing]:
        out["summary_legacy"] = (
            primary_summary if primary_key == "legacy"
            else _bout_summary(sides[dominant_wing]["summary_legacy"])
        )
    if "summary_new" in sides[dominant_wing]:
        out["summary_new"] = (
            primary_summary if primary_key == "new"
            else _bout_summary(sides[dominant_wing]["summary_new"])
        )
    return out

# ======================================================================
# from utils/sex_id.py
# ======================================================================


@dataclass
class SexIdConfig:
    #: Minimum song_fraction gap required for the song-based rule to be
    #: considered confident. If both flies sing with similar fractions
    #: (e.g. 0.10 vs 0.11) we fall back to body length.
    min_song_gap: float = 0.02
    #: Below this absolute song fraction on BOTH flies, the bout is
    #: effectively silent — fall back to body length.
    silence_threshold: float = 0.01
    #: Keypoints whose Euclidean distance we use as body length.
    body_length_kp_from: str = "Scutellum"
    body_length_kp_to: str = "Abd_tip"

def _median_body_length(
    kp: np.ndarray, kp_names: Sequence[str], cfg: SexIdConfig
) -> float:
    """Median |Scutellum − Abd_tip| over all frames with finite data."""
    kp = np.asarray(kp)
    if kp.ndim == 2:
        kp = kp.reshape(kp.shape[0], -1, 3)
    names = list(kp_names)
    i_from = names.index(cfg.body_length_kp_from)
    i_to = names.index(cfg.body_length_kp_to)
    d = np.linalg.norm(kp[:, i_from, :] - kp[:, i_to, :], axis=1)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if d.size else float("nan")

def identify_male_female(
    fly0_result: Dict,
    fly1_result: Dict,
    fly0_kp: np.ndarray,
    fly1_kp: np.ndarray,
    kp_names: Sequence[str],
    cfg: Optional[SexIdConfig] = None,
) -> Dict:
    """Return the male/female assignment for one bout.

    Parameters
    ----------
    fly0_result, fly1_result : dict
        Outputs of ``song_analysis.analyze_fly_song`` (must contain a
        ``summary.song_fraction`` entry).
    fly0_kp, fly1_kp : array
        ``kp_data`` (or any (T, N, 3) keypoint array) used for the
        body-length tiebreaker. Should be in the SAME units and frame of
        reference for both flies so body lengths are comparable.
    kp_names : Sequence[str]
        Names along the N axis.
    cfg : SexIdConfig, optional

    Returns
    -------
    dict with keys:
        ``male_id``          — 'fly0' or 'fly1'
        ``female_id``        — the other one
        ``criterion``        — 'song_fraction' or 'body_length'
        ``confidence``       — [0, 1], larger is more confident
        ``song_fraction_male`` / ``song_fraction_female``
        ``body_length_male`` / ``body_length_female`` — data-unit lengths
        ``disagree``         — True if the body-length criterion would
                               have flipped the song-based assignment
                               (sanity-check field for plots / tables).
    """
    if cfg is None:
        cfg = SexIdConfig()

    sf0 = float(fly0_result["summary"]["song_fraction"])
    sf1 = float(fly1_result["summary"]["song_fraction"])

    bl0 = _median_body_length(fly0_kp, kp_names, cfg)
    bl1 = _median_body_length(fly1_kp, kp_names, cfg)

    gap = sf0 - sf1
    both_silent = (sf0 < cfg.silence_threshold) and (sf1 < cfg.silence_threshold)
    ambiguous = abs(gap) < cfg.min_song_gap

    if both_silent or ambiguous:
        # Fall back to body-length: male is the smaller fly in Drosophila.
        if np.isfinite(bl0) and np.isfinite(bl1) and bl0 != bl1:
            male = "fly0" if bl0 < bl1 else "fly1"
            # Confidence = normalized size gap, capped at 1.
            size_gap = abs(bl0 - bl1) / max(bl0, bl1)
            confidence = float(min(1.0, size_gap * 5.0))
            criterion = "body_length"
        else:
            # Degenerate: no useful signal. Default to fly0 = male with
            # zero confidence so the caller can filter these out.
            male = "fly0"
            confidence = 0.0
            criterion = "body_length"
    else:
        male = "fly0" if gap > 0 else "fly1"
        # Confidence = normalized song-fraction gap, squashed.
        confidence = float(min(1.0, abs(gap) / max(1e-6, max(sf0, sf1)) ))
        criterion = "song_fraction"

    female = "fly1" if male == "fly0" else "fly0"
    sf_male = sf0 if male == "fly0" else sf1
    sf_female = sf1 if male == "fly0" else sf0
    bl_male = bl0 if male == "fly0" else bl1
    bl_female = bl1 if male == "fly0" else bl0

    # Sanity check: would the body-length rule have flipped this?
    disagree = False
    if criterion == "song_fraction" and np.isfinite(bl0) and np.isfinite(bl1):
        body_male = "fly0" if bl0 < bl1 else "fly1"
        disagree = (body_male != male)

    return {
        "male_id": male,
        "female_id": female,
        "criterion": criterion,
        "confidence": confidence,
        "song_fraction_male": sf_male,
        "song_fraction_female": sf_female,
        "body_length_male": bl_male,
        "body_length_female": bl_female,
        "disagree": bool(disagree),
    }

# ======================================================================
# from utils/locomotion.py
# ======================================================================


@dataclass
class LocomotionConfig:
    fs: float = 800.0
    centroid_kp: str = "Scutellum"
    forward_from_kp: str = "Scutellum"
    forward_to_kp: str = "Antenna_Base"

    # Savitzky-Golay smoothing of the centroid trajectory before
    # differentiating (frames). 1 disables smoothing.
    smooth_window: int = 7
    smooth_polyorder: int = 2

    # Floor / ground detection (reused from PairValidityConfig defaults).
    ground_kp_patterns: Sequence[str] = field(
        default_factory=lambda: ("*claw*", "*Claw*", "Tarsus*", "*_Ti*")
    )
    ground_epsilon: float = 0.05
    floor_percentile: float = 5.0

    # Stopped / walking classifier thresholds, in body-length / s.
    walking_speed_bl: float = 0.5
    stopped_speed_bl: float = 0.3  # hysteresis floor
    min_run_ms: float = 40.0        # minimum state duration before flipping

def _smooth_xy(xy: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Row-wise Savitzky-Golay smoothing of a (T, 2 or 3) trajectory.

    Falls back to the raw signal if the bout is shorter than ``window``.
    """
    if window <= 1 or xy.shape[0] < window:
        return xy
    out = np.empty_like(xy, dtype=float)
    for c in range(xy.shape[1]):
        out[:, c] = savgol_filter(xy[:, c], window, polyorder, mode="interp")
    return out

def compute_centroid_velocity(
    kp_world: np.ndarray,
    kp_names: Sequence[str],
    cfg: Optional[LocomotionConfig] = None,
    body_length: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Forward / lateral speed + turn rate derived from the body axis.

    Forward axis is a per-frame unit vector from ``forward_from_kp``
    (Scutellum) to ``forward_to_kp`` (Antenna_Base). Speed is projected
    onto this axis and its perpendicular in the horizontal plane.

    Returns a dict with (T,) arrays:
        ``speed``         — |centroid velocity| in data-units / s
        ``forward_speed`` — signed projection onto the body heading
        ``lateral_speed`` — signed perpendicular component (left +)
        ``turn_rate``     — d(heading)/dt in deg/s (horizontal plane)
        ``heading``       — (T,) rad, horizontal-plane heading angle
        ``speed_bl``      — ``speed / body_length`` if body_length given
        ``forward_speed_bl``, ``lateral_speed_bl`` — normalized variants
    """
    if cfg is None:
        cfg = LocomotionConfig()
    kp_world = np.asarray(kp_world)
    if kp_world.ndim == 2:
        kp_world = kp_world.reshape(kp_world.shape[0], -1, 3)
    T = kp_world.shape[0]

    names = list(kp_names)
    i_cent = names.index(cfg.centroid_kp)
    i_from = names.index(cfg.forward_from_kp)
    i_to = names.index(cfg.forward_to_kp)

    centroid = kp_world[:, i_cent, :].astype(float)
    centroid_s = _smooth_xy(centroid, cfg.smooth_window, cfg.smooth_polyorder)

    # Horizontal-plane velocity (use full 3D for magnitude, xy for heading).
    vel = np.gradient(centroid_s, 1.0 / cfg.fs, axis=0)
    speed = np.linalg.norm(vel, axis=1)

    # Body heading vector (horizontal projection).
    body_vec = kp_world[:, i_to, :] - kp_world[:, i_from, :]
    body_xy = body_vec[:, :2]
    body_norm = np.linalg.norm(body_xy, axis=1, keepdims=True) + 1e-12
    body_hat = body_xy / body_norm
    perp_hat = np.stack([-body_hat[:, 1], body_hat[:, 0]], axis=1)  # 90° CCW

    forward_speed = np.sum(vel[:, :2] * body_hat, axis=1)
    lateral_speed = np.sum(vel[:, :2] * perp_hat, axis=1)

    heading = np.arctan2(body_xy[:, 1], body_xy[:, 0])  # radians
    # Unwrap then differentiate so turn rate doesn't spike at ±π wraps.
    heading_u = np.unwrap(heading)
    turn_rate = np.degrees(np.gradient(heading_u, 1.0 / cfg.fs))

    out: Dict[str, np.ndarray] = {
        "speed": speed,
        "forward_speed": forward_speed,
        "lateral_speed": lateral_speed,
        "turn_rate": turn_rate,
        "heading": heading,
    }
    if body_length is not None and np.isfinite(body_length) and body_length > 0:
        out["speed_bl"] = speed / body_length
        out["forward_speed_bl"] = forward_speed / body_length
        out["lateral_speed_bl"] = lateral_speed / body_length
    return out

def compute_com_height(
    kp_world: np.ndarray,
    kp_names: Sequence[str],
    cfg: Optional[LocomotionConfig] = None,
    centroid_kp: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """Per-frame COM Z above the estimated floor plane.

    Floor Z is the ``floor_percentile``-th percentile of Z over the
    matched ground keypoints for this single fly, using the same helper
    that ``pair_validity.compute_pair_validity`` uses, so the two
    modules agree on what "ground" means.

    Returns ``(com_z, floor_z)`` where ``com_z`` is (T,) and equals
    centroid Z minus ``floor_z``.
    """
    if cfg is None:
        cfg = LocomotionConfig()
    kp_world = np.asarray(kp_world)
    if kp_world.ndim == 2:
        kp_world = kp_world.reshape(kp_world.shape[0], -1, 3)

    names = list(kp_names)
    ground_idx = _match_indices(names, cfg.ground_kp_patterns)
    _, floor_z = _ground_valid(
        kp_world, ground_idx, cfg.floor_percentile, cfg.ground_epsilon
    )
    i_cent = names.index(centroid_kp or cfg.centroid_kp)
    com_z = kp_world[:, i_cent, 2].astype(float) - floor_z
    return com_z, floor_z

def classify_walking_state(
    speed_bl: np.ndarray, cfg: Optional[LocomotionConfig] = None
) -> np.ndarray:
    """Per-frame state ∈ {'walking', 'stopped'} from a body-length speed
    signal. Uses a hysteresis gate plus a minimum-run-length filter so
    the labels aren't dominated by single-frame jitter.
    """
    if cfg is None:
        cfg = LocomotionConfig()
    T = len(speed_bl)
    state = np.full(T, "stopped", dtype=object)
    if T == 0:
        return state

    walking = False
    for t in range(T):
        s = speed_bl[t]
        if walking:
            if s < cfg.stopped_speed_bl:
                walking = False
        else:
            if s > cfg.walking_speed_bl:
                walking = True
        state[t] = "walking" if walking else "stopped"

    # Minimum-run-length filter: collapse runs shorter than min_run_ms.
    min_run = max(1, int(round(cfg.min_run_ms * 1e-3 * cfg.fs)))
    if min_run > 1:
        state = _apply_min_run(state, min_run)
    return state

def _apply_min_run(state: np.ndarray, min_run: int) -> np.ndarray:
    out = state.copy()
    T = len(out)
    i = 0
    while i < T:
        j = i
        while j < T and out[j] == out[i]:
            j += 1
        run_len = j - i
        if run_len < min_run and i > 0:
            # Extend the previous state over this short run.
            out[i:j] = out[i - 1]
        i = j
    return out

def summarize_by_song(
    song_labels: np.ndarray,
    metrics: Dict[str, np.ndarray],
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Group per-frame metrics by song label and compute mean / std.

    Parameters
    ----------
    song_labels : (T,) object array of label strings (from the song
        classifier). Typical labels: 'pulse', 'sine', 'waggle', 'quiet'.
    metrics : dict of (T,) arrays keyed by metric name.
    valid_mask : (T,) bool or None. Only True frames are included.

    Returns
    -------
    dict ``{label: {metric: {'mean': float, 'std': float, 'n': int}}}``.
    """
    song_labels = np.asarray(song_labels)
    T = len(song_labels)
    if valid_mask is None:
        valid_mask = np.ones(T, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    unique_labels = sorted(set(song_labels.tolist()))
    for lbl in unique_labels:
        sel = valid_mask & (song_labels == lbl)
        n = int(sel.sum())
        out[lbl] = {}
        for mname, arr in metrics.items():
            a = np.asarray(arr)
            if n == 0:
                out[lbl][mname] = {"mean": float("nan"), "std": float("nan"), "n": 0}
                continue
            vals = a[sel]
            finite = np.isfinite(vals)
            if finite.sum() == 0:
                out[lbl][mname] = {"mean": float("nan"), "std": float("nan"), "n": n}
                continue
            out[lbl][mname] = {
                "mean": float(np.mean(vals[finite])),
                "std": float(np.std(vals[finite])),
                "n": n,
            }
    return out

# ======================================================================
# from utils/pulse_types.py
# ======================================================================


@dataclass
class PulseTypeConfig:
    """Hyperparameters for the Pslow/Pfast clusterer."""

    # PCA dimensionality fed into the GMM. Clemens used 8 on 10 kHz / 251-
    # sample waveforms; 6 is adequate for 20-sample 800 Hz windows.
    n_pca: int = 6
    # Number of mixture components. Always 2 (Pslow / Pfast).
    n_components: int = 2
    # GMM covariance structure. "full" is fine for 2 components in 6-D.
    covariance_type: str = "full"
    # RNG seed for the PCA/GMM fit (fully reproducible labels).
    random_state: int = 0
    # Cluster → type assignment rule. "symmetry" picks the cluster with the
    # higher mean symmetry index as Pslow (Clemens: symmetric = slow).
    label_by: str = "symmetry"

@dataclass
class PulseTypeModel:
    """Fitted clusterer plus the artifacts callers need to plot and
    classify new pulses."""

    pca: PCA
    gmm: GaussianMixture
    # {cluster_id_int: 'Pslow'|'Pfast'}
    label_map: Dict[int, str]
    # (n_components, W) mean waveform per type, in input (waveform) space.
    centroid_waveforms: np.ndarray
    # Per-component mean symmetry index (used to verify the label_map).
    component_symmetry: np.ndarray

def _as_float(waveforms: np.ndarray) -> np.ndarray:
    waves = np.asarray(waveforms, dtype=float)
    if waves.ndim != 2:
        raise ValueError(
            f"waveforms must be (N, W); got shape {waves.shape}"
        )
    return waves

def fit_pulse_type_model(
    waveforms: np.ndarray,
    symmetry: np.ndarray,
    cfg: Optional[PulseTypeConfig] = None,
) -> PulseTypeModel:
    """Fit PCA → 2-component GMM on pooled z-scored pulse waveforms.

    Parameters
    ----------
    waveforms : (N, W) float
        Per-pulse waveforms as returned by
        :func:`utils.song_analysis.extract_pulse_waveforms` (already
        z-scored and sign-aligned).
    symmetry : (N,) float
        Per-pulse symmetry index from
        :func:`utils.song_analysis.compute_pulse_symmetry`. Used only
        for cluster→type assignment, not for fitting.
    cfg : PulseTypeConfig or None

    Returns
    -------
    PulseTypeModel
        Fitted PCA + GMM plus the cluster→type map and the mean
        waveform per type.
    """
    if cfg is None:
        cfg = PulseTypeConfig()
    waves = _as_float(waveforms)
    sym = np.asarray(symmetry, dtype=float)
    if sym.shape[0] != waves.shape[0]:
        raise ValueError(
            f"symmetry length {sym.shape[0]} != waveforms.shape[0] "
            f"{waves.shape[0]}"
        )
    if waves.shape[0] < cfg.n_components:
        raise ValueError(
            f"Need at least {cfg.n_components} pulses to fit a "
            f"{cfg.n_components}-component GMM; got {waves.shape[0]}."
        )

    n_pca = min(cfg.n_pca, waves.shape[1], waves.shape[0])
    pca = PCA(n_components=n_pca, random_state=cfg.random_state)
    X = pca.fit_transform(waves)

    gmm = GaussianMixture(
        n_components=cfg.n_components,
        covariance_type=cfg.covariance_type,
        random_state=cfg.random_state,
    )
    gmm.fit(X)
    hard = gmm.predict(X)

    # Per-component artefacts: mean symmetry and mean waveform.
    comp_sym = np.zeros(cfg.n_components, dtype=float)
    centroid_waves = np.zeros((cfg.n_components, waves.shape[1]), dtype=float)
    for k in range(cfg.n_components):
        sel = hard == k
        if sel.any():
            comp_sym[k] = float(np.nanmean(sym[sel]))
            centroid_waves[k] = waves[sel].mean(axis=0)
        else:
            comp_sym[k] = np.nan

    # Cluster → type assignment.
    if cfg.label_by != "symmetry":
        raise ValueError(f"Unknown label_by: {cfg.label_by}")
    pslow_k = int(np.nanargmax(comp_sym))
    label_map = {
        k: ("Pslow" if k == pslow_k else "Pfast")
        for k in range(cfg.n_components)
    }

    return PulseTypeModel(
        pca=pca,
        gmm=gmm,
        label_map=label_map,
        centroid_waveforms=centroid_waves,
        component_symmetry=comp_sym,
    )

def classify_pulses(
    waveforms: np.ndarray, model: PulseTypeModel
) -> np.ndarray:
    """Return (N,) array of ``'Pslow'`` / ``'Pfast'`` labels via the GMM.

    Hard-assigns each pulse to its most-likely component, then maps
    cluster IDs to type names via ``model.label_map``.
    """
    waves = _as_float(waveforms)
    if waves.shape[0] == 0:
        return np.zeros(0, dtype=object)
    X = model.pca.transform(waves)
    hard = model.gmm.predict(X)
    return np.array([model.label_map[int(k)] for k in hard], dtype=object)

# ======================================================================
# from utils/pulse_type_cache.py
# ======================================================================


CACHE_VERSION = 3  # bump when the cached dict schema changes

def _gather_waveforms(results: Sequence[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (waveform, symmetry) arrays across all pairs and sides."""
    wfs, syms = [], []
    for r in results:
        for side in ('L', 'R'):
            pf = r.get('song0', {}).get('sides', {}).get(side, {}).get('pulse_features')
            if not pf:
                continue
            w = np.asarray(pf.get('waveforms', np.zeros((0, 0))))
            s = np.asarray(pf.get('symmetry', np.zeros(0)))
            if w.ndim != 2 or w.shape[0] == 0 or s.shape[0] != w.shape[0]:
                continue
            wfs.append(w)
            syms.append(s)
    if not wfs:
        return np.zeros((0, 0)), np.zeros(0)
    W = max(w.shape[1] for w in wfs)
    if any(w.shape[1] != W for w in wfs):
        padded = []
        for w in wfs:
            if w.shape[1] == W:
                padded.append(w)
            else:
                pad = np.full((w.shape[0], W - w.shape[1]), np.nan)
                padded.append(np.concatenate([w, pad], axis=1))
        wfs = padded
    waveforms = np.concatenate(wfs, axis=0)
    symmetry = np.concatenate(syms, axis=0)
    finite = np.isfinite(waveforms).all(axis=1) & np.isfinite(symmetry)
    return waveforms[finite], symmetry[finite]

def get_pulse_type_labels(
    results: Sequence[dict],
    cache_path: Optional[str | Path] = None,
    force: bool = False,
    cfg: Optional[PulseTypeConfig] = None,
    fs: float = 800.0,
) -> Dict[str, Any]:
    """Fit one PulseTypeModel pooled across all pairs and classify each side.

    Returns a dict with keys ``labels``, ``centroids``, ``counts``,
    ``pooled_waveforms``, ``fs`` (see module docstring).
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and cached.get('version') == CACHE_VERSION:
                return cached
            # stale schema — fall through to refit

    if cfg is None:
        cfg = PulseTypeConfig()
    pooled_w, pooled_s = _gather_waveforms(results)
    empty: Dict[str, Any] = {
        'version':   CACHE_VERSION,
        'labels':    {},
        'centroids': {'Pslow': np.zeros(0), 'Pfast': np.zeros(0)},
        'counts':    {'Pslow': 0, 'Pfast': 0},
        'pooled_waveforms': {'Pslow': np.zeros((0, 0)), 'Pfast': np.zeros((0, 0))},
        'fs':        float(fs),
    }
    if pooled_w.shape[0] < cfg.n_components:
        return empty
    model: PulseTypeModel = fit_pulse_type_model(pooled_w, pooled_s, cfg)

    # Classify each pulse in the pooled set so we can stratify waveforms.
    pooled_labels = classify_pulses(pooled_w, model)
    pooled_by_type = {
        name: pooled_w[pooled_labels == name]
        for name in ('Pslow', 'Pfast')
    }

    # Centroids: rebuild from pooled-classified data (matches counts).
    centroids = {
        name: (arr.mean(axis=0) if arr.shape[0] else np.zeros(pooled_w.shape[1]))
        for name, arr in pooled_by_type.items()
    }
    counts = {name: int(arr.shape[0]) for name, arr in pooled_by_type.items()}

    labels_out: Dict[int, Dict[str, np.ndarray]] = {}
    for r in results:
        pair_idx = int(r.get('pair_idx', -1))
        side_labels: Dict[str, np.ndarray] = {}
        for side in ('L', 'R'):
            pf = r.get('song0', {}).get('sides', {}).get(side, {}).get('pulse_features')
            if not pf:
                continue
            w = np.asarray(pf.get('waveforms', np.zeros((0, 0))))
            if w.ndim != 2 or w.shape[0] == 0:
                continue
            finite = np.isfinite(w).all(axis=1)
            if not finite.any():
                continue
            labels_finite = classify_pulses(w[finite], model)
            labels = np.empty(w.shape[0], dtype=object)
            labels[:] = ''
            labels[finite] = labels_finite
            side_labels[side] = labels
        if side_labels:
            labels_out[pair_idx] = side_labels

    out: Dict[str, Any] = {
        'version':          CACHE_VERSION,
        'labels':           labels_out,
        'centroids':        centroids,
        'counts':           counts,
        'pooled_waveforms': pooled_by_type,
        'fs':               float(fs),
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(out, f)
    return out

# ======================================================================
# from utils/sam3_female_com.py
# ======================================================================


def _triangulate_point(
    coeffs: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    min_cams: int = 2,
) -> np.ndarray:
    """Linear DLT triangulation of a single 3D point from multiple cameras.

    Parameters
    ----------
    coeffs : (n_cams, 11) array of DLT coefficients.
    uv     : (n_cams, 2) array of (u, v) pixel coords.
    valid  : (n_cams,) bool mask of cameras whose observation is usable.
    min_cams : minimum valid cameras to attempt triangulation.

    Returns
    -------
    (3,) world-frame point, or (3,) NaN if fewer than ``min_cams`` are valid.
    """
    mask = np.asarray(valid, dtype=bool)
    if int(mask.sum()) < int(min_cams):
        return np.full(3, np.nan, dtype=float)
    L = np.asarray(coeffs, dtype=float)[mask]
    p = np.asarray(uv, dtype=float)[mask]
    u = p[:, 0]
    v = p[:, 1]
    # Two rows per camera of the standard DLT back-projection:
    #   (L1 - u L9) X + (L2 - u L10) Y + (L3 - u L11) Z = u - L4
    #   (L5 - v L9) X + (L6 - v L10) Y + (L7 - v L11) Z = v - L8
    A_u = np.stack(
        [L[:, 0] - u * L[:, 8], L[:, 1] - u * L[:, 9], L[:, 2] - u * L[:, 10]],
        axis=1,
    )
    A_v = np.stack(
        [L[:, 4] - v * L[:, 8], L[:, 5] - v * L[:, 9], L[:, 6] - v * L[:, 10]],
        axis=1,
    )
    A = np.concatenate([A_u, A_v], axis=0)
    b = np.concatenate([u - L[:, 3], v - L[:, 7]], axis=0)
    xyz, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.asarray(xyz, dtype=float).reshape(3)

def triangulate_sam3_female_com(
    npz_path: str | Path,
    calib_dir: str | Path,
    fly_idx: int = 0,
    camera_order: Optional[Sequence[str]] = None,
    min_cams: int = 2,
    verbose: bool = True,
) -> np.ndarray:
    """Triangulate per-frame female COM from SAM3 mask centroids.

    Parameters
    ----------
    npz_path : path to ``sam3_masks.npz`` with ``valid`` (n_flies, n_cams, T)
        and ``centroids`` (n_flies, n_cams, T, 2) arrays.
    calib_dir : directory containing ``Cam*_dlt.csv`` files (sorted by filename
        to define camera ordering unless ``camera_order`` is given).
    fly_idx : which fly in the SAM3 arrays corresponds to the female (default 0).
    camera_order : optional explicit filename order (e.g. ``['Cam2012630_dlt.csv',
        ...]``); when ``None``, ``sorted(glob('Cam*_dlt.csv'))`` is used.
    min_cams : minimum valid cameras to attempt triangulation (default 2).
    verbose : if True (default), print the resolved camera order to stdout.
        Set to False to suppress the side-effect in library/batch use.

    Returns
    -------
    (T, 3) array of world-frame female COM; NaN on frames with
    fewer than ``min_cams`` valid cameras.
    """
    npz_path = Path(npz_path)
    calib_dir = Path(calib_dir)
    if camera_order is None:
        dlt_files = sorted(calib_dir.glob('Cam*_dlt.csv'))
    else:
        dlt_files = [calib_dir / name for name in camera_order]
    if not dlt_files:
        raise FileNotFoundError(f'no Cam*_dlt.csv files found in {calib_dir}')

    coeffs = np.stack([_dlt_load(f) for f in dlt_files], axis=0)  # (n_cams, 11)
    if verbose:
        print(f'[sam3_female_com] camera order:')
        for idx, f in enumerate(dlt_files):
            print(f'  cam {idx}: {f.name}')

    with np.load(npz_path) as npz:
        valid = np.asarray(npz['valid'])                            # (n_flies, n_cams, T)
        # Promote centroids to float64 so triangulation precision is not
        # bottlenecked by on-disk float32 storage.
        centroids = np.asarray(npz['centroids'], dtype=np.float64)  # (n_flies, n_cams, T, 2)

    if valid.shape[1] != coeffs.shape[0]:
        raise ValueError(
            f'SAM3 n_cams ({valid.shape[1]}) != n DLT files ({coeffs.shape[0]})'
        )

    v_fly = valid[fly_idx]                 # (n_cams, T)
    c_fly = centroids[fly_idx]             # (n_cams, T, 2)
    T = v_fly.shape[1]
    out = np.full((T, 3), np.nan, dtype=float)
    for t in range(T):
        uv_t = c_fly[:, t, :]              # (n_cams, 2)
        valid_t = v_fly[:, t]              # (n_cams,)
        finite_t = np.isfinite(uv_t).all(axis=1)
        out[t] = _triangulate_point(coeffs, uv_t, valid_t & finite_t,
                                    min_cams=min_cams)
    return out

def sam3_camera_index(calib_dir: str | Path, cam_csv_name: str) -> int:
    """Return the SAM3-axis index for ``cam_csv_name`` in ``sorted(calib_dir)``.

    The SAM3 ``packed``/``valid``/``centroids`` arrays are stored with cameras
    in ``sorted(glob('Cam*_dlt.csv'))`` order (same convention as
    :func:`triangulate_sam3_female_com`). This helper converts a camera CSV
    filename (e.g. ``'Cam2012630_dlt.csv'``) into its SAM3 camera-axis slot.
    """
    files = sorted(Path(calib_dir).glob('Cam*_dlt.csv'))
    names = [f.name for f in files]
    if cam_csv_name not in names:
        raise ValueError(f'{cam_csv_name!r} not in {names}')
    return names.index(cam_csv_name)

def unpack_sam3_masks_for_frames(
    npz_path: str | Path,
    cam_idx: int,
    fly_indices: Sequence[int],
    frame_indices: Sequence[int],
) -> list:
    """Return one unpacked mask stack per fly, indexed by ``frame_indices``.

    Output is ``[mask_fly0, mask_fly1, ...]`` where each element has shape
    ``(len(frame_indices), H_full, W_full)`` of ``bool``. Invalid frames (per
    the npz ``valid`` array) return an all-False mask for that slot.
    """
    with np.load(npz_path) as npz:
        packed = npz['packed']           # (n_flies, n_cams, T, H, W/8)
        valid = npz['valid']             # (n_flies, n_cams, T)
        shape = tuple(int(x) for x in npz['shape'])  # (H, W_full)
    H, W = shape
    fidx = np.asarray(frame_indices, dtype=int)
    out = []
    for fi in fly_indices:
        stack = np.zeros((len(fidx), H, W), dtype=bool)
        for k, t in enumerate(fidx):
            if not bool(valid[int(fi), int(cam_idx), int(t)]):
                continue
            bits = np.unpackbits(
                packed[int(fi), int(cam_idx), int(t)], axis=-1,
            )
            stack[k] = bits[:H, :W].astype(bool)
        out.append(stack)
    return out

# ======================================================================
# from utils/courtship_figure_panels.py
# ======================================================================


def _dlt_load(csv_path: str | Path) -> np.ndarray:
    """Load 11 DLT coefficients (one per line) from a ``*_dlt.csv`` file."""
    coeffs = np.loadtxt(str(csv_path)).astype(float).reshape(-1)
    if coeffs.size != 11:
        raise ValueError(
            f'expected 11 DLT coefficients in {csv_path}, got {coeffs.size}'
        )
    return coeffs

def body_pitch_deg_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """Body-axis elevation in world frame from a MuJoCo quaternion.

    ``quat_wxyz`` is ``(..., 4)`` in ``[w, x, y, z]`` order (MuJoCo free-joint
    convention). The fly model's local ``+X`` is anterior, so the world-frame
    forward vector's z component is ``2*(qx*qz - qw*qy)``; its arcsin is the
    pitch-up angle of the thorax (positive = nose up).
    """
    q = np.asarray(quat_wxyz, dtype=float)
    sin_p = 2.0 * (q[..., 1] * q[..., 3] - q[..., 0] * q[..., 2])
    return np.degrees(np.arcsin(np.clip(sin_p, -1.0, 1.0)))

# ======================================================================
# from utils/mvq_pitch_alignment.py
# ======================================================================


@dataclass(frozen=True)
class BoutRef:
    """One bout's resolved inputs. Slots come from metadata, never inference."""
    recording_dir: Path
    recording: str
    bout: str
    male_fly: int
    female_fly: int
    kp3d_path: Path
    mask_npz: Path
    calib_dir: Path

# In the source repo these two lived in `scripts/figures/export_fig4_bundle.py`
# and `utils/mvq_pitch_alignment.py` reached them through a deferred import, to
# avoid a utils -> scripts layering inversion. Here there is no layering to
# invert, so they are inlined -- and they must be, because that deferred import
# would otherwise silently reach back into the source repo. `_load_kp3d` in
# particular carries the per-file keypoint-ORDER check, which is the guard
# against measuring the wrong body part with perfectly plausible numbers.

_DEFAULT_MALE_SLOT = 1

_DEFAULT_FEMALE_SLOT = 0

def _load_kp3d(npz_path, expected_n_kp=None, expected_kp_names=None) -> np.ndarray:
    """Load the ``(T, 50, 3)`` ``kp3d`` array (TRUE DLT world frame) from a
    processed-pose-tree bout npz (``<pose_dir>/bouts/<bout>/fly{0,1}/kp3d.npz``).
    Raises a clear error naming the path and the keys actually present when
    ``kp3d`` is missing, rather than letting a bare KeyError propagate.

    Round 11 finding D: every caller indexes this array by
    ``kp_names.index(<name>)`` from the COMBINED h5's keypoint order,
    assuming `kp3d`'s own second axis shares that order. Pre-MVQ `kp3d.npz`
    files carry no keypoint-name list of their own to check against
    (verified: their only keys are `kp3d`/`conf3d`) — the order match was
    instead verified empirically (round 6): projecting `kp3d[:, 0, :]`
    (`Scutellum`, index 0 in both the combined h5's `info/kp_names` and
    `configs/anatomy/v1.yaml`'s `model.KP_NAMES` MODEL order) through the
    Cam2012630 DLT landed at uv (1123.6, 118.6), matching `kp2d` ground
    truth to within 3.3 px. `configs/detector/vitpose_v3.yaml`'s DETECTOR
    order is DIFFERENT (Scutellum at index 3) — an unreordered array would
    silently substitute a nearby keypoint (~1 mm off) and still look
    plausible.

    MVQ-era npz files ship their own ``kp_names`` array. When one is
    present AND ``expected_kp_names`` is given, the two are compared
    ELEMENTWISE and a mismatch raises, naming the first differing index --
    turning the paragraph above from an assumption into a per-file check.
    A count-only match (same length, different order) is exactly the
    keypoint-order bug CLAUDE.md documents: a collapsed/scrambled pair
    reads as a real landmark with EXCELLENT jitter and confidence, so this
    refuses to load rather than let the caller index it wrong. When
    `expected_n_kp` is given (with or without names), the array's keypoint
    axis is also asserted against it, since count is the only guard left
    for pre-MVQ files with no `kp_names` of their own.
    """
    with np.load(npz_path, allow_pickle=True) as z:
        if "kp3d" not in z.files:
            raise KeyError(f"{npz_path} has no 'kp3d' key (found: {z.files})")
        arr = np.asarray(z["kp3d"])
        on_disk = ([str(s) for s in z["kp_names"]]
                   if "kp_names" in z.files else None)
    if expected_n_kp is not None:
        assert arr.shape[1] == expected_n_kp, (
            f"{npz_path}: kp3d has {arr.shape[1]} keypoints, expected "
            f"{expected_n_kp} (len(kp_names)) — ORDER between this array "
            f"and kp_names is ASSUMED, not verified per-file; a keypoint-"
            f"count mismatch means it cannot be trusted at all")
    if on_disk is not None and expected_kp_names is not None:
        want = [str(s) for s in expected_kp_names]
        if on_disk != want:
            for i, (a, b) in enumerate(zip(on_disk, want)):
                if a != b:
                    raise ValueError(
                        f"{npz_path}: kp3d keypoint ORDER disagrees with the "
                        f"expected order at index {i}: npz has {a!r}, "
                        f"expected {b!r}. Indexing this array by the expected "
                        f"order would measure the wrong body part with "
                        f"perfectly plausible numbers -- refusing to load.")
            raise ValueError(
                f"{npz_path}: kp3d has {len(on_disk)} keypoint names, "
                f"expected {len(want)}")
    return arr

def _resolve_male_female_slots(sex_meta, sex_json=None, override=None) -> tuple:
    """Return ``(male_slot, female_slot)``.

    Round 10 finding: the mask SLOT is a FOURTH id scheme — distinct from
    the combined h5's key0/key1 pairing (round 7) and the processed tree's
    fly0/fly1 DIRECTORY naming — and was hardcoded (`fly_indices=[1, 0]`,
    `fly_idx=0`).

    Two independent sexers can disagree on which mask slot is male:
    ``sex_meta`` (the SAM3 npz's own JSON string / dict, a mask-area VOTE —
    on the real exemplar a weak one, agreement 0.429) and ``sex_json``
    (``pose/bouts/<bout>/sex.json``, a dict — sometimes a HUMAN-CONFIRMED
    review). Verified on the real exemplar: `sex_meta` says `male_slot=1`
    and `sex_json` says `male_fly=1` (`confidence="user"`,
    `method="manual-gui"`) — they AGREE here (an earlier draft of this
    docstring wrongly claimed `sex_json` said `male_fly=0`/"INVERTED"; that
    was a false comment, corrected — always re-read the real file rather
    than trust a stale claim in code). Round 10's hardcode trusted only the
    weaker `sex_meta` vote; on any bout where the two sexers disagree, that
    could triangulate the MALE as the "female" COM (a degenerate male→male
    vector) and swap the panel-A mask colours — exactly the dir-index/
    mask-slot decoupling the repo's own sexing-canonicalization memory
    warns about (round 11 finding B).

    ``sex_json``'s `male_fly` is treated as authoritative ONLY when its
    `confidence` is `"user"` or its `method` mentions "manual" (a human
    review), never for a lower-confidence/automated `sex_json`. When BOTH a
    human-confirmed `sex_json` and a `sex_meta` vote are present, they must
    AGREE — a disagreement means neither can be trusted silently, so this
    raises `ValueError` naming both values rather than picking one (the
    figure would be meaningless with a degenerate male→male vector). When
    only one source is present, that source's value is used. Falls back to
    the documented default (`_DEFAULT_MALE_SLOT`, `_DEFAULT_FEMALE_SLOT`) —
    matching the historical hardcode — only when NEITHER is present; the
    caller should print when this fallback fires.
    """
    # An explicit operator decision wins outright. The guard below refuses to
    # GUESS between disagreeing sexers; it should not block a human who has
    # looked at the evidence and decided. Provenance is returned so the bundle
    # meta records that this was an override, not an inference.
    if override is not None:
        m = int(override)
        return m, 1 - m

    human_slot = None
    if sex_json:
        confidence = str(sex_json.get("confidence", "")).lower()
        method = str(sex_json.get("method", "")).lower()
        if confidence == "user" or "manual" in method:
            human_slot = int(sex_json["male_fly"])

    auto_slot = None
    if sex_meta:
        meta = json.loads(sex_meta) if isinstance(sex_meta, (str, bytes)) else dict(sex_meta)
        auto_slot = int(meta["male_slot"])

    if human_slot is not None and auto_slot is not None:
        if human_slot != auto_slot:
            raise ValueError(
                f"sexing sources disagree on the male mask slot: "
                f"sex.json male_fly={human_slot} (human-confirmed) vs sam3 "
                f"sex_meta male_slot={auto_slot} (mask-area vote) — "
                f"refusing to guess")
        male_slot = human_slot
    elif human_slot is not None:
        male_slot = human_slot
    elif auto_slot is not None:
        male_slot = auto_slot
    else:
        return _DEFAULT_MALE_SLOT, _DEFAULT_FEMALE_SLOT
    return male_slot, 1 - male_slot

def _read_male_fly(sex_path: Path,
                   accept_authorities: Sequence[str] = ()) -> Optional[int]:
    """`male_fly` from a human-reviewed sex.json, or None.

    PORT NOTE -- `accept_authorities` is new here and defaults to EMPTY, so
    with no argument this behaves exactly as the source does: human review
    only. It exists because the `pose_v2_20260914` run has had no human
    identity-review pass, and every one of its `sex.json` files instead
    records `authority: "mvq_typed_slots"` with `confidence: "high"` and
    per-fly `sex_prob` around 0.9999 / 7.6e-05. That is a RECORDED
    determination, not the absent-value case this guard was built to catch --
    the danger below is the DELEGATE'S DEFAULT firing and inverting a recorded
    slot, which cannot happen when `male_fly` is explicitly present.

    Naming an authority here is still not sufficient on its own, and
    `make_figure4.py` does not treat it as such: it passes this together with
    `h5_male_fly` built from the combined h5's own `info/sex` and
    `on_conflict="raise"`, so the tree's answer and the h5's answer must
    agree bout by bout. Two independent records agreeing is a stronger
    guarantee than a single human pass, and a disagreement is fatal.

    Important-1 (final review): this used to delegate the confidence/method
    policy entirely to `_resolve_male_female_slots(None, meta)` and trust
    whatever it returned. That function's contract is to fall back to a
    hardcoded `_DEFAULT_MALE_SLOT=1` when NEITHER a human `sex_json` nor an
    `sex_meta` vote is present -- correct for its OTHER callers, which want
    an opinion. Called here with `sex_meta=None`, that fallback fires for
    every non-human-reviewed `sex.json` too: demonstrated,
    `{"male_fly": 0, "confidence": "auto"}` came back as `1` -- the RECORDED
    SLOT INVERTED, with no raise and no skip. This function must never guess,
    so the confidence/method test is applied HERE, before delegating, and a
    sex.json that fails it returns None (the bout is then skipped by
    `find_mvq_bouts`) rather than reaching the delegate's default at all.
    Do not remove this check and go back to a bare delegate call --
    `_resolve_male_female_slots`'s own default must stay as-is for its other
    callers (see its docstring), so the guard belongs on this side.
    """
    try:
        meta = json.loads(sex_path.read_text())
    except Exception:                            # noqa: BLE001
        return None
    if "male_fly" not in meta:
        return None
    confidence = str(meta.get("confidence", "")).lower()
    method = str(meta.get("method", "")).lower()
    authority = str(meta.get("authority", "")).lower()
    accepted = {str(a).lower() for a in accept_authorities}
    if (confidence != "user" and "manual" not in method
            and authority not in accepted):
        return None
    try:
        male, _ = _resolve_male_female_slots(None, meta)
    except Exception:                            # noqa: BLE001
        return None
    return int(male)

def find_mvq_bouts(
    processed_root,
    video_root,
    pose_dir: str = "pose_v2_20260914",
    h5_male_fly: Optional[Dict[Tuple[str, str], int]] = None,
    on_conflict: str = "raise",
    skipped: Optional[List[str]] = None,
    accept_authorities: Sequence[str] = (),
) -> List[BoutRef]:
    """Discover every MVQ bout with the inputs the alignment needs.

    Walks `<processed_root>/<bucket>/<recording>/<pose_dir>/bouts/bout_*`.
    A bout is included only when its `sex.json`, both flies' `kp3d.npz`, its
    `sam3_masks.npz` and its recording's `calibration/` are all present;
    anything incomplete is skipped rather than guessed at.

    Important-5 (final review): when `skipped` is given, every skip along the
    way -- a missing `calibration/` dir, a missing/non-human-reviewed
    `sex.json`, or a missing `kp3d.npz`/`sam3_masks.npz` -- appends a short,
    specific reason naming the recording/bout and the path that was missing.
    Before this, all four of those were a bare `continue`: if
    `--courtship-video-root` pointed at the wrong tree, EVERY recording's
    `calibration/` would 404 and the whole pool would empty with nothing in
    `skipped` to say why -- the only message anywhere blamed
    `--processed-root` instead (the arg actually responsible for kp3d/masks/
    sex.json, not calibration). Do not go back to a silent `continue` for
    any of these four checks.

    `h5_male_fly` optionally maps `(recording, bout) -> male_fly` from the
    combined h5's `info/male_fly`. When a bout appears in it with a value in
    {0, 1} that disagrees with `sex.json`, this is a genuine conflict: the two
    are independent records of the same human review, and a disagreement
    means neither can be trusted silently. A value outside {0, 1} (observed:
    -1, the h5's "not sexed" sentinel) carries no opinion and is skipped
    rather than treated as a claim about which fly is male -- unaffected by
    `on_conflict`, in either mode.

    `on_conflict` governs a genuine 0-vs-1 conflict:
      * "raise" (default) aborts discovery entirely, naming the bout. This is
        correct for the LIVE pipeline and must stay the default: measured
        160/160 agreement between `sex.json` and the current combined h5's
        `info/male_fly` on this dataset, so a conflict here is a real defect
        worth stopping for.
      * "skip" excludes just the disputed bout (never guesses a slot for it)
        and, if `skipped` is given, appends a human-readable reason to it so
        the caller can report it rather than have the bout silently vanish.
        This exists for rebuilding the LEGACY comparison figure: the legacy
        h5's `info/male_fly` is -1 (the "unsexed" sentinel, already tolerated
        above) for 238 of 300 entries and, where it DOES have an opinion,
        disagrees with the LATER human review (`sex.json`, 2026-08-29,
        status=confirmed) for exactly one bout
        (Session1/2026_04_02_12_11_50/bout_00004) -- a stale source that
        should declare itself explicitly via an opt-in flag rather than the
        live pipeline quietly becoming more permissive.
    An unrecognised `on_conflict` is an error, not a silent fallback to
    either behaviour.
    """
    if on_conflict not in ("raise", "skip"):
        raise ValueError(
            f"on_conflict must be 'raise' or 'skip', got {on_conflict!r}")
    processed_root, video_root = Path(processed_root), Path(video_root)
    out: List[BoutRef] = []
    for bouts_dir in sorted(processed_root.glob(f"*/*/{pose_dir}/bouts")):
        rec_dir = bouts_dir.parent.parent
        recording = f"{rec_dir.parent.name}/{rec_dir.name}"
        calib_dir = video_root / recording / "calibration"
        if not calib_dir.is_dir():
            if skipped is not None:
                skipped.append(
                    f"align_violin recording: {recording}: no calibration/ "
                    f"at {calib_dir} -- check --courtship-video-root, not "
                    f"--processed-root")
            continue
        for bout_dir in sorted(bouts_dir.glob("bout_*")):
            bout = bout_dir.name
            sex_path = bout_dir / "sex.json"
            male = _read_male_fly(sex_path, accept_authorities)
            if male is None:
                if skipped is not None:
                    reason = ("missing" if not sex_path.exists() else
                              "present but its authority is not accepted "
                              "(or it is unreadable)")
                    skipped.append(
                        f"align_violin bout: {recording}/{bout}: sex.json "
                        f"{reason} ({sex_path})")
                continue
            if h5_male_fly is not None:
                want = h5_male_fly.get((recording, bout))
                # A value outside {0, 1} is the h5's "not sexed" sentinel, not
                # a claim that the male is fly -1 (or fly 60) -- measured on
                # ik_output_combined_v1_courtship_both.h5: info/male_fly is
                # -1 for 238 of its 300 entries (plus 1 "60", 2 "0"), while
                # the matching sex.json still carries a real human-reviewed
                # answer. Raising on those would empty a whole figure panel
                # over bouts that were simply never labelled in the h5. Only
                # a genuine 0-vs-1 disagreement is a real conflict, and only
                # THAT is subject to `on_conflict`.
                if want is not None and int(want) in (0, 1) and int(want) != male:
                    reason = (
                        f"{recording}/{bout}: sex.json says male_fly={male} "
                        f"but the combined h5's info/male_fly says {want} -- "
                        f"refusing to guess which fly is the male")
                    if on_conflict == "raise":
                        raise ValueError(reason)
                    # on_conflict == "skip": exclude this one bout, never
                    # guess a slot for it, but let the caller report it.
                    if skipped is not None:
                        skipped.append(
                            f"align_violin bout: {reason} "
                            f"(on_conflict='skip')")
                    continue
            kp3d = bout_dir / f"fly{male}" / "kp3d.npz"
            other_kp3d = bout_dir / f"fly{1 - male}" / "kp3d.npz"
            mask = rec_dir / "sam3_masks" / bout / "sam3_masks.npz"
            missing = [str(p) for p in (kp3d, other_kp3d, mask)
                      if not p.exists()]
            if missing:
                if skipped is not None:
                    skipped.append(
                        f"align_violin bout: {recording}/{bout}: missing "
                        f"{missing}")
                continue
            out.append(BoutRef(
                recording_dir=rec_dir, recording=recording, bout=bout,
                male_fly=male, female_fly=1 - male,
                kp3d_path=kp3d, mask_npz=mask, calib_dir=calib_dir))
    return out

def _body_pitch_deg(head: np.ndarray, scut: np.ndarray) -> np.ndarray:
    """Elevation of the head above the scutellum, in degrees.

    Copied unchanged from `utils.sam3_aligned_bouts._body_pitch_deg` so the
    panel keeps measuring exactly what it published.
    """
    vec = head - scut
    n = np.linalg.norm(vec, axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.degrees(np.arcsin(np.divide(
            vec[..., 2], n, out=np.full_like(n, np.nan), where=n > 0)))

def _target_pitch_deg(male_scut: np.ndarray,
                      female_com: np.ndarray) -> np.ndarray:
    """Elevation of the female from the male's scutellum, in degrees."""
    vec = female_com - male_scut
    n = np.linalg.norm(vec, axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.degrees(np.arcsin(np.divide(
            vec[..., 2], n, out=np.full_like(n, np.nan), where=n > 0)))

def _female_com(bout: BoutRef, kp_scale: float) -> np.ndarray:
    """Triangulated female COM in the KP frame, from this bout's masks.

    The camera order is taken from the npz's OWN `cameras` array rather than
    `triangulate_sam3_female_com`'s glob-sorted default: the two are not
    guaranteed to agree, and using the wrong one silently triangulates one
    camera's centroid against another's calibration. This mirrors what the
    exporter already does for the exemplar.

    `kp_scale` converts DLT units to the KP frame, matching the exporter's
    `/ args.kp_scale` on the same call.
    """
    # `triangulate_sam3_female_com` is defined above in this merged
    # module; the source's deferred import would reach the source repo.
    with np.load(bout.mask_npz, allow_pickle=True) as z:
        cams = z["cameras"] if "cameras" in z.files else None
    camera_order = ([f"{c}_dlt.csv" for c in [str(x) for x in cams]]
                    if cams is not None else None)
    com = triangulate_sam3_female_com(
        str(bout.mask_npz), str(bout.calib_dir), fly_idx=bout.female_fly,
        camera_order=camera_order, min_cams=2, verbose=False)
    return np.asarray(com, float) / kp_scale

def compute_pitch_alignment_mvq(
    bouts,
    *,
    head_name: str = "Antenna_Base",
    scut_name: str = "Scutellum",
    kp_scale: float = 0.1,
    expected_kp_names=None,
    scut_com_offset_tol: Optional[float] = None,
) -> dict:
    """Per-bout ``body_pitch - target_pitch`` traces and their median |value|.

    Returns the same keys `compute_pitch_alignment_all_sessions` did --
    `alignment_per_bout`, `bout_names`, `median_abs_alignment_deg` -- so the
    exporter and the `courtship.pitch_violin` panel need no change, plus
    `recordings` (the widened pool spans eleven, so a bout name alone is no
    longer unique) and `skipped` (bouts that yielded no finite frame, reported
    rather than silently dropped).

    When `scut_com_offset_tol` is given, the male's own mask COM is
    triangulated and compared against his `kp3d` Scutellum, in the same
    world units as `kp3d` (not verified to be mm -- see the module's other
    callers; do not call this quantity "mm" anywhere, including in the
    raised error below). This is a UNIT-SANITY check, not an agreement
    check: the expected distance is ~4.7 world units, NOT ~0, because a
    triangulated 2D-mask centroid is not the scutellum. That offset is a
    stable systematic bias -- measured 2026-09-11 over 22 bouts spanning all
    eleven recordings, per-bout medians ran 3.72 to 5.33 (median-of-medians
    4.68). Dropping the `/ kp_scale` conversion instead gives ~178, so the
    default tolerance of 15.0 passes every observed bout with 2.8x headroom
    while catching a 10x unit error with ~12x margin.

    It ALSO discriminates a male/female SLOT SWAP, not just a unit error --
    an accidental finding worth preserving on purpose: measured 2026-09-11
    over 4 bouts across 3 recordings, the CORRECT slot gives 4.71-6.27 while
    the WRONG slot (male_com computed against the female's own scutellum
    instead of his) gives 25.1-54.9 -- 15.0 sits with 1.7x margin below the
    lowest wrong-slot value and 3.3x above the highest correct one. Do NOT
    tighten this toward the ~4.7 unit-check value or loosen it toward the
    ~178 unit-error value without re-checking this margin -- either move
    could silently remove the slot guard this tolerance also happens to
    provide.

    Do NOT tighten this toward zero: a `kp_scale` error tilts `target_pitch`
    while leaving `body_pitch` untouched, which no smoothness or residual
    check would catch -- but the two quantities are genuinely different
    points on the animal, and a tight bound would reject correct data.
    """
    per_bout: List[np.ndarray] = []
    summaries: List[float] = []
    names: List[str] = []
    recordings: List[str] = []
    skipped: List[str] = []

    for b in bouts:
        # Important-3 (final review): `expected_n_kp` must travel WITH
        # `expected_kp_names`, not be omitted. `_load_kp3d`'s order check
        # only runs when the npz carries its OWN `kp_names` -- every
        # pre-reprocessing (legacy) file has none, so on that tree
        # `kp_names` a few lines down falls back to `expected_kp_names`
        # itself, and `kp_names.index(...)` below would run against an
        # array that was never checked for length OR order. Passing the
        # count here keeps the one guard that still applies on the legacy
        # path (`_load_kp3d`'s `expected_n_kp` assert), so a keypoint-count
        # mismatch still raises instead of indexing silently wrong.
        kp = _load_kp3d(b.kp3d_path,
                        expected_n_kp=(len(expected_kp_names)
                                       if expected_kp_names else None),
                        expected_kp_names=expected_kp_names)
        with np.load(b.kp3d_path, allow_pickle=True) as z:
            kp_names = ([str(s) for s in z["kp_names"]]
                        if "kp_names" in z.files else list(expected_kp_names or []))
        if head_name not in kp_names or scut_name not in kp_names:
            raise KeyError(
                f"{b.recording}/{b.bout}: kp3d lacks {head_name!r}/{scut_name!r} "
                f"(has {kp_names[:6]}...)")
        female = _female_com(b, kp_scale)
        T = min(len(kp), len(female))
        head = kp[:T, kp_names.index(head_name), :]
        scut = kp[:T, kp_names.index(scut_name), :]

        if scut_com_offset_tol is not None:
            # `_female_com` triangulates BoutRef.female_fly, so point that
            # field at the male to get HIS mask COM. dataclasses.replace, not
            # `**b.__dict__`, which relies on a frozen dataclass exposing one.
            male_com = _female_com(
                dataclasses.replace(b, female_fly=b.male_fly), kp_scale)
            k = min(T, len(male_com))
            d = np.linalg.norm(male_com[:k] - scut[:k], axis=-1)
            med = float(np.nanmedian(d)) if np.isfinite(d).any() else np.nan
            if not (med <= scut_com_offset_tol):
                raise ValueError(
                    f"{b.recording}/{b.bout}: male mask COM sits "
                    f"{med:.3f} world units from his kp3d Scutellum "
                    f"(tol {scut_com_offset_tol} world units) -- the mask "
                    f"COM and kp3d are not in the same frame, so "
                    f"target_pitch would be tilted against body_pitch (or "
                    f"the male/female slots are swapped -- see this "
                    f"function's docstring for the measured slot-swap "
                    f"magnitude)")

        alignment = _body_pitch_deg(head, scut) - _target_pitch_deg(scut, female[:T])
        finite = np.isfinite(alignment)
        if not finite.any():
            skipped.append(f"{b.recording}/{b.bout}: no finite frame")
        per_bout.append(alignment)
        names.append(b.bout)
        recordings.append(b.recording)
        summaries.append(float(np.median(np.abs(alignment[finite])))
                         if finite.any() else float("nan"))

    return {
        "alignment_per_bout": per_bout,
        "bout_names": names,
        "recordings": recordings,
        "median_abs_alignment_deg": np.asarray(summaries, float),
        "skipped": skipped,
    }

# ======================================================================
# from utils/courtship_loader.py
# ======================================================================


def load_courtship_h5(
    h5_path: str | Path,
    enable_jax: bool = False,
) -> Tuple[dict, dict, List[str], List[str]]:
    """Load a combined courtship h5 and extract metadata.

    Returns
    -------
    data : dict
        Full h5 contents (bout dicts + 'info').
    info : dict
        The ``data['info']`` sub-dict.
    kp_names : list[str]
        Keypoint names from ``info['kp_names']``.
    bout_keys : list[str]
        Sorted bout key names (excluding 'info').
    """
    data = load(str(h5_path), enable_jax=enable_jax)
    info = data.get('info', {}) or {}

    # kp_names may be stored as list or dict
    raw_names = info.get('kp_names', info.get('site_names_egocentric', []))
    if isinstance(raw_names, dict):
        kp_names = [raw_names[k] for k in sorted(raw_names.keys(), key=lambda x: int(x))]
    else:
        kp_names = list(raw_names)

    bout_keys = sorted(k for k in data.keys() if k != 'info')
    return data, info, kp_names, bout_keys

def pair_bouts(
    bout_keys: Sequence[str],
    info: dict,
) -> List[Tuple[str, str]]:
    """Pair consecutive fly0/fly1 bout keys using info['source_flies'].

    Falls back to simple even/odd pairing when source_flies is absent.
    """
    src = list(info.get('source_flies', []))
    bucket = list(info.get('bucket', []))
    pairs: List[Tuple[str, str]] = []

    if src and len(src) == len(bout_keys):
        i = 0
        while i + 1 < len(bout_keys):
            if (src[i] == 'fly0' and src[i + 1] == 'fly1'
                    and (not bucket or bucket[i] == 'both' == bucket[i + 1])):
                pairs.append((bout_keys[i], bout_keys[i + 1]))
                i += 2
            else:
                i += 1
    else:
        for i in range(0, len(bout_keys) - 1, 2):
            pairs.append((bout_keys[i], bout_keys[i + 1]))
    return pairs

def get_fields(
    bout: dict,
    despike: bool = True,
    despike_iterations: int = 1,
    medfilt_clean: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract kp_data, xpos_egocentric, qpos from a single bout dict.

    Parameters
    ----------
    bout : dict
        A single bout's data (e.g. ``data['bout_000']``).
    despike : bool
        Apply velocity-reversal spike removal.
    despike_iterations : int
        Number of despiking passes.  1 = single-frame only (safe for male
        song).  Use higher values (e.g. 6) for non-singing flies where
        multi-frame tracking glitches are common.
    medfilt_clean : bool
        After despiking, also apply a median-filter pass to catch
        multi-frame tracking excursions that lack a velocity reversal.
        **Not safe for signals with fast oscillations (male wing song).**
        Only enable for non-singing flies.

    Returns
    -------
    kp : ndarray, shape (T, N, 3)
        World-frame keypoints.
    xpos_ego : ndarray or None, shape (T, N, 3)
        Egocentric site positions (if present).
    qpos : ndarray or None, shape (T, D)
        Joint angles (if present).
    """
    kp = np.asarray(bout['kp_data'])
    if kp.ndim == 2:
        kp = kp.reshape(kp.shape[0], -1, 3)
    if despike:
        kp, _ = despike_isolated_spikes(kp, max_iterations=despike_iterations)
    if medfilt_clean:
        kp, _ = medfilt_despike(kp)

    xp = bout.get('xpos_egocentric')
    if xp is not None:
        xp = np.asarray(xp)
        if xp.ndim == 2:
            xp = xp.reshape(xp.shape[0], -1, 3)
        if despike:
            xp, _ = despike_isolated_spikes(xp, max_iterations=despike_iterations)
        if medfilt_clean:
            xp, _ = medfilt_despike(xp)

    qp = bout.get('qpos')
    if qp is not None:
        qp = np.asarray(qp)
        if despike:
            qp, _ = despike_isolated_spikes(qp, max_iterations=despike_iterations)
        if medfilt_clean:
            qp, _ = medfilt_despike(qp)

    return kp, xp, qp

def analyze_pair(
    key0: str,
    key1: str,
    bout0: dict,
    bout1: dict,
    kp_names: Sequence[str],
    *,
    song_cfg: Optional[SongAnalysisConfig] = None,
    sex_cfg: Optional[SexIdConfig] = None,
    loc_cfg: Optional[LocomotionConfig] = None,
    pair_cfg: Optional[PairValidityConfig] = None,
    despike: bool = True,
    repair_wing_swaps: bool = True,
) -> dict:
    """Run full per-pair analysis: song, sex ID, male-first reorder, locomotion.

    After this call, slot-0 is always male and slot-1 is always female.
    Tracker-slot fields are preserved for provenance.

    Parameters
    ----------
    key0, key1 : str
        Bout keys (e.g. 'bout_000', 'bout_001').
    bout0, bout1 : dict
        Bout data dicts from the h5.
    kp_names : sequence of str
        Keypoint names.
    song_cfg, sex_cfg, loc_cfg, pair_cfg : config objects, optional
        Analysis configs. Defaults are created if None.
    despike : bool
        Apply spike removal when loading fields.

    Returns
    -------
    dict with keys: key0, key1, T, valid_fly0, valid_fly1, colocated,
        song0, song1, sex, tracker_key0, tracker_key1, tracker_male_id,
        tracker_song_fraction_fly0, tracker_song_fraction_fly1,
        male_labels, male_valid, kin, com_z, floor_z, walking_state, by_song.
    """
    if song_cfg is None:
        song_cfg = SongAnalysisConfig()
    if sex_cfg is None:
        sex_cfg = SexIdConfig()
    if loc_cfg is None:
        loc_cfg = LocomotionConfig()
    if pair_cfg is None:
        pair_cfg = PairValidityConfig()

    kp0, xp0, q0 = get_fields(bout0, despike=despike)
    kp1, xp1, q1 = get_fields(bout1, despike=despike)

    # Clip to common length
    T = min(len(kp0), len(kp1))
    kp0, kp1 = kp0[:T], kp1[:T]
    if xp0 is not None: xp0 = xp0[:T]
    if xp1 is not None: xp1 = xp1[:T]
    if q0  is not None: q0  = q0[:T]
    if q1  is not None: q1  = q1[:T]

    # Inter-fly wing-tip identity repair — fixes short-run JARVIS
    # multi-animal swaps where fly0's wing tip was briefly assigned to fly1
    # (or vice versa) during close contact.  Only modifies WingL_V13 /
    # WingR_V13; all other keypoints pass through.
    if repair_wing_swaps:
        kp0, kp1, _ = repair_wing_tip_identity_swaps(kp0, kp1, kp_names)

    # Pair validity
    pv = compute_pair_validity(kp0, kp1, kp_names, cfg=pair_cfg)
    v0 = np.asarray(pv['valid_fly0']).astype(bool)
    v1 = np.asarray(pv['valid_fly1']).astype(bool)
    coloc = np.asarray(pv['pair_colocated']).astype(bool)

    # Song analysis
    song0 = analyze_fly_song(kp0, xp0, q0, kp_names, cfg=song_cfg, valid_mask=v0)
    song1 = analyze_fly_song(kp1, xp1, q1, kp_names, cfg=song_cfg, valid_mask=v1)

    # Sex ID
    sex = identify_male_female(song0, song1, kp0, kp1, kp_names, cfg=sex_cfg)

    # Preserve tracker-slot assignment before swapping
    tracker_song_fraction_fly0 = float(song0['summary']['song_fraction'])
    tracker_song_fraction_fly1 = float(song1['summary']['song_fraction'])
    tracker_key0 = key0
    tracker_key1 = key1
    tracker_male_id = sex['male_id']

    # Reorder so slot-0 = male, slot-1 = female
    if sex['male_id'] == 'fly1':
        key0, key1 = key1, key0
        kp0, kp1 = kp1, kp0
        xp0, xp1 = xp1, xp0
        q0, q1 = q1, q0
        v0, v1 = v1, v0
        song0, song1 = song1, song0
        sex['male_id'] = 'fly0'
        sex['female_id'] = 'fly1'

    # Locomotion for male (slot-0)
    bl = sex['body_length_male'] if np.isfinite(sex['body_length_male']) else None
    kin = compute_centroid_velocity(kp0, kp_names, loc_cfg, body_length=bl)
    com_z, floor_z = compute_com_height(kp0, kp_names, loc_cfg)
    speed_bl = kin.get('speed_bl', kin['speed'])
    wstate = classify_walking_state(np.asarray(speed_bl), loc_cfg)

    # Dominant-wing frame labels
    dw = str(song0.get('dominant_wing', 'L')).upper()
    side_key = 'L' if dw.startswith('L') else 'R'
    song_labels = np.asarray(song0['sides'][side_key]['frame_labels'])

    # Song-conditioned aggregates
    metrics = {
        'forward_speed_bl': np.asarray(kin.get('forward_speed_bl', kin['forward_speed'])),
        'speed_bl':         np.asarray(speed_bl),
        'turn_rate':        np.asarray(kin['turn_rate']),
        'com_z':            np.asarray(com_z),
    }
    by_song = summarize_by_song(song_labels, metrics, valid_mask=v0)

    return {
        'key0': key0, 'key1': key1,
        'T': T,
        'valid_fly0': v0, 'valid_fly1': v1, 'colocated': coloc,
        'song0': song0, 'song1': song1,
        'sex': sex,
        'tracker_key0': tracker_key0,
        'tracker_key1': tracker_key1,
        'tracker_male_id': tracker_male_id,
        'tracker_song_fraction_fly0': tracker_song_fraction_fly0,
        'tracker_song_fraction_fly1': tracker_song_fraction_fly1,
        'male_labels': song_labels,
        'male_valid':  v0,
        'kin': kin,
        'com_z': com_z, 'floor_z': floor_z,
        'walking_state': wstate,
        'by_song': by_song,
    }

def analyze_all_pairs(
    data: dict,
    pairs: List[Tuple[str, str]],
    kp_names: Sequence[str],
    *,
    cache_path: Optional[str | Path] = None,
    force: bool = False,
    song_cfg: Optional[SongAnalysisConfig] = None,
    sex_cfg: Optional[SexIdConfig] = None,
    loc_cfg: Optional[LocomotionConfig] = None,
    pair_cfg: Optional[PairValidityConfig] = None,
    despike: bool = True,
    repair_wing_swaps: bool = True,
    min_song_bout_frames: Optional[int] = 100,
    min_bilateral_dz_p95: Optional[float] = 12.0,
    verbose: bool = True,
) -> List[dict]:
    """Analyze all pairs with optional pickle caching.

    Parameters
    ----------
    data : dict
        Full h5 data dict.
    pairs : list of (key0, key1)
        Output of :func:`pair_bouts`.
    kp_names : sequence of str
        Keypoint names.
    cache_path : path, optional
        Pickle cache location. Loads from cache if it exists and force=False.
    force : bool
        Re-run analysis even if cache exists.
    min_song_bout_frames : int, optional
        Drop bouts shorter than this many frames (125 frames = 156 ms at
        800 Hz). Set to ``None`` or 0 to disable.
    min_bilateral_dz_p95 : float, optional
        Drop bouts whose 95th percentile of ``max(|dZ/dt|_L, |dZ/dt|_R)`` on
        the male's wing tips falls below this (mm/s). Filters out bouts with
        no meaningful wing-extension activity. Set to ``None`` to disable.
    verbose : bool
        Print progress every 25 pairs and emit a filter summary at the end.

    Returns
    -------
    list of dict
        Per-pair result dicts from :func:`analyze_pair`.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            if verbose:
                print(f'loading cache: {cache_path}')
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

    _song_cfg_eff = song_cfg if song_cfg is not None else SongAnalysisConfig()
    fs = float(_song_cfg_eff.fs)

    results = []
    dropped: list[tuple[int, str, str, str]] = []
    for i, (k0, k1) in enumerate(pairs):
        try:
            r = analyze_pair(
                k0, k1, data[k0], data[k1], kp_names,
                song_cfg=song_cfg, sex_cfg=sex_cfg,
                loc_cfg=loc_cfg, pair_cfg=pair_cfg,
                despike=despike,
                repair_wing_swaps=repair_wing_swaps,
            )
            r['pair_idx'] = i

            reason: Optional[str] = None
            T = int(r.get('T', 0))
            if min_song_bout_frames and T < int(min_song_bout_frames):
                reason = f'T={T} < {int(min_song_bout_frames)}'
            elif min_bilateral_dz_p95 is not None:
                wd = r['song0']['wing_data']
                z_L = np.asarray(wd[_song_cfg_eff.left_tip ]['z'], dtype=float)
                z_R = np.asarray(wd[_song_cfg_eff.right_tip]['z'], dtype=float)
                if z_L.size and z_R.size:
                    dz_L = np.abs(np.diff(z_L, prepend=z_L[0]) * fs)
                    dz_R = np.abs(np.diff(z_R, prepend=z_R[0]) * fs)
                    dz_max = np.maximum(dz_L, dz_R)
                    finite = dz_max[np.isfinite(dz_max)]
                    p95 = float(np.percentile(finite, 95)) if finite.size else 0.0
                    r['bilateral_dz_p95'] = p95
                    if p95 < float(min_bilateral_dz_p95):
                        reason = f'dz_p95={p95:.2f} < {float(min_bilateral_dz_p95):.2f}'

            if reason is None:
                r['filtered_idx'] = len(results)
                results.append(r)
            else:
                dropped.append((i, k0, k1, reason))
        except Exception as e:
            if verbose:
                print(f'  pair {i} ({k0}/{k1}): {type(e).__name__}: {e}')
        if verbose and (i + 1) % 25 == 0:
            print(f'  processed {i + 1}/{len(pairs)}')

    if verbose and dropped:
        print(f'  filtered out {len(dropped)}/{len(pairs)} non-singing pairs '
              f'(min_T={min_song_bout_frames}, '
              f'min_dz_p95={min_bilateral_dz_p95}):')
        for i, k0, k1, why in dropped:
            print(f'    pair {i:>3d} ({k0}/{k1}): {why}')

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
        if verbose:
            print(f'cached {len(results)} pair results -> {cache_path}')

    return results
