"""The nine Figure 4 panel plotters, plus their spec -> plotter adapters.

Extracted from `3d_tracking_dataset/utils/courtship_figure_panels.py` by an
AST trace of what the figure calls: 42 top-level definitions and 2343 lines
reduce to 15, because everything touching cv2, MuJoCo or video decoding is on
the bundle-export path rather than the plotting path. Bodies are unchanged.

Each plotter takes `(ax, *arrays, **style)` and draws into an axes it does not
own, which is why the source repo's figbuilder tile-compose machinery was
droppable: a plain `fig.add_axes(rect)` satisfies the same contract. The
adapters at the end of this file are the only part of figbuilder that was
needed -- they translate a panel's `spec` from fig4.json into plotter keyword
arguments, then apply shared axes cosmetics.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SONG_COLORS: Dict[str, str] = {
    'pulse':  '#F97316',  # warm orange  (pulse song → F, H)
    'sine':   '#14B8A6',  # teal         (sine song  → F, H)
    'waggle': '#9467bd',
    'quiet':  '#bdbdbd',
}

_SEG_FILL: Dict[str, str] = {
    'pulse':  '#E76F5133',
    'sine':   '#2A9D8F33',
    'waggle': '#E9C46A33',
    'quiet':  '#cccccc11',
}

_SEG_EDGE: Dict[str, str] = {
    'pulse':  '#E76F51',
    'sine':   '#2A9D8F',
    'waggle': '#E9C46A',
    'quiet':  '#999999',
}

WING_COLORS: Dict[str, str] = {
    'WingL_V13': '#A855F7',  # purple
    'WingR_V13': '#38BDF8',  # blue
}

PULSE_TYPE_COLORS: Dict[str, str] = {
    'Pslow': '#fb8c00',  # orange
    'Pfast': '#1565c0',  # deep blue
}

def _hex_to_fill(hex_rgb: str, alpha: float = 0.20) -> str:
    """Append an alpha byte to a ``#rrggbb`` hex color; pass through ``#rrggbbaa``."""
    s = str(hex_rgb)
    if s.startswith('#') and len(s) == 7:
        return f'{s}{int(round(alpha * 255)):02x}'
    return s

def _colored_text_legend(
    ax: plt.Axes,
    handles_labels: Optional[Tuple[list, list]] = None,
    **legend_kwargs,
):
    """Draw a legend whose text entries are colored to match their series
    (no line/box marker). Replaces the standard handle+label legend.

    ``handles_labels`` defaults to ``ax.get_legend_handles_labels()``. Any
    extra ``legend_kwargs`` are forwarded to ``ax.legend`` and override the
    marker-hiding defaults if the caller wants to tweak placement/spacing.
    """
    if handles_labels is None:
        handles, labels = ax.get_legend_handles_labels()
    else:
        handles, labels = handles_labels
    if not handles:
        return None

    defaults = {
        'frameon': False,
        'handlelength': 0,
        'handletextpad': 0,
        'borderpad': 0.2,
        'labelspacing': 0.25,
    }
    defaults.update(legend_kwargs)
    leg = ax.legend(handles, labels, **defaults)

    colors = []
    for h in handles:
        c = None
        # axvspan Patches use a very pale fill but a saturated edgecolor —
        # prefer edgecolor so the legend text reads at full strength.
        for getter in ('get_color', 'get_edgecolor', 'get_facecolor'):
            if hasattr(h, getter):
                try:
                    c = getattr(h, getter)()
                except Exception:
                    c = None
                if c is not None:
                    break
        colors.append(c if c is not None else 'k')

    # Strip alpha so labels render at full opacity even when derived from
    # semi-transparent shading (facecolor 0x33) or invisible edgelines.
    solid_colors = []
    for c in colors:
        rgba = mpl.colors.to_rgba(c)
        if rgba[3] <= 0.0:
            solid_colors.append('k')
        else:
            solid_colors.append((rgba[0], rgba[1], rgba[2], 1.0))
    for txt, color in zip(leg.get_texts(), solid_colors):
        txt.set_color(color)
    for h in leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles:
        try:
            h.set_visible(False)
        except Exception:
            pass
    return leg

def _merge_segments_by_type(
    segs_list: Sequence[Iterable[dict]],
) -> List[dict]:
    """Union overlapping/adjacent same-type intervals across multiple segment
    lists (e.g. L + R sides) so downstream shading paints each region once
    and there's no double-shading overlap where sides agree.
    """
    by_type: Dict[str, List[Tuple[int, int]]] = {}
    for segs in segs_list:
        for s in segs or []:
            by_type.setdefault(s['type'], []).append(
                (int(s['start']), int(s['end']))
            )
    merged: List[dict] = []
    for t, ivs in by_type.items():
        ivs.sort()
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append({'type': t, 'start': cur_s, 'end': cur_e})
                cur_s, cur_e = s, e
        merged.append({'type': t, 'start': cur_s, 'end': cur_e})
    merged.sort(key=lambda d: d['start'])
    return merged

def _resolve_pulse_subtype(
    seg: dict,
    peak_frames: Optional[np.ndarray],
    subtype_labels: Optional[np.ndarray],
) -> Optional[str]:
    """Majority Pslow/Pfast label among pulses whose peaks fall in ``seg``.

    Returns ``None`` if the segment isn't a pulse segment, or the pulse-type
    inputs are missing / have no peaks inside the segment.
    """
    if seg.get('type') != 'pulse':
        return None
    if peak_frames is None or subtype_labels is None:
        return None
    pf = np.asarray(peak_frames)
    sl = np.asarray(subtype_labels)
    if pf.size == 0 or sl.size == 0 or pf.size != sl.size:
        return None
    s, e = int(seg['start']), int(seg['end'])
    m = (pf >= s) & (pf <= e)
    if not m.any():
        return None
    inside = sl[m]
    n_slow = int((inside == 'Pslow').sum())
    n_fast = int((inside == 'Pfast').sum())
    if n_slow == 0 and n_fast == 0:
        return None
    return 'Pslow' if n_slow >= n_fast else 'Pfast'

def _shade_segments(
    ax: plt.Axes,
    segments: Iterable[dict],
    fs: float,
    frame_range: Optional[Tuple[int, int]] = None,
    seen_labels: Optional[set] = None,
    skip: Tuple[str, ...] = ('quiet',),
    fills: Optional[Dict[str, str]] = None,
    edges: Optional[Dict[str, str]] = None,
    pulse_peak_frames: Optional[np.ndarray] = None,
    pulse_subtype_labels: Optional[np.ndarray] = None,
    pulse_type_colors: Optional[Dict[str, str]] = None,
    fill_alpha: float = 0.20,
    time_unit: str = 'ms',
) -> None:
    """Paint axvspans for each non-quiet song segment.

    When ``pulse_peak_frames`` + ``pulse_subtype_labels`` are supplied, pulse
    segments are colored per the dominant pulse sub-type (Pslow/Pfast). Caller
    may override fills / edges via ``fills`` / ``edges`` dicts (shallow-merged
    over module defaults), and pulse-subtype colors via ``pulse_type_colors``.
    """
    if seen_labels is None:
        seen_labels = set()
    _fills = {**_SEG_FILL, **(fills or {})}
    _edges = {**_SEG_EDGE, **(edges or {})}
    _pt = {**PULSE_TYPE_COLORS, **(pulse_type_colors or {})}
    _ms_scale = 1e-3 if time_unit == 's' else 1.0

    for seg in segments:
        stype = seg['type']
        if stype in skip:
            continue
        s, e = int(seg['start']), int(seg['end'])
        if frame_range is not None:
            lo, hi = frame_range
            if e <= lo or s >= hi:
                continue
            s, e = max(s, lo), min(e, hi)
        s_ms = s / fs * 1000.0 * _ms_scale
        e_ms = e / fs * 1000.0 * _ms_scale

        key = stype
        fill = _fills.get(stype, '#cccccc33')
        edge = _edges.get(stype)
        sub = _resolve_pulse_subtype(seg, pulse_peak_frames, pulse_subtype_labels)
        if sub is not None:
            key = sub
            fill = _hex_to_fill(_pt[sub], fill_alpha)
            edge = _pt[sub]

        label = key.capitalize() if key not in seen_labels else None
        seen_labels.add(key)
        ax.axvspan(s_ms, e_ms, facecolor=fill, edgecolor=edge,
                   linewidth=0, zorder=0, label=label)

def panel_wing_z_traces(
    ax: plt.Axes,
    t_ms: np.ndarray,
    wingL_z: np.ndarray,
    wingR_z: np.ndarray,
    segments_L: Iterable[dict],
    segments_R: Iterable[dict],
    fs: float = 800.0,
    frame_range: Optional[Tuple[int, int]] = None,
    pulse_type_side: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    wing_colors: Optional[Dict[str, str]] = None,
    pulse_type_colors: Optional[Dict[str, str]] = None,
    line_kwargs: Optional[Dict] = None,
    legend_kwargs: Optional[Dict] = None,
    pulse_vline_kwargs: Optional[Dict] = None,
    min_segment_ms: float = 0.0,
    time_unit: str = 'ms',
) -> None:
    """Plot left/right wing-V13 z-position with song shading + per-pulse markers.

    Segment shading is drawn once over the union of L+R segments (so L/R
    overlap does not double-shade). Each individual pulse event is marked
    with a vertical line colored by its own Pslow / Pfast label, so every
    pulse is individually identifiable.

    Parameters
    ----------
    pulse_type_side : optional {'L': {'peak_frames': ndarray, 'labels': ndarray},
        'R': {...}}. When provided, vertical lines are drawn at each
        ``peak_frame`` colored per the corresponding Pslow/Pfast label.
    wing_colors, pulse_type_colors : shallow-merge overrides for module defaults.
    line_kwargs : extra kwargs passed to ``ax.plot`` for both wing traces.
    pulse_vline_kwargs : extra kwargs passed to ``ax.axvline`` for pulse markers.
    legend_kwargs : extra kwargs forwarded to ``ax.legend``.
    """
    wc = {**WING_COLORS, **(wing_colors or {})}
    pt = {**PULSE_TYPE_COLORS, **(pulse_type_colors or {})}
    lk = {'lw': 0.7, **(line_kwargs or {})}
    vk = {'lw': 1.0, 'alpha': 0.9, 'zorder': 4, **(pulse_vline_kwargs or {})}
    lg = {'loc': 'lower left', 'bbox_to_anchor': (0.38, 0.02),
          'ncols': 1, 'columnspacing': 0.6, **(legend_kwargs or {})}

    _ms_scale = 1e-3 if time_unit == 's' else 1.0

    seen: set = set()
    merged = _merge_segments_by_type([segments_L, segments_R])
    if min_segment_ms > 0:
        min_frames = max(1, int(round(float(min_segment_ms) * fs / 1000.0)))
        merged = [s for s in merged
                  if (int(s['end']) - int(s['start'])) >= min_frames]
    _shade_segments(ax, merged, fs, frame_range=frame_range, seen_labels=seen,
                    time_unit=time_unit)

    peaks_all: List[np.ndarray] = []
    labs_all: List[np.ndarray] = []
    if pulse_type_side:
        for side in ('L', 'R'):
            d = pulse_type_side.get(side) or {}
            pf = np.asarray(d.get('peak_frames', []))
            lb = np.asarray(d.get('labels', []))
            if pf.size and lb.size == pf.size:
                peaks_all.append(pf)
                labs_all.append(lb)
    if peaks_all:
        pf_all = np.concatenate(peaks_all)
        lb_all = np.concatenate(labs_all).astype(str)
        order = np.argsort(pf_all)
        pf_all, lb_all = pf_all[order], lb_all[order]
        if pf_all.size >= 2:
            keep = np.concatenate(([True], np.diff(pf_all) > 1))
            pf_all, lb_all = pf_all[keep], lb_all[keep]
        if frame_range is not None:
            lo, hi = frame_range
            m = (pf_all >= lo) & (pf_all < hi)
            pf_all, lb_all = pf_all[m], lb_all[m]
        for f, lab in zip(pf_all, lb_all):
            tm = float(f) / fs * 1000.0 * _ms_scale
            color = pt.get(lab, 'k')
            label = lab if lab not in seen else None
            seen.add(lab)
            ax.axvline(tm, color=color, label=label, **vk)

    t_plot = np.asarray(t_ms, dtype=float) * _ms_scale
    ax.plot(t_plot, wingL_z, color=wc['WingL_V13'], label='Wing L V13', **lk)
    ax.plot(t_plot, wingR_z, color=wc['WingR_V13'], label='Wing R V13', **lk)
    ax.set_ylabel('Wing V13\nz (mm)')
    if t_plot.size:
        ax.set_xlim(0.0, float(t_plot[-1]))

    # Keep only song-type (Sine/Pulse) and wing-trace labels; Pslow/Pfast
    # per-pulse markers are visible as vertical lines but not in the legend.
    _keep = {'Sine', 'Pulse', 'Wing L V13', 'Wing R V13'}
    _h, _l = ax.get_legend_handles_labels()
    _pairs = [(h, l) for h, l in zip(_h, _l) if l in _keep]
    if _pairs:
        _hs, _ls = zip(*_pairs)
        _colored_text_legend(ax, handles_labels=(list(_hs), list(_ls)), **lg)
    else:
        _colored_text_legend(ax, **lg)

def panel_scutellum_z_trace(
    ax: plt.Axes,
    t_ms: np.ndarray,
    scutellum_z: np.ndarray,
    segments: Optional[Iterable[dict]] = None,
    fs: float = 800.0,
    frame_range: Optional[Tuple[int, int]] = None,
    pulse_peak_frames: Optional[np.ndarray] = None,
    pulse_subtype_labels: Optional[np.ndarray] = None,
    pulse_type_colors: Optional[Dict[str, str]] = None,
    line_color: str = 'k',
    line_kwargs: Optional[Dict] = None,
    time_unit: str = 'ms',
) -> None:
    """Plot scutellum (body) z-position over the same time interval.

    When ``pulse_peak_frames`` + ``pulse_subtype_labels`` are provided, pulse
    segments are shaded by dominant Pslow/Pfast type.
    """
    lk = {'lw': 0.7, **(line_kwargs or {})}
    _ms_scale = 1e-3 if time_unit == 's' else 1.0
    if segments is not None:
        _shade_segments(ax, segments, fs, frame_range=frame_range,
                        pulse_peak_frames=pulse_peak_frames,
                        pulse_subtype_labels=pulse_subtype_labels,
                        pulse_type_colors=pulse_type_colors,
                        time_unit=time_unit)
    t_plot = np.asarray(t_ms, dtype=float) * _ms_scale
    ax.plot(t_plot, scutellum_z, color=line_color, **lk)
    ax.set_xlabel('Time (s)' if time_unit == 's' else 'Time (ms)')
    ax.set_ylabel('Scutellum\nz (mm)')

def panel_male_pitch(
    ax: plt.Axes,
    t_ms: np.ndarray,
    male_pitch_deg: np.ndarray,
    target_pitch_deg: np.ndarray,
    segments: Optional[Iterable[dict]] = None,
    fs: float = 800.0,
    frame_range: Optional[Tuple[int, int]] = None,
    min_segment_ms: float = 10.0,
    male_color: str = '#d62728',
    target_color: str = '#1f77b4',
    zero_line_color: str = '#bdbdbd',
    line_kwargs: Optional[Dict] = None,
    legend_kwargs: Optional[Dict] = None,
    title: str = '',
    time_unit: str = 'ms',
) -> None:
    """Plot male thorax pitch (red) and target pitch to the female (blue).

    ``male_pitch_deg`` is the thorax body-axis elevation (positive = nose up)
    and ``target_pitch_deg`` is the elevation of the male-scutellum → female-COM
    vector. Both are degrees; where the two traces overlap, the male is aimed
    at the female. Song segments are shaded behind the traces via the same
    `_shade_segments` helper as Panel B; ``min_segment_ms`` filters ultra-short
    detections.
    """
    lk = {'lw': 0.8, **(line_kwargs or {})}
    lg = {'loc': 'upper left', 'ncols': 2,
          'columnspacing': 0.6, **(legend_kwargs or {})}
    _ms_scale = 1e-3 if time_unit == 's' else 1.0

    if segments is not None:
        segs = list(segments)
        if min_segment_ms > 0:
            min_frames = max(1, int(round(float(min_segment_ms) * fs / 1000.0)))
            segs = [s for s in segs
                    if (int(s['end']) - int(s['start'])) >= min_frames]
        _shade_segments(ax, segs, fs, frame_range=frame_range,
                        time_unit=time_unit)

    t_plot = np.asarray(t_ms, dtype=float) * _ms_scale
    ax.axhline(0.0, color=zero_line_color, lw=0.6, zorder=1)
    ax.plot(t_plot, male_pitch_deg, color=male_color, linestyle='-',
            label='Male', zorder=3, **lk)
    ax.plot(t_plot, target_pitch_deg, color=target_color, linestyle='-',
            label='Female', zorder=2, **lk)
    ax.set_xlabel('Time (s)' if time_unit == 's' else 'Time (ms)')
    ax.set_ylabel('Pitch (°)')
    if t_plot.size:
        ax.set_xlim(0.0, float(t_plot[-1]))
    if title:
        ax.set_title(title, pad=2)
    _colored_text_legend(ax, **lg)

def panel_pitch_alignment_violin(
    ax: plt.Axes,
    per_bout_values: Sequence[float],
    exemplar_idx: Optional[int] = None,
    violin_color: str = '#c0c0c0',
    dot_color: str = '#555555',
    exemplar_color: str = '#d62728',
    jitter_width: float = 0.12,
    rng_seed: int = 0,
    title: str = '',
) -> None:
    """Violin + per-bout dots of median |pitch alignment| across bouts.

    Each dot is one bout's median absolute alignment (degrees); the
    violin shows the distribution across all bouts. ``exemplar_idx``
    highlights that bout in ``exemplar_color``.
    """
    vals = np.asarray(per_bout_values, dtype=float)
    finite_mask = np.isfinite(vals)
    finite = vals[finite_mask]
    if finite.size >= 2:
        parts = ax.violinplot(
            finite, positions=[0], widths=0.7, showextrema=False,
            showmedians=False,
        )
        for body in parts['bodies']:
            body.set_facecolor(violin_color)
            body.set_edgecolor('none')
            body.set_alpha(0.55)
    rng = np.random.default_rng(rng_seed)
    jitter = rng.uniform(-jitter_width, jitter_width, size=vals.size)
    colors = [
        exemplar_color if (exemplar_idx is not None and i == int(exemplar_idx))
        else dot_color
        for i in range(vals.size)
    ]
    sizes = [
        20.0 if (exemplar_idx is not None and i == int(exemplar_idx))
        else 10.0
        for i in range(vals.size)
    ]
    zorders = [
        4 if (exemplar_idx is not None and i == int(exemplar_idx))
        else 3
        for i in range(vals.size)
    ]
    for i in range(vals.size):
        if not finite_mask[i]:
            continue
        ax.scatter(
            jitter[i], vals[i], s=sizes[i], c=colors[i],
            edgecolors='k', linewidths=0.3, zorder=zorders[i],
        )
    if finite.size:
        med = float(np.median(finite))
        ax.hlines(med, -0.35, 0.35, color='k', lw=0.8, zorder=5)
    ax.set_xticks([0])
    ax.set_xticklabels([f'n={int(finite.size)}'])
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylabel('|Pitch align| (°)')
    if title:
        ax.set_title(title, pad=2)

def panel_z_height_singing_vs_walking(
    ax: plt.Axes,
    pulse_z: np.ndarray,
    sine_z: np.ndarray,
    walking_z: np.ndarray,
    kind: str = 'box',
    colors: Optional[Sequence[str]] = None,
    alpha: float = 0.55,
    box_kwargs: Optional[Dict] = None,
    violin_kwargs: Optional[Dict] = None,
    show_points: bool = True,
    point_kwargs: Optional[Dict] = None,
    jitter_width: float = 0.15,
    rng_seed: int = 0,
    title: str = 'z height by state',
) -> None:
    """Compare scutellum z-height during pulse, sine, and free walking.

    ``colors`` overrides the default (pulse/sine/walking) triplet. When
    ``show_points`` is True (default), raw samples are jittered and scattered
    on top of each box/violin.
    """
    p = np.asarray(pulse_z,   dtype=float); p = p[np.isfinite(p)]
    s = np.asarray(sine_z,    dtype=float); s = s[np.isfinite(s)]
    w = np.asarray(walking_z, dtype=float); w = w[np.isfinite(w)]
    data = [p, s, w]
    labels = [
        f'pulse\n(n={p.size})',
        f'sine\n(n={s.size})',
        f'free walk\n(n={w.size})',
    ]
    cols = list(colors) if colors else [
        SONG_COLORS['pulse'], SONG_COLORS['sine'], '#888888',
    ]
    positions = [0, 1, 2]

    if show_points:
        pk = {'s': 6, 'linewidths': 0.3, 'edgecolors': 'k',
              'alpha': 0.7, 'zorder': 1, **(point_kwargs or {})}
        rng = np.random.default_rng(rng_seed)
        for pos, vals, c in zip(positions, data, cols):
            if vals.size == 0:
                continue
            jitter = rng.uniform(-jitter_width, jitter_width, size=vals.size)
            ax.scatter(pos + jitter, vals, c=c, **pk)

    if kind == 'violin':
        vk = {'widths': 0.7, 'showmeans': True, 'showextrema': False,
              **(violin_kwargs or {})}
        parts = ax.violinplot(data, positions=positions, **vk)
        for pc, c in zip(parts['bodies'], cols):
            pc.set_facecolor(c); pc.set_alpha(alpha)
            pc.set_edgecolor('k'); pc.set_linewidth(0.4)
            pc.set_zorder(2)
        for key in ('cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes'):
            lc = parts.get(key)
            if lc is not None:
                lc.set_color('k'); lc.set_linewidth(0.9); lc.set_zorder(3)
    else:
        bk = {'widths': 0.55, 'patch_artist': True, 'showfliers': False,
              'medianprops': dict(color='k', lw=0.8),
              'whiskerprops': dict(lw=0.6),
              'capprops':     dict(lw=0.6),
              'boxprops':     dict(lw=0.6),
              **(box_kwargs or {})}
        bp = ax.boxplot(data, positions=positions, **bk)
        for patch, c in zip(bp['boxes'], cols):
            patch.set_facecolor(c); patch.set_alpha(alpha)
            patch.set_zorder(2)
        for key in ('medians', 'whiskers', 'caps'):
            for line in bp.get(key, []):
                line.set_zorder(3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Scutellum z (mm)')
    ax.set_title(title, pad=2)

WING_PHASE_COLORS: Dict[str, str] = {
    'extended': WING_COLORS['WingR_V13'],  # blue   (matches Wing R)
    'folded':   WING_COLORS['WingL_V13'],  # purple (matches Wing L)
}

def panel_sine_wing_inphase(
    ax: plt.Axes,
    t_ms: np.ndarray,
    wing_extended_z: np.ndarray,
    wing_folded_z: np.ndarray,
    fs: float = 800.0,
    frame_range: Optional[Tuple[int, int]] = None,
    sine_segments: Optional[Iterable[dict]] = None,
    colors: Optional[Dict[str, str]] = None,
    line_kwargs: Optional[Dict] = None,
    legend_kwargs: Optional[Dict] = None,
    title: str = 'Sine song: extended + folded wing in phase',
) -> None:
    """Overlaid extended- vs folded-wing V13 z traces over a slice.

    Caller resolves which wing is extended at each frame (typically by
    comparing per-frame extension angles) and passes the two resulting traces
    here.
    """
    cc = {**WING_PHASE_COLORS, **(colors or {})}
    lk = {'lw': 0.8, **(line_kwargs or {})}
    lg = {'loc': 'upper center', **(legend_kwargs or {})}

    t_ms = np.asarray(t_ms, dtype=float)
    wing_extended_z = np.asarray(wing_extended_z, dtype=float)
    wing_folded_z = np.asarray(wing_folded_z, dtype=float)
    # `frame_range` was declared and documented ("over a slice") but never
    # applied, so this panel always drew the WHOLE bout: at 2508 ms the wing
    # beat aliases into an apparent slow drift and the in-phase relationship
    # the panel exists to show is invisible. Slice here, as the sibling
    # `panel_wing_z_traces` does (:1567).
    if frame_range is not None:
        lo, hi = int(frame_range[0]), int(frame_range[1])
        sl = slice(max(lo, 0), min(hi, t_ms.size))
        t_ms = t_ms[sl]
        wing_extended_z = wing_extended_z[sl]
        wing_folded_z = wing_folded_z[sl]

    ax.plot(t_ms, wing_extended_z, color=cc['extended'],
            label='Extending', **lk)
    ax.plot(t_ms, wing_folded_z, color=cc['folded'],
            label='Folding', **lk)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Wing V13 z (mm)')
    ax.set_title(title, pad=2)
    _colored_text_legend(ax, **lg)

def _gaussian_kde_1d(
    x: np.ndarray, grid: np.ndarray, bw: Optional[float] = None
) -> np.ndarray:
    """Lightweight 1-D Gaussian KDE (Silverman bandwidth by default)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.zeros_like(grid, dtype=float)
    if bw is None:
        sd = float(np.std(x, ddof=1))
        if sd <= 0:
            sd = float(np.std(x))
        if sd <= 0:
            return np.zeros_like(grid, dtype=float)
        # Silverman's rule of thumb for univariate Gaussian KDE.
        bw = 1.06 * sd * n ** (-1.0 / 5.0)
    if bw <= 0:
        return np.zeros_like(grid, dtype=float)
    # Vectorised Gaussian sum; chunk to bound memory if needed.
    diff = (grid[:, None] - x[None, :]) / bw
    K = np.exp(-0.5 * diff * diff) / np.sqrt(2.0 * np.pi)
    return K.sum(axis=1) / (n * bw)

def panel_joint_angle_density(
    ax: plt.Axes,
    ext_pulse: np.ndarray,
    ext_sine: np.ndarray,
    bins: int = 40,
    range_deg: Tuple[float, float] = (0.0, 180.0),
    colors: Optional[Dict[str, str]] = None,
    hist_alpha: float = 0.25,
    kde_lw: float = 1.4,
    show_hist: bool = True,
    show_kde: bool = True,
    bw: Optional[float] = None,
    title: str = 'Wing angle: pulse vs sine',
    legend_kwargs: Optional[Dict] = None,
) -> None:
    """Overlaid 1-D histogram + KDE of the extended-wing angle by song state.

    Two distributions are drawn on a single axes:

    * Pulse  (solid, ``SONG_COLORS['pulse']``)
    * Sine   (solid, ``SONG_COLORS['sine']``)

    Histograms (probability-density normalised) are drawn translucent and
    KDE curves (Silverman bandwidth) are overlaid on top.
    """
    sc = {'pulse': SONG_COLORS['pulse'], 'sine': SONG_COLORS['sine'],
          **(colors or {})}
    edges = np.linspace(range_deg[0], range_deg[1], int(bins) + 1)
    grid = np.linspace(range_deg[0], range_deg[1], 400)

    series = (
        ('Pulse', ext_pulse, sc['pulse'], '-'),
        ('Sine',  ext_sine,  sc['sine'],  '-'),
    )

    handles, labels = [], []
    for label, x, color, ls in series:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            continue
        if show_hist:
            ax.hist(x, bins=edges, density=True, color=color,
                    alpha=hist_alpha, histtype='stepfilled',
                    edgecolor='none')
        if show_kde:
            y = _gaussian_kde_1d(x, grid, bw=bw)
            (line,) = ax.plot(grid, y, color=color, lw=kde_lw,
                              linestyle=ls, label=label)
            handles.append(line); labels.append(label)
        else:
            handles.append(plt.Line2D([], [], color=color, lw=kde_lw,
                                      linestyle=ls))
            labels.append(label)

    ax.set_xlim(*range_deg)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Wing angle (deg)')
    ax.set_ylabel('Density')
    ax.set_title(title, pad=2)
    lg = {'loc': 'upper right', **(legend_kwargs or {})}
    if handles:
        _colored_text_legend(ax, handles_labels=(handles, labels), **lg)

def panel_wing_phase_polar(
    ax: plt.Axes,
    phase_rad: np.ndarray,
    bins: int = 36,
    color: Optional[str] = None,
    density: bool = True,
    mean_vector: bool = True,
    center_stat: str = 'mean',
    title: str = 'L–R wing phase (sine)',
    bar_kwargs: Optional[Dict] = None,
    mean_kwargs: Optional[Dict] = None,
) -> None:
    """Polar histogram of L-vs-R wing phase difference (radians).

    Parameters
    ----------
    ax : matplotlib polar axes (must be created with ``projection='polar'``).
    phase_rad : 1-D array of phase differences (radians, any wrapping).
    bins : number of equal-width angular bins over [-pi, pi].
    density : if True, normalise so the histogram integrates to 1 over 2*pi
        (probability density per radian).
    mean_vector : if True, draw a marker on the rim at the center direction.
    center_stat : ``'mean'`` (default) uses the circular mean arg(mean(e^{iθ}));
        ``'median'`` uses the circular median (angle minimising the sum of
        circular distances Σ π − |π − |θᵢ − α||).
    """
    if not getattr(ax, 'name', '') == 'polar':
        raise ValueError('panel_wing_phase_polar requires a polar axes')

    color = color or SONG_COLORS['sine']
    bk = {'edgecolor': 'white', 'linewidth': 0.4, 'alpha': 0.85,
          **(bar_kwargs or {})}
    mk = {'color': '#222222', 'linewidth': 1.4, **(mean_kwargs or {})}

    x = np.asarray(phase_rad, dtype=float)
    x = x[np.isfinite(x)]
    x = np.angle(np.exp(1j * x))                  # wrap to [-pi, pi]

    if x.size == 0:
        ax.set_title(title, pad=2)
        return

    edges = np.linspace(-np.pi, np.pi, int(bins) + 1)
    counts, _ = np.histogram(x, bins=edges)
    width = 2 * np.pi / bins
    centers = edges[:-1] + width / 2.0
    if density and counts.sum() > 0:
        heights = counts / (counts.sum() * width)
    else:
        heights = counts.astype(float)

    bar_width = 0.65 * width
    ax.bar(centers, heights, width=bar_width, color=color,
           bottom=0.0, **bk)

    rmax = float(heights.max()) if heights.size else 1.0
    if rmax <= 0.0:
        rmax = 1.0
    ax.set_ylim(0.0, rmax * 1.08)

    if mean_vector and x.size > 1:
        stat = str(center_stat).lower()
        if stat == 'median':
            # Circular median: angle α minimising Σ (π − |π − |xᵢ − α||).
            diffs = np.abs(x[:, None] - x[None, :])
            dist = np.pi - np.abs(np.pi - diffs)
            r_ang = float(x[int(np.argmin(dist.sum(axis=1)))])
        elif stat == 'mean':
            r_ang = float(np.angle(np.mean(np.exp(1j * x))))
        else:
            raise ValueError(
                f"center_stat must be 'mean' or 'median', got {center_stat!r}")
        ax.plot([r_ang], [rmax * 1.04],
                marker='o', markersize=4.5,
                markerfacecolor=mk.get('color', '#222222'),
                markeredgecolor='white', markeredgewidth=0.6,
                linestyle='none', zorder=5, clip_on=False)
        # ax.text(0.5, 0.5, f'|R|={r_len:.2f}',
        #         transform=ax.transAxes, ha='center', va='center',
        #         fontsize=6, color='#222222')

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_thetalim(-np.pi, np.pi)
    ax.set_thetagrids(
        [0, 90, 180, -90],
        labels=['0', 'π/2', '±π', '-π/2'],
        fontsize=6,
    )
    ax.set_rgrids(
        np.linspace(rmax / 3.0, rmax, 3),
        labels=[''] * 3,
    )
    ax.tick_params(axis='x', pad=-2)
    ax.grid(True, color='#bbbbbb', linewidth=0.4, alpha=0.8)
    ax.set_facecolor('#f4f4f4')
    for spine in ax.spines.values():
        spine.set_color('#888888')
        spine.set_linewidth(0.6)
    ax.set_title(title, pad=2, fontsize=7)

def panel_pulse_classification(
    ax: plt.Axes,
    pulse_type_results: Dict[str, object],
    show_std: bool = True,
    show_examples: bool = False,
    max_examples: int = 40,
    colors: Optional[Dict[str, str]] = None,
    mean_kwargs: Optional[Dict] = None,
    std_alpha: float = 0.20,
    example_kwargs: Optional[Dict] = None,
    legend_kwargs: Optional[Dict] = None,
    title: str = 'Pslow vs Pfast waveform',
) -> None:
    """Plot mean Pslow and Pfast pulse waveforms.

    Parameters
    ----------
    pulse_type_results : dict returned by
        :func:`utils.pulse_type_cache.get_pulse_type_labels`
        (keys: ``centroids``, ``counts``, ``pooled_waveforms``, ``fs``).
    show_std : if True, shade +/- 1 std around each mean.
    show_examples : if True, draw up to ``max_examples`` individual
        waveforms per class as thin semi-transparent lines.
    """
    centroids = pulse_type_results.get('centroids', {}) or {}
    counts = pulse_type_results.get('counts', {}) or {}
    pooled = pulse_type_results.get('pooled_waveforms', {}) or {}
    fs = float(pulse_type_results.get('fs', 800.0))

    pt = {**PULSE_TYPE_COLORS, **(colors or {})}
    mk = {'lw': 1.2, **(mean_kwargs or {})}
    ek = {'lw': 0.3, 'alpha': 0.15, **(example_kwargs or {})}
    lg = {'loc': 'upper right', 'borderaxespad': 0.2,
          **(legend_kwargs or {})}

    slow = np.asarray(centroids.get('Pslow', np.zeros(0)))
    fast = np.asarray(centroids.get('Pfast', np.zeros(0)))

    if slow.size == 0 and fast.size == 0:
        ax.text(0.5, 0.5, 'no pulses', ha='center', va='center',
                transform=ax.transAxes)
        return

    W = max(slow.size, fast.size)
    t_ms = (np.arange(W) - W / 2.0) / fs * 1000.0  # centered on pulse peak

    for name, mean_wf in (('Pslow', slow), ('Pfast', fast)):
        if mean_wf.size == 0:
            continue
        color = pt[name]
        n = counts.get(name, 0)

        if show_examples:
            pool = np.asarray(pooled.get(name, np.zeros((0, 0))))
            if pool.size > 0:
                if pool.shape[0] > max_examples:
                    rng = np.random.default_rng(0)
                    sel = rng.choice(pool.shape[0], max_examples, replace=False)
                    pool = pool[sel]
                ax.plot(t_ms[:pool.shape[1]], pool.T,
                        color=color, zorder=1, **ek)

        if show_std:
            pool = np.asarray(pooled.get(name, np.zeros((0, 0))))
            if pool.shape[0] > 1:
                std = pool.std(axis=0)
                ax.fill_between(t_ms[:mean_wf.size],
                                mean_wf - std, mean_wf + std,
                                color=color, alpha=std_alpha,
                                linewidth=0, zorder=2)

        ax.plot(t_ms[:mean_wf.size], mean_wf, color=color,
                label=f'{name} (n={n})', zorder=3, **mk)

    ax.axhline(0, color='k', lw=0.3, alpha=0.4)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('wing z (z-score)')
    ax.set_title(title, pad=2)
    _colored_text_legend(ax, **lg)


# ======================================================================
# spec -> plotter adapters
#
# Ported from `figbuilder/panels/courtship.py`, whose PanelType subclasses did
# exactly one useful thing: translate a panel's `spec` dict from fig4.json
# into the plotter's keyword arguments, then apply shared axes cosmetics. The
# registry, JSON schemas and browser-editor plumbing around them are not
# needed to draw a figure, so only the `draw` bodies came across -- unchanged,
# because each one encodes a styling decision the published figure depends on
# (panel D's rebased time axis, panel G's "free running" relabel, panel C's
# pinned x limits).
# ======================================================================

#: The assay is FREE RUNNING, not free walking. `panel_z_height_singing_vs_
#: walking` hardcodes a 'free walk\n(n=...)' tick label; the zheight adapter
#: substitutes this in after drawing, preserving the '(n=...)' suffix.
FREE_LABEL = "free running"

_PULSE_TYPES = ("Pslow", "Pfast")


def array_to_segments(arr) -> List[dict]:
    """Structured segment array -> the list-of-dicts the plotters expect."""
    return [{"start": int(r["start"]), "end": int(r["end"]),
             "type": r["type"].decode("utf-8") if isinstance(r["type"], bytes)
                     else str(r["type"])} for r in arr]


def _segs_arg(value) -> List[dict]:
    """Accept a structured segment array or an already-decoded list."""
    if value is None:
        return []
    arr = np.asarray(value)
    if arr.dtype.names:
        return array_to_segments(arr)
    return list(value)


def _frame_range(spec):
    fr = spec.get("frame_range")
    return None if fr is None else (int(fr[0]), int(fr[1]))


def _legend_kwargs(spec):
    lk = spec.get("legend_kwargs")
    return lk if isinstance(lk, dict) and lk else None


def _draw_zoom_marker(ax, spec) -> None:
    """Outline `spec['zoom_marker']` (a FRAME range) on an already-drawn axes.

    A zoomed companion panel is uninterpretable unless the reader can see
    where in the parent trace it came from; the published Figure 4 marks it
    with a dashed rectangle. The range is in FRAMES and converted here with
    the axes' own fs/time_unit, because the trace's x axis may be ms or s.
    """
    zm = spec.get("zoom_marker")
    if zm is None:
        return
    from matplotlib.patches import Rectangle

    fs = float(spec.get("fs", 800.0))
    scale = 1000.0 if str(spec.get("time_unit", "s")) == "ms" else 1.0
    x0, x1 = (float(zm[0]) / fs * scale, float(zm[1]) / fs * scale)
    y0, y1 = ax.get_ylim()
    ax.add_patch(Rectangle(
        (x0, y0), x1 - x0, y1 - y0, transform=ax.transData,
        facecolor="none", edgecolor="k", linestyle="--", linewidth=0.8,
        zorder=10, clip_on=False))
    ax.set_ylim(y0, y1)


def apply_cosmetics(ax, spec) -> None:
    """Shared per-panel cosmetics, applied AFTER the plotter has drawn.

    The plotters are consumed unmodified, so anything to change about the
    resulting axes -- spines, tick labels, legend placement -- has to happen
    afterwards on the axes object. Every option is absent-by-default.
    """
    if not spec:
        return
    spines = spec.get("spines")
    if isinstance(spines, dict):
        for name, visible in spines.items():
            spine = ax.spines.get(name)
            if spine is not None:
                spine.set_visible(bool(visible))
    if spec.get("hide_xticklabels"):
        ax.tick_params(bottom=False, labelbottom=False)
    if spec.get("hide_yticklabels"):
        ax.tick_params(left=False, labelleft=False)
    legend = spec.get("legend")
    if isinstance(legend, dict):
        leg = ax.get_legend()
        if leg is None:
            return
        if legend.get("hide"):
            leg.remove()
            return
        bbox = legend.get("bbox_to_anchor")
        if bbox is not None:
            leg.set_bbox_to_anchor(tuple(float(v) for v in bbox))
        if legend.get("loc") is not None:
            leg.set_loc(legend["loc"])


def draw_wing_z(ax, data, spec):
    pts = {}
    for side in ("L", "R"):
        pf, lb = data.get(f"pulse_{side}_peak_frames"), data.get(f"pulse_{side}_labels")
        if pf is None or lb is None:
            continue
        pf = np.asarray(pf).ravel()
        lb = np.asarray([str(x) for x in np.asarray(lb).ravel()])
        if pf.size and pf.size == lb.size:
            pts[side] = {"peak_frames": pf, "labels": lb}
    panel_wing_z_traces(
        ax, np.asarray(data["t_ms"]), np.asarray(data["wingL_z"]),
        np.asarray(data["wingR_z"]),
        _segs_arg(data.get("seg_L")), _segs_arg(data.get("seg_R")),
        pulse_type_side=pts or None, fs=float(spec.get("fs", 800.0)),
        frame_range=_frame_range(spec),
        min_segment_ms=float(spec.get("min_segment_ms", 0.0)),
        time_unit=spec.get("time_unit", "s"), legend_kwargs=_legend_kwargs(spec))
    _draw_zoom_marker(ax, spec)


def draw_scutellum_z(ax, data, spec):
    # Shade the UNION of L+R, matching the wing panel above -- seg_L alone
    # gave panel C's two rows different backgrounds.
    segs = _segs_arg(data.get("segments")) + _segs_arg(data.get("segments_R"))
    panel_scutellum_z_trace(
        ax, np.asarray(data["t_ms"]), np.asarray(data["scutellum_z"]),
        segments=segs, fs=float(spec.get("fs", 800.0)),
        frame_range=_frame_range(spec), line_color=spec.get("line_color", "k"),
        time_unit=spec.get("time_unit", "s"))
    # matplotlib's 5% x-margins inset this row while the wing row above spans
    # fully, so the two did not line up. Pin both to the data extent.
    t_arr = np.asarray(data["t_ms"], float)
    if t_arr.size:
        scale = 1.0 if spec.get("time_unit", "s") == "ms" else 1.0 / 1000.0
        ax.set_xlim(t_arr[0] * scale, t_arr[-1] * scale)
    ax.margins(x=0)


def draw_male_pitch(ax, data, spec):
    panel_male_pitch(
        ax, np.asarray(data["t_ms"]), np.asarray(data["male_pitch"]),
        np.asarray(data["target_pitch"]), segments=None,
        fs=float(spec.get("fs", 800.0)), frame_range=_frame_range(spec),
        male_color=spec.get("male_color", "#d62728"),
        target_color=spec.get("target_color", "#1f77b4"),
        time_unit=spec.get("time_unit", "s"), legend_kwargs=_legend_kwargs(spec))


def draw_video_kp(ax, data, spec):
    """Raw video frame (raster) with the keypoints drawn as VECTOR marks.

    Baking the dots into the bitmap made them blobs that blurred with the
    frame, so the crop rides at NATIVE resolution and the keypoints are
    scattered here in crop-local pixel coordinates.
    """
    img = np.asarray(data["img"])
    ax.imshow(img, interpolation=spec.get("interpolation", "nearest"))
    size = float(spec.get("kp_size", 1.4))
    # NB: "kp_uv_female".endswith("male") is True -- pair each key with its
    # spec key explicitly rather than testing the suffix.
    for key, spec_key, default in (("kp_uv_male", "kp_color", "#e74c3c"),
                                   ("kp_uv_female", "kp_color_fly1", "#3a7bff")):
        uv = data.get(key)
        if uv is None:
            continue
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        # An unprojectable keypoint is OMITTED, never drawn at (0, 0), which
        # would put a mark on the frame corner.
        m = np.isfinite(uv).all(axis=1)
        if not m.any():
            continue
        ax.scatter(uv[m, 0], uv[m, 1], s=size, c=spec.get(spec_key, default),
                   linewidths=0, zorder=3)
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)
    ax.set_axis_off()


def draw_image(ax, data, spec):
    ax.imshow(np.asarray(data["img"]),
              interpolation=spec.get("interpolation", "nearest"))
    ax.set_axis_off()


def draw_pitch_violin(ax, data, spec):
    if "exemplar_idx" in spec:
        idx = spec.get("exemplar_idx")
    else:
        raw = data.get("exemplar_idx")
        idx = None if raw is None else int(np.asarray(raw).ravel()[0])
        if idx is not None and idx < 0:      # -1 is the "no exemplar" sentinel
            idx = None
    panel_pitch_alignment_violin(ax, np.asarray(data["per_bout"]), exemplar_idx=idx)


def draw_zheight(ax, data, spec):
    panel_z_height_singing_vs_walking(
        ax, np.asarray(data["pulse_z"]), np.asarray(data["sine_z"]),
        np.asarray(data["walking_z"]), kind=spec.get("kind", "violin"),
        point_kwargs={"s": 3, "alpha": 0.25}, title=spec.get("title", ""))
    free_label = spec.get("free_label", FREE_LABEL)
    ax.set_xticks(ax.get_xticks())   # avoid FixedFormatter/FixedLocator warning
    ax.set_xticklabels([t.get_text().replace("free walk", free_label)
                        for t in ax.get_xticklabels()])


def draw_angle_density(ax, data, spec):
    rd = spec.get("range_deg", [0.0, 90.0])
    panel_joint_angle_density(
        ax, np.asarray(data["ext_pulse"]), np.asarray(data["ext_sine"]),
        range_deg=(float(rd[0]), float(rd[1])), title=spec.get("title", ""),
        legend_kwargs=_legend_kwargs(spec))


def draw_wing_polar(ax, data, spec):
    panel_wing_phase_polar(
        ax, np.asarray(data["phase_diffs"]),
        center_stat=spec.get("center_stat", "median"), title=spec.get("title", ""))


def draw_sine_inphase(ax, data, spec):
    t_ms = np.asarray(data["t_ms"], float)
    fr = _frame_range(spec)
    # A zoom inset's ABSOLUTE times carry nothing the reader needs -- the
    # dashed marker on panel C already says where the window came from -- and
    # read as an arbitrary broken range. Shift the whole trace, not the slice,
    # so the plotter's own frame_range still selects the same frames.
    if spec.get("rebase_time") and fr is not None and t_ms.size:
        lo = max(0, min(int(fr[0]), t_ms.size - 1))
        t_ms = t_ms - t_ms[lo]
    panel_sine_wing_inphase(
        ax, t_ms, np.asarray(data["ext_z"]), np.asarray(data["fold_z"]),
        fs=float(spec.get("fs", 800.0)), frame_range=fr,
        sine_segments=_segs_arg(data.get("sine_segments")),
        title=spec.get("title", ""), legend_kwargs=_legend_kwargs(spec))


def draw_pulse_class(ax, data, spec):
    counts = {}
    for t in _PULSE_TYPES:
        if f"count_{t}" in spec:
            counts[t] = int(spec[f"count_{t}"])
        else:
            pooled = data.get(f"pooled_{t}")
            counts[t] = int(np.asarray(pooled).shape[0]) if pooled is not None else 0
    results = {
        "centroids": {t: np.asarray(data[f"centroid_{t}"], dtype=float)
                      for t in _PULSE_TYPES if f"centroid_{t}" in data},
        "counts": counts,
        "pooled_waveforms": {t: np.asarray(data[f"pooled_{t}"], dtype=float)
                             for t in _PULSE_TYPES if f"pooled_{t}" in data},
        "fs": float(spec.get("fs", 800.0)),
    }
    panel_pulse_classification(
        ax, results, show_std=bool(spec.get("show_std", True)),
        title=spec.get("title", ""), legend_kwargs=_legend_kwargs(spec))


#: Panel type -> drawing function, keyed by the `type` in fig4.json.
DRAW = {
    "courtship.wing_z": draw_wing_z,
    "courtship.scutellum_z": draw_scutellum_z,
    "courtship.male_pitch": draw_male_pitch,
    "courtship.video_kp": draw_video_kp,
    "courtship.pitch_violin": draw_pitch_violin,
    "courtship.zheight": draw_zheight,
    "courtship.angle_density": draw_angle_density,
    "courtship.wing_polar": draw_wing_polar,
    "courtship.sine_inphase": draw_sine_inphase,
    "courtship.pulse_class": draw_pulse_class,
    "image": draw_image,
}

#: Panel types needing a non-rectilinear matplotlib projection.
PROJECTION = {"courtship.wing_polar": "polar"}
