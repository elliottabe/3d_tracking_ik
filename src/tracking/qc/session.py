"""One session's per-bout QC gathered into a scorecard.

Two files with the same numbers: `session_qc.json` for machines,
`session_qc.md` for reading. A scorecard nobody reads is what lets 42/160
NaN-cost pairs into a published figure.

A missing or unparseable per-bout `qc.json` becomes a ROW with a `status`, never
an exception and never an absence. Dropping it makes a part-processed session
indistinguishable from a complete one, and the reader counts rows.

Sex is read from each bout's `sex.json`, never assumed from the fly index.
`fly0 = female` is a convention this pipeline enforces; a future swap must
relabel the column rather than silently mislabel it.

Aggregates are split by sex. A pooled posture number hid a 46/160 female versus
0/160 male split.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

__all__ = ["COLUMNS", "SessionRow", "collect_rows", "render_markdown", "write_scorecard"]


@dataclass(frozen=True)
class SessionRow:
    bout: str
    fly: int
    sex: str
    status: str
    n_frames: int | None
    missing_fraction: float | None
    max_gap: int | None
    worst_keypoint: str | None
    fitted_px: float | None
    measured_px: float | None
    reproj_ratio: float | None
    worst_cv_invariant: str | None
    n_collapsed: int | None
    eye_cv: float | None
    pitch_p95_deg: float | None
    n_pitched: int | None
    n_flagged: int | None


COLUMNS = tuple(f.name for f in fields(SessionRow))


def _sex_for(bout_dir: Path, fly: int) -> str:
    from tracking.qc.posture import fly_sex_label

    path = bout_dir / "sex.json"
    if not path.exists():
        return "unknown"
    try:
        return fly_sex_label(json.loads(path.read_text()), fly)
    except (json.JSONDecodeError, ValueError, TypeError):
        return "unknown"


def _blank_row(bout: str, fly: int, sex: str, status: str) -> SessionRow:
    return SessionRow(
        bout=bout,
        fly=fly,
        sex=sex,
        status=status,
        **{
            f.name: None
            for f in fields(SessionRow)
            if f.name not in ("bout", "fly", "sex", "status")
        },
    )


def _row_from_qc(bout: str, fly: int, sex: str, qc: dict, status: str) -> SessionRow:
    rep = qc.get("reproj", {}) or {}
    inv = qc.get("invariants", {}) or {}
    cov = qc.get("coverage", {}) or {}
    pos = qc.get("posture", {}) or {}
    eye = (inv.get("invariants", {}) or {}).get("EyeL--EyeR", {}) or {}
    return SessionRow(
        bout=bout,
        fly=fly,
        # `sex` (from `sex.json`, read fresh by `_sex_for`) is authoritative: a
        # canonicalisation swap rewrites `sex.json` but never a bout's already-written
        # `qc.json`, so the baked-in `posture.sex` can go stale. It is a fallback only
        # for the rare case the fresh read itself came back "unknown".
        sex=sex if sex != "unknown" else (pos.get("sex") or "unknown"),
        status=status,
        n_frames=pos.get("n_frames") or cov.get("n_frames"),
        missing_fraction=cov.get("missing_fraction"),
        max_gap=(cov.get("gap_runs", {}) or {}).get("max"),
        worst_keypoint=cov.get("worst_keypoint"),
        fitted_px=rep.get("fitted_px"),
        measured_px=rep.get("measured_px"),
        reproj_ratio=rep.get("ratio"),
        worst_cv_invariant=inv.get("worst_cv_invariant"),
        n_collapsed=inv.get("n_collapsed"),
        eye_cv=eye.get("cv"),
        pitch_p95_deg=(pos.get("pitch_deg", {}) or {}).get("p95"),
        n_pitched=pos.get("n_pitched"),
        n_flagged=pos.get("n_flagged"),
    )


# A bout-fly whose keypoints are mostly absent still SOLVES -- the IK fits
# whatever frames exist -- and the fit is then meaningless while looking
# ordinary. Measured on Session0: the female is 87.7% missing in bout 19 and
# 92.1% in bout 26 (max_gap 395 of 395, and 718 of 748), yielding 15.85 px and
# 12.96 px fits, against 8.7% missing and 2.91 px in bout 28. Every one of those
# rows was labelled "ok" until 2026-09-14.
#
# 0.5 separates those populations with room to spare in both directions. It is a
# LABEL, never a gate: the data is kept and reported, because a partly-tracked
# fly is still the analyst's to judge -- and discarding it would also be the
# wrong call for the case this exists to serve, where only one fly is tracked
# and the other's absence must be visible rather than silently dropped.
SPARSE_MISSING_FRACTION = 0.5


def collect_rows(run_root, *, excluded=()) -> list[SessionRow]:
    """One row per `bouts/bout_*/fly*/` directory found, in bout then fly order."""
    run_root = Path(run_root)
    drop = {(str(e["key"]), int(e["fly"])): str(e["reason"]) for e in (excluded or ())}
    rows: list[SessionRow] = []
    for fly_dir in sorted(run_root.glob("bouts/bout_*/fly*")):
        if not fly_dir.is_dir():
            continue
        bout = fly_dir.parent.name
        fly = int(fly_dir.name.removeprefix("fly"))
        sex = _sex_for(fly_dir.parent, fly)
        status = "ok"
        if (bout, fly) in drop:
            status = f"excluded:{drop[(bout, fly)]}"
        qc_path = fly_dir / "qc.json"
        if not qc_path.exists():
            # NOT "ok;qc.json missing", which asserted ok and missing at once.
            # This fly produced no IK solve; the OTHER fly in the same bout is
            # unaffected and keeps its full row, so a male's song stays usable
            # when the female was never tracked.
            reason = "not_fit:no qc.json (ik produced no solve)"
            rows.append(_blank_row(bout, fly, sex, _join_status(status, reason)))
            continue
        try:
            qc = json.loads(qc_path.read_text())
        except json.JSONDecodeError as exc:
            reason = f"not_fit:qc.json unreadable ({exc.msg})"
            rows.append(_blank_row(bout, fly, sex, _join_status(status, reason)))
            continue
        row = _row_from_qc(bout, fly, sex, qc, status)
        missing = row.missing_fraction
        if missing is not None and float(missing) >= SPARSE_MISSING_FRACTION:
            row = replace(
                row,
                status=_join_status(status, f"sparse:{float(missing) * 100:.0f}% missing"),
            )
        rows.append(row)
    return rows


def _join_status(base: str, note: str) -> str:
    """`note` alone when nothing else is wrong, else both, worst-first.

    `base` is "ok" unless the caller excluded this bout-fly. Emitting
    "ok;<problem>" reads as a contradiction and is what this replaces.
    """
    return note if base == "ok" else f"{base};{note}"


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_markdown(rows: list[SessionRow]) -> str:
    """A pipe table with one row per bout-fly, columns in `COLUMNS` order."""
    head = "| " + " | ".join(COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    body = ["| " + " | ".join(_fmt(getattr(r, c)) for c in COLUMNS) + " |" for r in rows]
    return "\n".join([head, rule, *body]) + "\n"


def write_scorecard(run_root, rows, *, json_path, md_path) -> dict:
    """Write both files and return the aggregate report, split by sex."""
    # `n_rows` counts bout-flies LOOKED AT; `n_scored` counts those with a
    # number. They differ whenever a fly was not fit, and quoting a median
    # beside a row count it was not computed from is how a cohort of 3
    # measurements gets read as 4.
    #
    # The median is over `ok` rows ONLY. A row 90% missing still carries a
    # ratio, and folding it in drags the cohort statistic toward a fit nobody
    # would use -- while `n_sparse` keeps it visible rather than hidden.
    by_sex: dict[str, dict] = {}
    for row in rows:
        bucket = by_sex.setdefault(
            row.sex,
            {
                "n_rows": 0,
                "n_scored": 0,
                "n_sparse": 0,
                "n_not_fit": 0,
                "n_flagged": 0,
                "ratios": [],
            },
        )
        bucket["n_rows"] += 1
        bucket["n_flagged"] += int(row.n_flagged or 0)
        status = str(row.status)
        if "not_fit" in status:
            bucket["n_not_fit"] += 1
        if "sparse" in status:
            bucket["n_sparse"] += 1
        if row.reproj_ratio is not None:
            bucket["n_scored"] += 1
            if status == "ok":
                bucket["ratios"].append(float(row.reproj_ratio))
    for bucket in by_sex.values():
        ratios = bucket.pop("ratios")
        bucket["median_reproj_ratio"] = float(sorted(ratios)[len(ratios) // 2]) if ratios else None
        bucket["n_median_over"] = len(ratios)

    report = {
        "run_root": str(run_root),
        "n_rows": len(rows),
        "by_sex": by_sex,
        "rows": [asdict(r) for r in rows],
    }
    Path(json_path).write_text(json.dumps(report, indent=2))
    Path(md_path).write_text(render_markdown(rows))
    return report
