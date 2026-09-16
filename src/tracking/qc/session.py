"""One session's per-bout QC gathered into a scorecard."""

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
    """`note` alone when nothing else is wrong, else both, worst-first."""
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
