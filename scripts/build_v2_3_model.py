"""Build a derived, adhesion-free v2.3 MJCF for IK/reference-kinematics use.

Why this exists
----------------
The shared body-model repo (`fruitfly_body_models`, a SIBLING checkout of
this one) keeps its 8 ``<adhesion>`` actuators enabled --
``mjx.put_model(m, impl="warp")`` accepts them (warp is what training/rollout
uses), and removing them breaks checkpoints trained with adhesion. See
`fruitfly_body_models` commit 072a293.

The plain (non-warp) MJX path this port's IK uses cannot load ``mjTRN_BODY``
(adhesion) actuators at all. Rather than re-editing the shared model in
place, this script derives a kinematically-identical copy with the adhesion
actuators stripped, written to ``fruitfly_v2_3_ik/`` *beside* its source in
the shared repo and gitignored there -- so it sits next to the model it
tracks without ever displacing it. **The shared source XML is only ever
read, never written.**

Because it is gitignored rather than committed, a fresh ``fruitfly_body_models``
checkout has no ``fruitfly_v2_3_ik/`` and every ``anatomy=v2_3`` run fails at
``ParseXML`` about two minutes in. Run this script once per checkout (also
reported, non-fatally, by ``scripts/fetch_assets.py``'s ``v2_3_model`` row).

The two models are kinematically identical except for ``nu`` (272 vs 264)
and ``nsite`` (this port adds 50 -- see "New in this port" below): same
nq/nv/nbody/njnt/ngeom, identical joint/body/site name order, identical
jnt_type/jnt_range/body_pos/body_quat/site_pos/body_parentid, and FK from a
random qpos agrees exactly. This port's dataset stores qpos/qvel/xpos/xquat
-- none of which depend on ``nu`` or on the extra sites -- so output produced
with the derived model stays valid against checkpoints trained on the
272-actuator model.

This script also folds in what ``add_v2_3_tracking_sites.py`` used to do (a
separate script in the source repo, ported into this one): STAC creates its
own marker sites at runtime (`inverse_kinematics.anatomy.build_marker_model`),
so IK itself does not need the XML sites. `tracking.preprocess.scale` reads
them out of the XML directly, though (``_tracking_site_rest`` selects sites
via ``name.startswith("tracking[")`` to get the model's REST-pose keypoint
positions for the body-scale plausibility guard) -- with zero tracking sites
present it would raise rather than silently misbehave (`implied_body_length_mm`
raises `OrderMismatch` on a missing name), but that is a hard stop this
script exists to avoid, not a soft one worth relying on. (The shared source
XML happens to already carry these 50 sites -- stripping and reinserting them
here keeps their positions in sync with the anatomy config's
``KEYPOINT_INITIAL_OFFSETS`` rather than assuming the two never drift apart.)

New in this port -- ``aligned[<keypoint>]`` placeholder sites (VERIFIED FACT
3 in the task brief this script was written from does NOT mention these; see
`insert_aligned_sites`'s own docstring for the full reason). In short: this
repo's `inverse_kinematics.anatomy` module (absent from the source repo) uses
"a site named ``aligned[<name>]`` exists" as its definition of "the model has
keypoint `<name>`", and the upstream v2.3 model carries none -- unlike v1's
committed XML, which happens to already have 50 such placeholders as a
historical artifact. Without this step every `anatomy=v2_3` keypoint would
read as model-absent and `load_anatomy` (default `strict=True`, exactly how
`tracking.run` calls it) would refuse to load the anatomy at all.

Idempotent and safe to re-run whenever the upstream shared XML changes: reads
the shared XML fresh each time, strips adhesion actuators, strips any
pre-existing tracking[...] and aligned[...] sites, reinserts them, and
recompiles to verify the expected shape. Never writes to the source tree --
``--out-dir`` defaults beside the source, and every write happens under it or
under a caller-supplied override, both checked against ``--source`` before
anything is written (see ``main``).

Usage:
    # defaults already point at the sibling fruitfly_body_models checkout:
    python scripts/build_v2_3_model.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import mujoco as mj
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]

# Sibling checkout of Brunton-Lab/fruitfly_body_models, i.e.
# `${repo_root:}/../fruitfly_body_models` (configs/anatomy/v2_3.yaml's own
# `root:`, so the two never drift apart).
SHARED_BODY_MODELS = REPO_ROOT.parent / "fruitfly_body_models"
DEFAULT_SOURCE = SHARED_BODY_MODELS / "fruitfly_v2.3" / "fruitfly_muscles_warp.xml"
DEFAULT_OUT_DIR = SHARED_BODY_MODELS / "fruitfly_v2_3_ik"
DEFAULT_ANATOMY = REPO_ROOT / "configs" / "anatomy" / "v2_3.yaml"
OUT_XML_NAME = "fruitfly_v2_3_ik.xml"

ACTUATOR_SECTION_RE = re.compile(r"<actuator>.*?</actuator>", flags=re.S)
ADHESION_RE = re.compile(r"[ \t]*<adhesion\b[^>]*/>[ \t]*\n?")
SITE_RE = re.compile(r'[ \t]*<site name="tracking\[[^\]]*\]"[^>]*/>[ \t]*\n')
ALIGNED_SITE_RE = re.compile(r'[ \t]*<site name="aligned\[[^\]]*\]"[^>]*/>[ \t]*\n')
WORLDBODY_RE = re.compile(r"(<worldbody>[ \t]*\n)")

EXPECTED_NQ = 101
EXPECTED_NV = 100
EXPECTED_NBODY = 74
EXPECTED_NJNT = 95
EXPECTED_NU = 264
EXPECTED_N_TRACKING_SITES = 50


def refresh_symlink(link_path: Path, target_path: Path) -> None:
    """Create/replace a relative symlink at ``link_path`` pointing at ``target_path``.

    Uses os.path.relpath so the derived tree stays movable together. Replaces
    an existing symlink rather than failing; refuses to clobber a real file
    or directory that isn't already a symlink.
    """
    if link_path.is_symlink() or link_path.exists():
        if not link_path.is_symlink():
            raise SystemExit(
                f"ERROR: {link_path} exists and is not a symlink -- refusing to overwrite"
            )
        link_path.unlink()
    rel_target = os.path.relpath(target_path, start=link_path.parent)
    link_path.symlink_to(rel_target)


def strip_adhesion_actuators(xml: str) -> tuple[str, int]:
    """Remove every ``<adhesion .../>`` actuator inside the ``<actuator>`` section.

    Scoped to the actuator section only, so ``<default class="adhesion*">``
    blocks (which also contain bare ``<adhesion .../>`` elements setting
    defaults, not actuators) and the ``adhesion-collision`` geom class are
    left untouched.
    """
    m = ACTUATOR_SECTION_RE.search(xml)
    if not m:
        raise SystemExit("ERROR: <actuator> section not found in source XML")
    section = m.group(0)
    new_section, n_removed = ADHESION_RE.subn("", section)
    xml = xml[: m.start()] + new_section + xml[m.end() :]
    return xml, n_removed


def strip_existing_tracking_sites(xml: str) -> tuple[str, int]:
    """Remove any tracking[...] sites so re-running cannot duplicate them."""
    n = len(SITE_RE.findall(xml))
    return SITE_RE.sub("", xml), n


def strip_existing_aligned_sites(xml: str) -> tuple[str, int]:
    """Remove any aligned[...] sites so re-running cannot duplicate them."""
    n = len(ALIGNED_SITE_RE.findall(xml))
    return ALIGNED_SITE_RE.sub("", xml), n


def insert_aligned_sites(xml: str, kp_names: list) -> tuple[str, int]:
    """Insert one inert ``aligned[<name>]`` placeholder site per keypoint.

    This is NOT part of the source script this file ports (VERIFIED FACT 3):
    it exists only because this repo's `inverse_kinematics.anatomy.
    validate_anatomy`/`filter_anatomy` (a port-only addition, absent from the
    source repo entirely -- `grep` for `aligned\\[` there finds nothing)
    treats "the model has a site named ``aligned[<keypoint>]``" as ITS
    definition of "this model has that keypoint" (module docstring,
    "keypoint" category), independent of the `tracking[...]` sites STAC
    builds at runtime from `KEYPOINT_MODEL_PAIRS`.

    `model/fruitfly_v1/fruitfly_v1_free.xml` happens to already carry 50 such
    sites -- inert placeholders under `<worldbody>`, all at `pos="0 0 0"`,
    never read for their POSITION anywhere in this repo (only `anatomy.py`'s
    `in sites` name-set membership check reads them at all) -- so v1's own
    strict `filter_anatomy` passes by coincidence of its committed XML's
    history, not because anything downstream needs these sites to exist.
    `fruitfly_v2.3` (the shared, upstream model) carries none, so without
    this step every `anatomy=v2_3` keypoint would be reported "model-absent"
    and `load_anatomy(cfg)` (default `strict=True`, exactly how
    `tracking.run._load_anatomy` calls it) would raise `AnatomyMismatch` on
    every single keypoint before the run ever reached a stage that uses one.

    Mirrors v1's own placement (`<worldbody>`, `pos="0 0 0"`, `group="3"`)
    exactly, so both anatomies satisfy the SAME validation convention the
    same (inert) way -- this does not weaken the check for either model, it
    only gives v2_3 the placeholder v1 already had.
    """
    lines = "".join(
        f'    <site name="aligned[{kp}]" pos="0 0 0" group="3" rgba="0 1 0 1"/>\n'
        for kp in kp_names
    )
    m = WORLDBODY_RE.search(xml)
    if not m:
        raise SystemExit("ERROR: <worldbody> not found in XML")
    xml = xml[: m.end()] + lines + xml[m.end() :]
    return xml, len(kp_names)


def build_site_xml(kp_name: str, pos: str) -> str:
    return f'      <site name="tracking[{kp_name}]" pos="{pos}" size="0.01" group="3"/>\n'


def insert_tracking_sites(xml: str, pairs: dict, offsets: dict) -> tuple[str, int]:
    """Insert one <site> per keypoint as the first child of its parent body.

    Matches ``<body name="X" ...>`` and inserts immediately after that tag.
    """
    by_body: dict[str, list[str]] = {}
    for kp, body in pairs.items():
        by_body.setdefault(body, []).append(build_site_xml(kp, str(offsets[kp]).strip()))

    inserted = 0
    for body, site_lines in by_body.items():
        pat = re.compile(r'(<body name="' + re.escape(body) + r'"[^>]*>[ \t]*\n)')
        m = pat.search(xml)
        if not m:
            raise SystemExit(f'ERROR: body "{body}" not found in XML')
        xml = xml[: m.end()] + "".join(site_lines) + xml[m.end() :]
        inserted += len(site_lines)
    return xml, inserted


def verify_compiled_model(
    xml_path: Path, n_tracking_expected: int, n_aligned_expected: int
) -> mj.MjModel:
    m = mj.MjModel.from_xml_path(str(xml_path))

    shape = (m.nq, m.nv, m.nbody, m.njnt, m.nu)
    expected_shape = (EXPECTED_NQ, EXPECTED_NV, EXPECTED_NBODY, EXPECTED_NJNT, EXPECTED_NU)
    if shape != expected_shape:
        raise SystemExit(
            f"ERROR: compiled model shape {shape} != expected {expected_shape} "
            "(nq, nv, nbody, njnt, nu)"
        )

    n_body_trn = int((m.actuator_trntype == mj.mjtTrn.mjTRN_BODY).sum())
    if n_body_trn != 0:
        raise SystemExit(f"ERROR: {n_body_trn} mjTRN_BODY actuators remain (expected 0)")

    site_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
    n_tracking = sum(1 for n in site_names if n and n.startswith("tracking["))
    if n_tracking != n_tracking_expected:
        raise SystemExit(
            f"ERROR: {n_tracking} tracking[...] sites found, expected {n_tracking_expected}"
        )
    n_aligned = sum(1 for n in site_names if n and n.startswith("aligned["))
    if n_aligned != n_aligned_expected:
        raise SystemExit(
            f"ERROR: {n_aligned} aligned[...] sites found, expected {n_aligned_expected}"
        )

    return m


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Shared, adhesion-enabled source MJCF (never modified).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for the derived, adhesion-free MJCF.",
    )
    ap.add_argument(
        "--anatomy",
        type=Path,
        default=DEFAULT_ANATOMY,
        help="Anatomy config providing KEYPOINT_MODEL_PAIRS / KEYPOINT_INITIAL_OFFSETS.",
    )
    args = ap.parse_args()

    source = args.source.resolve()
    # Resolve both sides before relpath: on Hyak /gscratch is a symlink to
    # /mmfs1/gscratch, so resolving only `source` yields symlink targets that
    # climb out to / and back down a second, unresolved prefix -- broken links
    # that only surface later as "Error opening file 'assets/include/...'".
    out_dir = args.out_dir.resolve()

    # Never write to the shared source tree, even via a caller-supplied
    # --out-dir: the whole point of deriving a copy is that the checkout other
    # projects share is read-only from here. `refresh_symlink` independently
    # refuses to replace a real (non-symlink) `assets`/`floor.xml` -- true of
    # the source directory's own copies -- but that guard fires only for
    # those two specific paths, so the directory-identity check below is the
    # one thing standing between a bad `--out-dir` and `out_xml_path.write_text`
    # landing on the source XML itself.
    if out_dir == source.parent:
        raise SystemExit(f"ERROR: --out-dir must not be the source directory {source.parent}")
    if (out_dir / OUT_XML_NAME) == source:
        raise SystemExit(f"ERROR: --out-dir/{OUT_XML_NAME} would overwrite --source {source}")

    out_dir.mkdir(parents=True, exist_ok=True)

    source_dir = source.parent
    refresh_symlink(out_dir / "assets", source_dir / "assets")
    refresh_symlink(out_dir / "floor.xml", source_dir / "floor.xml")

    cfg = OmegaConf.load(args.anatomy)
    pairs = OmegaConf.to_container(cfg.model.KEYPOINT_MODEL_PAIRS)
    offsets = OmegaConf.to_container(cfg.model.KEYPOINT_INITIAL_OFFSETS)
    missing = [k for k in pairs if k not in offsets]
    if missing:
        raise SystemExit(f"ERROR: no KEYPOINT_INITIAL_OFFSETS for {missing}")

    kp_names = list(pairs.keys())

    xml = source.read_text()
    xml, n_removed_adhesion = strip_adhesion_actuators(xml)
    xml, n_removed_sites = strip_existing_tracking_sites(xml)
    xml, n_inserted_sites = insert_tracking_sites(xml, pairs, offsets)
    xml, n_removed_aligned = strip_existing_aligned_sites(xml)
    xml, n_inserted_aligned = insert_aligned_sites(xml, kp_names)

    out_xml_path = out_dir / OUT_XML_NAME
    out_xml_path.write_text(xml)

    print(f"removed {n_removed_adhesion} adhesion actuators")
    print(f"removed {n_removed_sites} pre-existing tracking sites; inserted {n_inserted_sites}")
    print(
        f"removed {n_removed_aligned} pre-existing aligned[] sites; inserted {n_inserted_aligned}"
    )

    m = verify_compiled_model(
        out_xml_path, n_tracking_expected=len(pairs), n_aligned_expected=len(kp_names)
    )
    print(
        f"compiled OK: nq={m.nq} nv={m.nv} nbody={m.nbody} njnt={m.njnt} nu={m.nu}, "
        f"0 mjTRN_BODY, {len(pairs)} tracking sites, {len(kp_names)} aligned[] sites"
    )
    print(f"wrote {out_xml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
