"""Verify every external asset the pipeline needs, BEFORE any stage starts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir

import tracking.utils.path_utils  # noqa: F401 -- registers ${repo_root:} etc. before compose

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
_REPO_ROOT = Path(__file__).resolve().parents[1]

_V2_3_MODEL_PATH = (
    _REPO_ROOT.parent / "fruitfly_body_models" / "fruitfly_v2_3_ik" / "fruitfly_v2_3_ik.xml"
)


def _asset(
    *,
    name: str,
    required: bool,
    path: Path | None,
    config_key: str,
    missing_hint: str,
    present_hint: str = "",
) -> dict[str, Any]:
    """One asset's report row: name, resolved path, and BOTH halves of the
    "what's wrong and how do I fix it" message -- named per the module
    docstring's own complaint about orbax's uninformative failure.
    """
    present = path is not None and path.exists()
    return {
        "name": name,
        "required": required,
        "status": "OK" if present else "MISSING",
        "present": present,
        "path": str(path) if path is not None else None,
        "config_key": config_key,
        "hint": present_hint if present else missing_hint,
    }


def _compose(ckpt_dir: str | None):
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        overrides = [f"paths.ckpt_dir={ckpt_dir}"] if ckpt_dir is not None else []
        return compose(config_name="pipeline", overrides=overrides)


def gather_assets(*, ckpt_dir: str | None) -> dict[str, Any]:
    cfg = _compose(ckpt_dir)

    mvq_path = Path(str(cfg.mvq.checkpoint))
    centerdetect_path = Path(str(cfg.centerdetect.checkpoint))

    assets = {
        "mvq_checkpoint": _asset(
            name="mvq_checkpoint",
            required=True,
            path=mvq_path,
            config_key="mvq.checkpoint",
            missing_hint=(
                f"MVQ checkpoint missing at {mvq_path} (mvq.checkpoint, "
                f"configs/mvq/v2.yaml). Fetch or train it, or point paths.ckpt_dir at "
                f"a machine that has it."
            ),
            present_hint="present (mvq.checkpoint, configs/mvq/v2.yaml)",
        ),
        "centerdetect_checkpoint": _asset(
            name="centerdetect_checkpoint",
            required=True,
            path=centerdetect_path,
            config_key="centerdetect.checkpoint",
            missing_hint=(
                f"CenterDetect checkpoint missing at {centerdetect_path} "
                f"(centerdetect.checkpoint, configs/centerdetect/default.yaml). "
                f"Fetch or train it, or point paths.ckpt_dir at a machine that has it."
            ),
            present_hint="present (centerdetect.checkpoint, configs/centerdetect/default.yaml)",
        ),
        "v2_3_model": _asset(
            name="v2_3_model",
            required=False,
            path=_V2_3_MODEL_PATH,
            config_key="(no config group yet -- Task 10's configs/anatomy/v2_3.yaml)",
            missing_hint=(
                f"MISSING (build it: scripts/build_v2_3_model.py) -- expected at "
                f"{_V2_3_MODEL_PATH}. Only needed for anatomy=v2_3 (Task 10); "
                f"anatomy=v1, this repo's default, does not use it."
            ),
            present_hint=(
                f"present at {_V2_3_MODEL_PATH}. Only needed for anatomy=v2_3; "
                f"rebuild with scripts/build_v2_3_model.py if the shared "
                f"fruitfly_body_models XML changed."
            ),
        ),
    }

    expected_hf_home = "/gscratch/portia/eabe/data/Johnson_lab/sam3"
    actual_hf_home = os.environ.get("HF_HOME")
    hf_home_ok = actual_hf_home == expected_hf_home
    assets["hf_home"] = {
        "name": "hf_home",
        "required": False,
        "status": "OK" if hf_home_ok else "MISSING",
        "present": hf_home_ok,
        "path": actual_hf_home,
        "config_key": "$HF_HOME env var",
        "hint": (
            f"HF_HOME={actual_hf_home!r}, expected {expected_hf_home!r} (SAM3/DINOv3 "
            f"weights were deliberately moved off ~/.cache -- MEMORY 'SAM3 weights "
            f"location'). export HF_HOME={expected_hf_home}"
            if not hf_home_ok
            else f"HF_HOME correctly points at {expected_hf_home}"
        ),
    }

    token_present = bool(os.environ.get("HF_TOKEN"))
    assets["hf_token"] = {
        "name": "hf_token",
        "required": False,
        "status": "OK" if token_present else "MISSING",
        "present": token_present,
        "path": None,
        "config_key": "$HF_TOKEN env var",
        "hint": (
            "HF_TOKEN is set"
            if token_present
            else "HF_TOKEN is unset -- only required for a gated HF model or a cold cache"
        ),
    }

    return assets


def _print_report(assets: dict[str, Any]) -> None:
    for asset in assets.values():
        flag = "OK     " if asset["status"] == "OK" else "MISSING"
        where = asset["path"] if asset["path"] is not None else "(env var, no path)"
        req = "required" if asset["required"] else "optional"
        print(f"[{flag}] {asset['name']:<24} ({req}, {asset['config_key']})  {where}")
        if asset["status"] != "OK":
            print(f"          {asset['hint']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt-dir",
        default=None,
        help="Override paths.ckpt_dir (for testing against an empty/synthetic root).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print a machine-readable report instead of text."
    )
    args = parser.parse_args(argv)

    assets = gather_assets(ckpt_dir=args.ckpt_dir)
    ok = all(a["status"] == "OK" for a in assets.values() if a["required"])

    if args.json:
        print(json.dumps({"ok": ok, "assets": assets}, indent=2))
    else:
        _print_report(assets)
        print("all required assets present" if ok else "MISSING required asset(s) -- see above")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
