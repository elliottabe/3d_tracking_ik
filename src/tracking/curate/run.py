"""`python -m tracking.curate` -- the curation Hydra entry point."""

from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig, OmegaConf

import tracking.utils.path_utils  # noqa: F401  registers ${repo_root:}
from tracking.curate.masks.adapter import import_masks
from tracking.curate.merge import TierSpec, merge_tiers
from tracking.curate.package import package_root
from tracking.curate.stages import ordered
from tracking.curate.validate import format_findings, validate_root

__all__ = ["main"]


def _tiers(cfg) -> list[TierSpec]:
    return [
        TierSpec(
            path=t["path"],
            source_id=t["source_id"],
            kind=t.get("kind", "human"),
            weight=float(t.get("weight", 1.0)),
            checkpoint=t.get("checkpoint"),
            gates=dict(t.get("gates") or {}),
            review=dict(t.get("review") or {}),
        )
        for t in OmegaConf.to_container(cfg.curate.tiers, resolve=True)
    ]


def _run_stage(name, cfg) -> int:
    c = cfg.curate
    if name == "merge":
        manifest = merge_tiers(_tiers(cfg), c.out_root)
        print(f"[curate] merged {len(manifest['sources'])} source(s) into {c.out_root}")
    elif name == "import_masks":
        if not c.sam_dir:
            print("[curate] import_masks: curate.sam_dir is null, skipping")
            return 0
        print(f"[curate] {import_masks(c.sam_dir, c.masks_root, tool=c.mask_tool)}")
    elif name == "validate":
        findings = validate_root(c.out_root, masks_root=c.masks_root)
        print(format_findings(findings))
        return 1 if any(f.level == "error" for f in findings) else 0
    elif name == "package":
        result = package_root(
            c.out_root, c.package_zip, tier=c.package_tier, masks_root=c.masks_root
        )
        print(f"[curate] {result}")
    return 0


@hydra.main(version_base=None, config_path="../../../configs", config_name="curate")
def main(cfg: DictConfig) -> int:
    names = ordered(cfg.stages)
    if cfg.dry_run:
        print(f"[curate] stages (dependency order): {names}")
        print(f"[curate] out_root = {cfg.curate.out_root}")
        return 0
    for name in names:
        print(f"[curate] === {name} ===")
        rc = _run_stage(name, cfg)
        if rc:
            raise SystemExit(rc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
