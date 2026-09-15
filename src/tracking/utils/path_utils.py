"""Path and config helpers for the Hydra-based project.

Importing this module **registers the project's custom OmegaConf resolvers**
(notably ``multirun_save_dir``), so configs that interpolate them resolve
correctly. The package's ``utils/__init__.py`` imports it for that side effect,
and ``main.py`` imports the package — so the resolvers are always registered
before Hydra composes a config.
"""

import logging
from pathlib import Path

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def _detect_repo_root() -> Path:
    """Return the repo root: the nearest parent containing ``pyproject.toml``.

    Resolved from this file's location, so it's independent of the launch
    directory and of Hydra (which is important — ``hydra.sweep.dir`` is resolved
    before the ``HydraConfig`` singleton is set, so ``${hydra:runtime.cwd}`` is
    not available there). Falls back to ``<file>/../../..`` (``src/<pkg>/utils``
    → repo root) if no marker is found.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


_REPO_ROOT = _detect_repo_root()


def register_custom_resolvers() -> None:
    """Register the custom OmegaConf resolvers used in the configs.

    Safe to call repeatedly (``replace=True``).
    """

    def multirun_aware_save_dir(base_dir: str, run_id: str) -> str:
        """Resolve a run's save dir, aware of Hydra ``--multirun`` sweeps.

        - **Single run:** ``<base_dir>/<run_id>``.
        - **Multirun:** Hydra's per-job output dir (the sweep subdir), falling
          back to ``<base_dir>/<run_id>/<override_dirname>`` if unavailable.
        """
        try:
            from hydra.core.hydra_config import HydraConfig

            if HydraConfig.initialized():
                hydra_cfg = HydraConfig.get()
                if hydra_cfg.mode.name == "MULTIRUN":
                    output_dir = hydra_cfg.runtime.output_dir
                    if output_dir:
                        return str(output_dir)
                    override_dirname = hydra_cfg.job.override_dirname
                    if override_dirname:
                        return f"{base_dir}/{run_id}/{override_dirname}"
            return f"{base_dir}/{run_id}"
        except Exception as exc:  # defensive: never let path resolution crash a run
            logger.debug("multirun_save_dir fell back to single-run path: %s", exc)
            return f"{base_dir}/{run_id}"

    # Repo root, independent of cwd / HydraConfig. Use as `${repo_root:}`.
    OmegaConf.register_new_resolver(
        "repo_root", lambda *_: str(_REPO_ROOT), use_cache=True, replace=True
    )
    OmegaConf.register_new_resolver(
        "multirun_save_dir", multirun_aware_save_dir, use_cache=False, replace=True
    )
    OmegaConf.register_new_resolver(
        "eq", lambda x, y: str(x).lower() == str(y).lower(), use_cache=False, replace=True
    )
    OmegaConf.register_new_resolver("divide", lambda x, y: x // y, use_cache=False, replace=True)
    OmegaConf.register_new_resolver(
        "contains", lambda x, y: str(x).lower() in str(y).lower(), use_cache=False, replace=True
    )
    OmegaConf.register_new_resolver(
        "resolve_default",
        lambda default, arg: default if arg == "" else arg,
        use_cache=False,
        replace=True,
    )


# Register on import so any config that uses these resolvers composes correctly.
register_custom_resolvers()


def convert_to_string(value):
    """Return ``value`` as a string (handles ``Path`` objects)."""
    return str(value) if isinstance(value, Path) else value


def convert_to_path(value):
    """Return ``value`` as a ``Path`` (handles strings)."""
    return Path(value) if isinstance(value, str) else value


def convert_dict_to_string(d: dict) -> dict:
    """Return a copy of ``d`` with all values stringified."""
    return {k: convert_to_string(v) for k, v in d.items()}


def convert_dict_to_path(d: dict) -> dict:
    """Convert a resolved ``paths`` dict to ``Path`` objects, creating each dir.

    The ``user`` key is left as-is; every other entry is treated as a directory
    and created (``mkdir(parents=True, exist_ok=True)``). Pass a *resolved* plain
    dict, e.g. ``OmegaConf.to_container(cfg.paths, resolve=True)``.
    """
    out = {}
    for key, value in d.items():
        if key == "user" or value is None:
            out[key] = value
            continue
        path = convert_to_path(value)
        path.mkdir(parents=True, exist_ok=True)
        out[key] = path
    return out


def save_config(cfg, path) -> None:
    """Save a fully-resolved copy of ``cfg`` to ``path`` as YAML.

    Resolves the ``paths`` section (so interpolations like ``${multirun_save_dir:...}``
    become concrete) before writing, giving a self-contained record of the run.
    """
    cfg_copy = OmegaConf.create(cfg)
    if "paths" in cfg_copy:
        OmegaConf.resolve(cfg_copy.paths)
        cfg_copy.paths = convert_dict_to_string(
            OmegaConf.to_container(cfg_copy.paths, resolve=True)
        )
    OmegaConf.save(cfg_copy, path)
