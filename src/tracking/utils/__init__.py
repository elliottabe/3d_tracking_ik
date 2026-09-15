"""Utilities for the package.

Importing this subpackage registers the custom OmegaConf resolvers (via
``path_utils``), so it's enough for ``main.py`` to import from here for configs
to compose correctly.
"""

from tracking.utils.path_utils import (
    convert_dict_to_path,
    convert_dict_to_string,
    convert_to_path,
    convert_to_string,
    register_custom_resolvers,
    save_config,
)

__all__ = [
    "register_custom_resolvers",
    "convert_to_string",
    "convert_to_path",
    "convert_dict_to_string",
    "convert_dict_to_path",
    "save_config",
]
