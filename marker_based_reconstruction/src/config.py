"""Runtime configuration loaded once from YAML into a nested SimpleNamespace.

The source is cfg/marker_based_reconstruction.yaml, or the path in the TREE_CFG
environment variable. Usage:
    import marker_based_reconstruction.src.config as config
    config.CFG.pose_graph.icp_voxel
"""

import os
from types import SimpleNamespace

import yaml

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_DIR = os.path.join(PACKAGE_DIR, "cfg")
DEFAULT_CFG = os.path.join(CFG_DIR, "marker_based_reconstruction.yaml")


def _namespacify(obj):
    """Recursively convert dicts to SimpleNamespace for attribute (dot) access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _namespacify(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_namespacify(v) for v in obj]
    return obj


def load_config(path=None):
    """Load a YAML config: explicit `path`, else TREE_CFG env var, else default."""
    path = path or os.environ.get("TREE_CFG", DEFAULT_CFG)
    with open(path) as f:
        return _namespacify(yaml.safe_load(f))


CFG = load_config()
