"""Check local assets/dependencies; optionally forbid all reads of an old checkout."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forbid", type=Path)
    args = parser.parse_args()
    if args.forbid:
        forbidden = args.forbid.resolve()

        def audit(event, values):
            if event == "open" and values and isinstance(values[0], (str, bytes)):
                path = Path(
                    values[0].decode() if isinstance(values[0], bytes) else values[0]
                ).absolute()
                if path.is_relative_to(forbidden):
                    raise RuntimeError(f"Attempt to read forbidden checkout: {path}")

        sys.addaudithook(audit)
    from supermimic.kinematics import G1Kinematics
    from supermimic.paths import URDF_PATH, MJCF_PATH, USD_PATH
    from supermimic.train import main as train_main
    from supermimic.evaluate import main as evaluate_main
    from supermimic.tasks.config import TrackingEnvCfg
    from supermimic.tasks.environment import TrackingEnv
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils import validate
    from pxr import Sdf, UsdUtils

    assert callable(train_main) and callable(evaluate_main)
    assert issubclass(TrackingEnv, ManagerBasedRLEnv)
    validate(TrackingEnvCfg())
    assert len(G1Kinematics("cpu").q_lo) == 29
    for name in (URDF_PATH, MJCF_PATH, USD_PATH):
        path = Path(name).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(), path
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(
        Sdf.AssetPath(USD_PATH)
    )
    assert not unresolved, unresolved
    for path in [layer.realPath for layer in layers] + list(assets):
        if path:
            assert Path(path).resolve().is_relative_to(ROOT), path
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("supermimic", "isaaclab")) and getattr(
            module, "__file__", None
        ):
            assert Path(module.__file__).resolve().is_relative_to(ROOT), (
                name,
                module.__file__,
            )
    # Python environments may legitimately link their interpreter outside the
    # checkout. Inspect project/dependency sources and assets, not local caches
    # or environment internals.
    excluded = {".git", ".venv", "venv", ".cache", ".ruff_cache", ".pytest_cache", "__pycache__"}
    for directory, children, files in os.walk(ROOT):
        children[:] = [name for name in children if name not in excluded]
        for name in children + files:
            path = Path(directory) / name
            if path.is_symlink():
                assert path.resolve().is_relative_to(ROOT), path
    print(
        "PASS: local package, pinned Isaac Lab, MJCF/FK, USD dependency closure and symlinks"
    )


if __name__ == "__main__":
    main()
