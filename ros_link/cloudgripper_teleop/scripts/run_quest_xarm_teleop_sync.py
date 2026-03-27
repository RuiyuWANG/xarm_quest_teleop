#!/usr/bin/env python3
import os
import sys

def _discover_repo_root() -> str:
    candidates = []

    env_root = os.environ.get("CLOUDGRIPPER_REPO")
    if env_root:
        candidates.append(os.path.abspath(env_root))

    try:
        import rospkg  # type: ignore
        pkg_root = rospkg.RosPack().get_path("cloudgripper_teleop")
        candidates.append(os.path.abspath(os.path.join(pkg_root, "..", "..")))
    except Exception:
        pass

    candidates.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

    for root in candidates:
        if os.path.isfile(os.path.join(root, "src", "scripts", "run_quest_xarm_teleop_sync.py")):
            return root
    raise RuntimeError(
        "Could not locate CloudGripper repo root. "
        "Set CLOUDGRIPPER_REPO or install via symlink with ./install.sh."
    )


if __name__ == "__main__":
    try:
        repo_root = _discover_repo_root()
        if repo_root not in sys.path:
            sys.path.append(repo_root)
        from src.scripts.run_quest_xarm_teleop_sync import main
        main()
    except SystemExit as e:
        sys.exit(int(e.code))
    except Exception as e:
        print(f"Fatal: {e}")
        sys.exit(1)
