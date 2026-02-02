# src/utils/config_utils.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DatasetInfo:
    # ---- dataset section ----
    root_dir: str
    task: str
    robot: str
    gripper: str
    description: str
    num_demos: int
    collection_freq_hz: float

    demo_id_start: int = 0
    demo_prefix: str = "episode_"
    random_start: bool = False
    operator: str = ""
    notes: str = ""

    # ---- calibration section (raw dict, camera-keyed) ----
    calibration: Dict[str, Any] = None

    @property
    def task_dir(self) -> str:
        # convention: <root>/<task>/
        return os.path.join(self.root_dir, self.task)

    def as_dict(self) -> Dict[str, Any]:
        # serialize in the same shape as your JSON
        return {
            "dataset": {
                "root_dir": self.root_dir,
                "task": self.task,
                "robot": self.robot,
                "gripper": self.gripper,
                "description": self.description,
                "num_demos": int(self.num_demos),
                "demo_id_start": int(self.demo_id_start),
                "demo_prefix": self.demo_prefix,
                "random_start": bool(self.random_start),
                "operator": self.operator,
                "notes": self.notes,
                "collection_freq_hz": float(self.collection_freq_hz),
            },
            "calibration": self.calibration if self.calibration is not None else {},
        }


def load_dataset_json(path: str) -> DatasetInfo:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "dataset" not in cfg:
        raise ValueError("dataset config must contain top-level key 'dataset'")
    if "calibration" not in cfg:
        cfg["calibration"] = {}

    ds = cfg["dataset"]
    cal = cfg["calibration"]

    # Required fields
    required = ["root_dir", "task", "robot", "gripper", "description", "num_demos"]
    missing = [k for k in required if k not in ds]
    if missing:
        raise ValueError(f"missing required dataset fields: {missing}")

    # Resolve root_dir relative to the config file location (handy)
    root_dir = str(ds["root_dir"])
    if not os.path.isabs(root_dir):
        root_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), root_dir))

    return DatasetInfo(
        root_dir=root_dir,
        task=str(ds["task"]),
        robot=str(ds["robot"]),
        gripper=str(ds["gripper"]),
        description=str(ds["description"]),
        num_demos=int(ds["num_demos"]),
        demo_id_start=int(ds.get("demo_id_start", 0)),
        demo_prefix=str(ds.get("demo_prefix", "episode_")),
        random_start=bool(ds.get("random_start", False)),
        operator=str(ds.get("operator", "")),
        notes=str(ds.get("notes", "")),
        collection_freq_hz=float(ds.get("collection_freq_hz", None)),
        calibration=cal,
    )


def next_demo_id(parent_dir: str, prefix: str) -> int:
    """
    Finds next numeric id for episodes under parent_dir with names like f"{prefix}{id:04d}"
    """
    if not os.path.isdir(parent_dir):
        return 0
    best = -1
    for name in os.listdir(parent_dir):
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        try:
            idx = int(suffix)
            best = max(best, idx)
        except Exception:
            continue
    return best + 1
