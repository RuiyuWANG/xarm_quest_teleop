from __future__ import annotations

from dataclasses import dataclass
import os
from typing import List
import yaml


@dataclass
class DatasetInfo:
    root_dir: str
    name: str
    task: str
    operator: str
    scene: str = ""
    notes: str = ""

    demo_prefix: str = "demo_"
    demo_id_start: int = 0
    num_demos: int = 1

    random_start_enabled: bool = True
    pos_mm_xyz: List[float] = None
    rot_rad_rpy: List[float] = None


def load_dataset_yaml(path: str) -> DatasetInfo:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    d = y["dataset"]
    n = y.get("naming", {})
    c = y.get("collection", {})
    rs = y.get("random_start", {})

    return DatasetInfo(
        root_dir=str(d["root_dir"]),
        name=str(d.get("name", "dataset")),
        task=str(d.get("task", "task")),
        operator=str(d.get("operator", "operator")),
        scene=str(d.get("scene", "")),
        notes=str(d.get("notes", "")),

        demo_prefix=str(n.get("demo_prefix", "demo_")),
        demo_id_start=int(c.get("demo_id_start", 0)),
        num_demos=int(c.get("num_demos", 1)),

        random_start_enabled=bool(rs.get("enabled", True)),
        pos_mm_xyz=list(rs.get("pos_mm_xyz", [10.0, 10.0, 5.0])),
        rot_rad_rpy=list(rs.get("rot_rad_rpy", [0.10, 0.10, 0.10])),
    )


def next_demo_id(root_dir: str, prefix: str) -> int:
    if not os.path.isdir(root_dir):
        return 0
    best = -1
    for nm in os.listdir(root_dir):
        if not nm.startswith(prefix):
            continue
        tail = nm[len(prefix):]
        try:
            best = max(best, int(tail))
        except ValueError:
            pass
    return best + 1
