# tests/test_data_collector.py
import os, json
import numpy as np
import types

def test_collector_writes_episode(tmp_path, fake_rospy, monkeypatch):
    # Import after rospy stub
    from vr_pipeline.data.collector import DataCollector

    # Fake camera rig shaped like your collector expects.
    # If your collector iterates self.cameras.cams.items() with .rgb/.depth, mimic that.
    class FakeCamState:
        def __init__(self):
            # minimal Image-like objects for rgb/depth
            rgb = (np.random.rand(10, 12, 3) * 255).astype(np.uint8)
            depth = (np.random.rand(10, 12) * 1000).astype(np.uint16)
            self.rgb = types.SimpleNamespace(height=10, width=12, data=rgb.tobytes(), encoding="bgr8")
            self.depth = types.SimpleNamespace(height=10, width=12, data=depth.tobytes(), encoding="16UC1")
            self.points = None

    class FakeCameraRig:
        def __init__(self):
            self.cams = {"hand": FakeCamState()}

    # Fake quest client shaped like collector expects (quest.right.pose/twist/inputs)
    class FakeInputs:
        button_upper = True
        button_lower = False
        thumb_stick_horizontal = 0.0
        thumb_stick_vertical = 0.0
        press_index = 0.2
        press_middle = 0.0

    class FakeHand:
        pose = None
        twist = None
        inputs = FakeInputs()

    class FakeQuest:
        right = FakeHand()
        left = FakeHand()

    cams = FakeCameraRig()
    quest = FakeQuest()

    # Patch cv_bridge conversions used by your collector utils (if any).
    # If your collector uses cv_bridge, it may not exist in pure unit env.
    # Easiest: monkeypatch conversion functions in your collector module.
    import vr_pipeline.data.collector as collector_mod
    collector_mod.img_to_bgr = lambda msg: np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
    collector_mod.depth_to_m = lambda msg: np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).astype(np.float32) * 0.001

    c = DataCollector(str(tmp_path), cams, quest, control_source="right")

    meta = {"test": True}
    c.start_episode(meta)
    c.write_step(action={"mode": "test", "gripper_pulse": 123})
    c.stop_episode()

    # Verify episode directory contents
    # Find created episode dir
    eps = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(eps) == 1
    ep = eps[0]

    assert (ep / "meta.json").exists()
    assert (ep / "steps.jsonl").exists()

    with open(ep / "meta.json", "r") as f:
        m = json.load(f)
    assert m["test"] is True

    with open(ep / "steps.jsonl", "r") as f:
        lines = f.readlines()
    assert len(lines) >= 1
    rec = json.loads(lines[0])
    assert "action" in rec
