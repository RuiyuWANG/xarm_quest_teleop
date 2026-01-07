import os, json, time
import numpy as np
import cv2
from dataclasses import dataclass

import rospy
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge

_bridge = CvBridge()

def now_str():
    return time.strftime("%Y-%m-%d_%H-%M-%S")

def img_to_bgr(msg):
    return _bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

def depth_to_m(msg):
    d = _bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    if msg.encoding in ["16UC1", "mono16"]:
        return d.astype(np.float32) * 0.001
    return d.astype(np.float32)

@dataclass
class EpisodeWriter:
    root: str
    episode_dir: str = None
    f: any = None
    step: int = 0

    def start(self):
        os.makedirs(self.root, exist_ok=True)
        self.episode_dir = os.path.join(self.root, f"{now_str()}_ep")
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(os.path.join(self.episode_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self.episode_dir, "depth"), exist_ok=True)
        self.f = open(os.path.join(self.episode_dir, "steps.jsonl"), "w")
        self.step = 0
        return self.episode_dir

    def stop(self):
        if self.f:
            self.f.close()
        self.f = None
        self.episode_dir = None

class DataCollector:
    def __init__(self, data_root: str, camera_rig, quest_client, control_source: str = "right"):
        self.data_root = data_root
        self.cameras = camera_rig
        self.quest = quest_client
        self.control_source = control_source
        self.writer = EpisodeWriter(root=data_root)

        self.joint_state = None
        rospy.Subscriber("/joint_states", JointState, self._on_joints, queue_size=10)

        self.recording = False

    def _on_joints(self, msg: JointState):
        self.joint_state = msg

    def start_episode(self, meta: dict):
        ep_dir = self.writer.start()
        with open(os.path.join(ep_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self.recording = True
        rospy.loginfo(f"[DataCollector] recording -> {ep_dir}")

    def stop_episode(self):
        self.recording = False
        self.writer.stop()
        rospy.loginfo("[DataCollector] stopped")

    def write_step(self, action: dict):
        if not self.recording or self.writer.f is None:
            return

        t = rospy.Time.now().to_sec()
        step = self.writer.step
        ep_dir = self.writer.episode_dir

        cams_out = {}
        for name, st in self.cameras.cams.items():
            cams_out[name] = {"rgb": None, "depth": None, "points_stamp": None}

            if st.rgb is not None:
                rgb = img_to_bgr(st.rgb)
                rgb_path = os.path.join(ep_dir, "rgb", f"{name}_{step:06d}.png")
                cv2.imwrite(rgb_path, rgb)
                cams_out[name]["rgb"] = os.path.relpath(rgb_path, ep_dir)

            if st.depth is not None:
                depth = depth_to_m(st.depth)
                depth_path = os.path.join(ep_dir, "depth", f"{name}_{step:06d}.npy")
                np.save(depth_path, depth)
                cams_out[name]["depth"] = os.path.relpath(depth_path, ep_dir)

            if st.points is not None:
                cams_out[name]["points_stamp"] = float(st.points.header.stamp.to_sec())

        # choose hand state
        hand = self.quest.right if self.control_source == "right" else self.quest.left

        rec = {
            "t": t,
            "step": step,
            "cameras": cams_out,
            "quest": {
                "pose": None if hand.pose is None else {
                    "frame": hand.pose.header.frame_id,
                    "p": [hand.pose.pose.position.x, hand.pose.pose.position.y, hand.pose.pose.position.z],
                    "q": [hand.pose.pose.orientation.x, hand.pose.pose.orientation.y,
                          hand.pose.pose.orientation.z, hand.pose.pose.orientation.w],
                },
                "twist": None if hand.twist is None else {
                    "lin": [hand.twist.linear.x, hand.twist.linear.y, hand.twist.linear.z],
                    "ang": [hand.twist.angular.x, hand.twist.angular.y, hand.twist.angular.z],
                },
                "inputs": None if hand.inputs is None else {
                    "button_upper": bool(hand.inputs.button_upper),
                    "button_lower": bool(hand.inputs.button_lower),
                    "thumb_h": float(hand.inputs.thumb_stick_horizontal),
                    "thumb_v": float(hand.inputs.thumb_stick_vertical),
                    "press_index": float(hand.inputs.press_index),
                    "press_middle": float(hand.inputs.press_middle),
                }
            },
            "robot": None if self.joint_state is None else {
                "names": list(self.joint_state.name),
                "pos": list(self.joint_state.position),
                "vel": list(self.joint_state.velocity) if self.joint_state.velocity else None,
            },
            "action": action,
        }

        self.writer.f.write(json.dumps(rec) + "\n")
        self.writer.f.flush()
        self.writer.step += 1
