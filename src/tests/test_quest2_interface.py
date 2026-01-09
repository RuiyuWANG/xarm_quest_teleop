#!/usr/bin/env python3
import os
import sys
import time
import signal
import subprocess
from typing import List, Optional

import rospy

# Ensure the repository root is in the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.io.quest2 import Quest2Interface
from src.configs.quest2_config import (
    Q2R_RIGHT_HAND_POSE, Q2R_RIGHT_HAND_INPUTS,
    Q2R_LEFT_HAND_POSE, Q2R_LEFT_HAND_INPUTS,
    Quest2Defaults
)

# --------- USER: set these to your actual bringup launch ----------
QUEST_LAUNCH_CMD = [
    "roslaunch",
    "ros_tcp_endpoint",
    "endpoint.launch",
    "tcp_ip:=192.168.0.182",
    "tcp_port:=10000",
]
# ---------------------------------------------------------------

REQUIRED_TOPICS = [
    Q2R_RIGHT_HAND_POSE,
    Q2R_RIGHT_HAND_INPUTS,
    Q2R_LEFT_HAND_POSE,
    Q2R_LEFT_HAND_INPUTS,
]

def wait_for_ros_master(timeout_s: float = 10.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            import rosgraph
            if rosgraph.is_master_online():
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("ROS master is not online. Is roscore running?")

def wait_for_topics(topics: List[str], timeout_s: float = 20.0) -> None:
    start = time.time()
    remaining = set(topics)
    print(f"[test] Waiting for Unity handshake on topics...")
    while time.time() - start < timeout_s and remaining:
        published = set([t for (t, _type) in rospy.get_published_topics()])
        for t in list(remaining):
            if t in published:
                remaining.discard(t)
        time.sleep(0.5)
    if remaining:
        raise RuntimeError(f"Handshake failed. Unity is not publishing: {sorted(remaining)}")

def launch_quest_node() -> subprocess.Popen:
    """Start ROS TCP Endpoint using roslaunch."""
    env = os.environ.copy()
    p = subprocess.Popen(
        QUEST_LAUNCH_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,  # Kill process group later
        text=True,
        bufsize=1,
    )
    return p

def shutdown_launch(p: subprocess.Popen) -> None:
    if p is None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
    except Exception:
        pass
    try:
        p.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    print("finished checked")

def main():
    rospy.init_node("test_quest2_interface", anonymous=True)
    
    # Use defaults from your config
    defaults = Quest2Defaults()
    hand = defaults.active_hand 

    p = None
    try:
        print("[test] launching quest interface bringup:", " ".join(QUEST_LAUNCH_CMD))
        p = launch_quest_node()

        # 1) Infrastructure Checks
        wait_for_ros_master(timeout_s=10.0)
        
        # 2) Unity Topic Handshake
        wait_for_topics(REQUIRED_TOPICS, timeout_s=30.0)

        # 3) Interface Initialization
        q = Quest2Interface(debug=False)
        
        # 4) Data Flow Handshake (Check if messages contain content)
        rospy.loginfo(f"[test] Checking for {hand} data stream...")
        t0 = time.time()
        while not rospy.is_shutdown():
            # This verifies the callback is actually receiving data
            if q.hand(hand).inputs is not None:
                print(f"[test] Unity Handshake Verified ✅ (Got {hand} data)")
                break
            if time.time() - t0 > 10.0:
                raise RuntimeError(f"Topics found but no data flowing for {hand}. Is the Quest active?")
            rospy.sleep(0.5)

        # 5) Haptic Test
        rospy.loginfo(f"[test] Sending {hand} haptic pulse...")
        q.vibrate(hand, frequency=120.0, amplitude=0.5)
        rospy.sleep(0.1)
        q.vibrate(hand, 0.0, 0.0)

        # 6) Main Monitor Loop
        rospy.loginfo("[test] Entering monitor loop. Press Ctrl+C to exit.")
        r = rospy.Rate(20)
        while not rospy.is_shutdown():
            lower = q.button_lower(hand)
            idx = q.press_index(hand)
            mid = q.press_middle(hand)
            
            h = q.hand(hand)
            tw = h.twist
            lin = (tw.linear.x, tw.linear.y, tw.linear.z) if tw else None
            
            rospy.loginfo_throttle(
                0.5,
                f"[{hand}] lower={lower} | triggers: idx={idx:.2f}, mid={mid:.2f} | lin_velo: {lin}"
            )

            # Use Lower Button as a vibration test (Deadman feedback)
            if lower:
                q.vibrate(hand, frequency=80.0, amplitude=0.1)
            else:
                q.vibrate(hand, 0.0, 0.0)
                
            r.sleep()

        print("\n[test] ALL PASSED ✅")

    except Exception as e:
        rospy.logerr(f"[test] Error occurred: {e}")

    finally:
        print("[test] shutting down bringup")
        shutdown_launch(p)

if __name__ == "__main__":
    main()