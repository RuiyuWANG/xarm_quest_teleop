from glob import glob
from pathlib import Path

from setuptools import find_packages, setup


package_name = "xarm_quest_teleop"
PACKAGE_ROOT = Path(__file__).resolve().parent


def data_files():
    files = [
        ("share/ament_index/resource_index/packages", [str(Path("resource") / package_name)]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}", ["all_cams_calib.json"]),
    ]
    install_roots = [
        (PACKAGE_ROOT / "config", "config"),
        (PACKAGE_ROOT / "launch", "launch"),
    ]
    for source_root, install_root in install_roots:
        for path in glob(str(source_root / "**" / "*"), recursive=True):
            if Path(path).is_file():
                rel_parent = Path(path).parent.relative_to(source_root)
                dest = Path("share") / package_name / install_root / rel_parent
                files.append((str(dest), [str(Path(install_root) / rel_parent / Path(path).name)]))
    return files


setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=data_files(),
    install_requires=[
        "numpy",
        "opencv-python",
        "Pillow",
        "PyYAML",
        "scipy",
        "setuptools",
    ],
    zip_safe=False,
    maintainer="Ruiyu",
    maintainer_email="ruiyu@example.com",
    description="ROS2 teleoperation, data collection, and policy evaluation for xArm Quest Teleop manipulation.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "quest_stamped_node = xarm_quest_teleop.nodes.quest_stamped_node:main",
            "run_quest_xarm_teleop_sync = xarm_quest_teleop.scripts.run_quest_xarm_teleop_sync:main",
            "run_data_collection = xarm_quest_teleop.scripts.run_data_collection:main",
            "run_policy_eval = xarm_quest_teleop.scripts.run_policy_eval:main",
            "camera_calibration = xarm_quest_teleop.scripts.camera_calibration:main",
            "convert_raw_data_to_cache = xarm_quest_teleop.scripts.convert_raw_data_to_cache:main",
            "convert_eval_videos_to_gifs = xarm_quest_teleop.scripts.convert_eval_videos_to_gifs:main",
            "export_real_task_gifs = xarm_quest_teleop.scripts.export_real_task_gifs:main",
            "gif_temporal_strips = xarm_quest_teleop.scripts.gif_temporal_strips:main",
            "project_lmdb_action_chunk = xarm_quest_teleop.scripts.project_lmdb_action_chunk:main",
            "render_eval_feature_activation_videos = xarm_quest_teleop.scripts.render_eval_feature_activation_videos:main",
            "retime_eval_gifs = xarm_quest_teleop.scripts.retime_eval_gifs:main",
            "ros_rgb_view = xarm_quest_teleop.scripts.ros_rgb_view:main",
        ],
        "xarm_quest_teleop.policies": [
            "seeker = xarm_quest_teleop.policy.seeker_policy:SeekerPolicy",
            "cache_replay = xarm_quest_teleop.policy.cache_action_replay_policy:CacheActionReplayPolicy",
        ],
    },
)
