import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

def load_all_ee_positions(root_dir):
    episode_dirs = sorted(glob(os.path.join(root_dir, "episode*")))
    all_positions = []

    for epi_dir in tqdm(episode_dirs, desc="Loading EE poses"):
        json_path = os.path.join(epi_dir, "low_dim.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, "r") as f:
            frames = json.load(f)

        for k in sorted(frames.keys(), key=lambda x: int(x)):
            ee_pose = np.array(frames[k]["ee_pose"])
            t = ee_pose[:3, 3]  # Extract translation
            all_positions.append(t)

    return np.array(all_positions)

def plot_3d_trajectory(positions):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1, alpha=0.6)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Distribution of End-Effector Positions")
    ax.view_init(elev=30, azim=135)
    plt.tight_layout()
    plt.show()

# Example usage:
root_dir = "../data/real_stack/place"
positions = load_all_ee_positions(root_dir)
plot_3d_trajectory(positions)
