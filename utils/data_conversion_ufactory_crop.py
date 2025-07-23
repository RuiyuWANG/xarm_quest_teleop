import os
import re
import json
import h5py
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
from easydict import EasyDict
from collections import defaultdict
import matplotlib.pyplot as plt

import palm.utils.transform_utils as TUtils
import palm.utils.image_utils as PalmUtils
from palm.utils.config_utils import load_config, dict_to_namespace

INTRINSICS_FILE = "charuco_intrinsics.npz"
HAND_EYE_FILE = "handeye_result.npz"
handeye_data = np.load(HAND_EYE_FILE)
X_C = handeye_data["base_T_cam"]
intr_data = np.load(INTRINSICS_FILE)
K = intr_data["K"]

def is_motion_still(pose1, pose2, gripper1, gripper2, trans_thresh=0.002, rot_thresh=np.deg2rad(3), gripper_thresh=5):
    """
    Check if the motion between two SE(3) poses and gripper states is nearly static.
    """
    T1 = np.array(pose1)
    T2 = np.array(pose2)

    # Translation difference
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]
    trans_diff = np.linalg.norm(t1 - t2)

    # Rotation difference
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    R_diff = R1.T @ R2
    cos_angle = (np.trace(R_diff) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    # Gripper position difference
    gripper_diff = abs(gripper1 - gripper2)
    return trans_diff < trans_thresh and angle < rot_thresh and gripper_diff < gripper_thresh

def shift_by_n_and_pad(n, arr):
    arr = np.asarray(arr)
    shifted = arr[n:]
    pad_shape = (n,) + arr.shape[1:]
    pad_last = np.broadcast_to(arr[-1], pad_shape)
    return np.concatenate([shifted, pad_last], axis=0)

def extract_episode_number(path):
    match = re.search(r"episode(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else float("inf")

def get_cropped_image(image: np.ndarray, crop_kwargs: dict, visualize=False) -> np.ndarray:
    ops = crop_kwargs["ops"]
    # original_img_size = np.array([720, 1280])
    # pasted_images = []
    # for i in range(image.shape[0]):
    #     img = image[i]
    #     h, w = img.shape[:2]
    #     canvas = np.zeros((original_img_size[0], original_img_size[1], 3), dtype=np.uint8)
    #     y_offset = (original_img_size[0] - h) // 2
    #     x_offset = (original_img_size[1] - w) // 2
    #     canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img
    #     pasted_images.append(canvas)
    # pasted_images = np.stack(pasted_images)
    
    if "overlay_tcp_crop" in ops:
        K, X_C = crop_kwargs["K"], crop_kwargs["X_C"]
        ee_poses = crop_kwargs["eef_poses"]
        B = ee_poses.shape[0]
        # Expand K and X_C to match the batch if needed
        if K.ndim == 2:
            K = np.repeat(K[None], B, axis=0)
        if X_C.ndim == 2:
            X_C = np.repeat(X_C[None], B, axis=0)
        image_overlay = PalmUtils.overlay_poses(images=image, poses=ee_poses, K=K, X_C=X_C, axis_length=0.08)
        coords = PalmUtils.project_points(K=K, X_C=X_C, poses=ee_poses)
        image_crop = PalmUtils.crop_at_coords(
            images=image_overlay, 
            coords=coords,
            z=ee_poses[:, 2, 3],
            crop_size=crop_kwargs["tcp_crop_size"], 
            height_offset=crop_kwargs["tcp_height_offset"]
        )

        image_resize = np.stack([
            np.array(Image.fromarray(img).resize(crop_kwargs["resize_size"])) for img in image_crop
        ])
    return image_resize

def convert_dataset_to_hdf5(root_dir, config, debug):
    if config.conversion.tcp_crop_size is not None:
        crop_kwargs = {
            "tcp_crop_size": config.conversion.tcp_crop_size,
            "K": K,
            "X_C": X_C,
            "tcp_height_offset": config.conversion.tcp_height_offset,
            "ops": config.conversion.ops.rgb,
            "resize_size": (config.conversion.img_size[1], config.conversion.img_size[0])
        }
    else:
        crop_kwargs = None
    
    episode_dirs = sorted(glob(os.path.join(root_dir, "episode*")), key=extract_episode_number)

    hdf5_path = os.path.join(root_dir, "data.h5")
    if os.path.exists(hdf5_path):
        usr_input = input(f"File {hdf5_path} already exists. Overwrite? (y/n)")
        if usr_input.lower() != "y":
            print("Exiting...")
            exit()
        os.remove(hdf5_path)

    with h5py.File(hdf5_path, "w") as h5f:
        data_group = h5f.create_group("data")
        if debug:
            episode_dirs = episode_dirs[:2]
        for epi_id, epi_dir in enumerate(tqdm(episode_dirs, desc="Processing episodes")):
            rgb_dir = os.path.join(epi_dir, "images")
            json_path = os.path.join(epi_dir, "low_dim.json")
            with open(json_path, "r") as f:
                frames = json.load(f)

            images = []
            cropped_images = []
            states = []
            actions = []
            gripper_status = []
            gripper_qposes = []

            # Precompute still moments and skip those
            keys = sorted([int(k) for k in frames.keys()])
            still_idxs = set()
            for i in range(len(keys) - 1):
                curr_idx = int(keys[i])
                next_idx = int(keys[i + 1])

                # Look back for last non-still index to use as the reference
                if curr_idx in still_idxs:
                    for j in range(i - 1, -1, -1):
                        prev_idx = int(keys[j])
                        if prev_idx not in still_idxs:
                            ref_idx = prev_idx
                            break
                    # else:
                    #     ref_idx = curr_idx  # fallback to current if all before are still
                else:
                    ref_idx = curr_idx

                # print(f"ref_idx: {ref_idx}, next_idx: {next_idx}")
                pose1 = frames[str(ref_idx)]['eef_pose']
                pose2 = frames[str(next_idx)]['eef_pose']
                grip1 = frames[str(ref_idx)]['gripper_qpos'][0]
                grip2 = frames[str(next_idx)]['gripper_qpos'][0]

                if is_motion_still(pose1, pose2, grip1, grip2):
                    still_idxs.add(next_idx)

            # filtered_keys = [int(k) for k in keys if int(k) not in still_idxs]
            filtered_keys = [int(k) for k in keys]
            # print(f"Episode {epi_id}: {len(filtered_keys)} frames, {still_idxs} still moments, {len(still_idxs)}")

            for num in sorted(filtered_keys):
                frame = frames[str(num)]
                img_path = os.path.join(rgb_dir, f"{num}.png")
                if not os.path.exists(img_path):
                    print(f"Warning: Image not found: {img_path}")
                    continue

                # original image
                img_original = np.array(Image.open(img_path)) # (h, w, c)
                img = Image.fromarray(img_original).resize(config.conversion.img_size)
                images.append(np.array(img))
                
                # low dim state
                X_state = np.array(frame["eef_pose"])
                states.append(X_state)
                
                # cropped image
                if crop_kwargs is not None:
                    ee_pose = X_state.copy().reshape(-1, 4, 4)
                    img_to_crop = img_original.copy().reshape(-1, *img_original.shape)
                    crop_kwargs["eef_poses"] = ee_pose
                    cropped_img = get_cropped_image(img_to_crop, crop_kwargs, visualize=False)
                    if len(cropped_img.shape) == 4:
                        cropped_img = cropped_img[0]
                    cropped_images.append(cropped_img)

                gripper_qpos = frame["gripper_qpos"][0]
                gripper_open = 1.0 if gripper_qpos > 400 else 0.0
                gripper_status.append(gripper_open)
                gripper_qposes.append(gripper_qpos)

            if len(states) == 0:
                print(f"Skipping episode {epi_id} due to no valid frames.")
                continue
            
            # create dataset
            episode_group = data_group.create_group(f"episode{epi_id}")
            obs_group = episode_group.create_group("obs")
            action_group = episode_group.create_group("actions")
            low_dim_group = obs_group.create_group("low_dim")
            rgb_group = obs_group.create_group("rgb")
            
            # low dim state
            states = np.stack(states, axis=0)
            state_6d_xyz = TUtils.SE3_to_6D_xyz(states)
            XC = X_C.reshape(-1, 4, 4)
            if XC.shape[0] == 1:
                XC = XC.repeat(states.shape[0], axis=0)
            C_X_H = np.linalg.inv(XC) @ states
            X_Hybrid = C_X_H.copy()
            X_Hybrid[:, :3, 3] = states[:, :3, 3]
            state_6d_in_cam_z = TUtils.SE3_to_6D_z(X_Hybrid)

            # ee action
            SHIFT = 1
            eef_poses_t_shifted = shift_by_n_and_pad(SHIFT, states)
            delta_eef_poses = np.linalg.inv(states) @ eef_poses_t_shifted
            actions = TUtils.SE3_to_6D_xyz(delta_eef_poses)
            
            # gripper action
            prev_state = 1.0 if gripper_qposes[0] > 400 else 0.0
            gripper_qpos_t_shifted = shift_by_n_and_pad(SHIFT, gripper_qposes)
            delta_qpose = gripper_qpos_t_shifted - gripper_qposes
            gripper_action = []
            for i in range(len(delta_qpose)):
                if delta_qpose[i] > 5.0:
                    gripper_action.append(1)
                    prev_state = 1.0
                elif delta_qpose[i] < -5.0:
                    gripper_action.append(0)
                    prev_state = 0.0
                else:
                    gripper_action.append(prev_state)
                    
            gripper_action = np.array(gripper_action)
            images_np = np.array(images)
            cropped_images_np = np.array(cropped_images)
            gripper_np = np.array(gripper_status)
            
            # for i in range(images_np.shape[0]):
            #     cropped_img = cropped_images_np[i]
            #     gripper_state = gripper_np[i]
            #     gripper_act = gripper_action[i]
                
            #     gripper_state_val = "Open" if gripper_state == 1.0 else "Close"
            #     gripper_action_val = "Open" if gripper_act == 1.0 else "Close"
                
            #     text1 = f"State: {gripper_state_val}"
            #     text2 = f"Action: {gripper_action_val}"

            #     # Write text on image
            #     cv2.putText(cropped_img, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #     cv2.putText(cropped_img, text2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #     cv2.imwrite(
            #         os.path.join("../data/debug_imgs", f"{epi_id}_{i}.png"), cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            #     )

            rgb_group.create_dataset("front_rgb", data=images_np)
            rgb_group.create_dataset("front_rgb_overlay_tcp_crop", data=cropped_images_np)
            low_dim_group.create_dataset("eef_6d_xyz", data=state_6d_xyz)
            low_dim_group.create_dataset("gripper_state", data=gripper_np)
            low_dim_group.create_dataset("eef_6d_in_cam_z", data=state_6d_in_cam_z)
            action_group.create_dataset("delta_6d_xyz", data=actions)
            action_group.create_dataset("gripper_action", data=gripper_action)
            episode_group.attrs["num_frames"] = actions.shape[0]

    print(f"Saved dataset to {hdf5_path}")
    return hdf5_path
    
def get_split_indices(f_path, train_split):
    assert 0 < train_split < 1, "Train split must be between 0 and 1"

    with h5py.File(f_path, "a") as f:
        num_frames_per_eps = []
        for eps in f["data"].keys():
            num_frames_per_eps.append(f["data"][eps].attrs["num_frames"])
        hash_map = global_index_to_episode_index(num_frames_per_eps)
        num_train = int(train_split * sum(num_frames_per_eps))
        if f.attrs.get("train_split") != train_split and f.attrs.get("train_split") is not None:
            usr_input = input(
                "Train split mismatch. Do you want to overwrite the existing split? (y/n)"
            )
            if usr_input.lower() == "y":
                if "train_split" in f.attrs.keys():
                    del f.attrs["train_split"]
            else:
                print("Exiting...")
                exit()
        if f.attrs.get("train_split") is None:
            total_num_frames = np.arange(sum(num_frames_per_eps))
            train_indices = np.random.choice(total_num_frames, num_train, replace=False)
            train_hash = [hash_map[train_indices[i]] for i in range(len(train_indices))]
            val_indices = np.setdiff1d(total_num_frames, train_indices)
            val_hash = [hash_map[val_indices[i]] for i in range(len(val_indices))]
            if "train_hash" in f.keys():
                del f["train_hash"]
            f.create_dataset("train_hash", data=train_hash)
            if "val_hash" in f.keys():
                del f["val_hash"]
            f.create_dataset("val_hash", data=val_hash)
        else:
            print("Train split already exists. Exiting...")
            exit()
        f.attrs["train_split"] = train_split


def global_index_to_episode_index(num_frames_per_episode):
    index_chunks = np.cumsum(num_frames_per_episode)
    index_chunks = np.insert(index_chunks, 0, 0)
    total_num_frames = sum(num_frames_per_episode)
    hash_map = {}
    for i in range(total_num_frames):
        for j in range(len(index_chunks) - 1):
            if index_chunks[j] <= i < index_chunks[j + 1]:
                hash_map[i] = (j, int(i - index_chunks[j]))
    return hash_map


def compute_low_dim_mean_and_std(h5_path):
    assert os.path.exists(h5_path), f"File {h5_path} does not exist."
    mean = {"low_dim": defaultdict(float), "actions": defaultdict(float)}
    std = {"low_dim": defaultdict(float), "actions": defaultdict(float)}
    data_min = {"low_dim": defaultdict(float), "actions": defaultdict(float)}
    data_max = {"low_dim": defaultdict(float), "actions": defaultdict(float)}

    with h5py.File(h5_path, "r+") as f:
        eps_keys = list(f["data"].keys())
        low_dim_keys = f["data"][eps_keys[0]]["obs"]["low_dim"].keys()
        for low_dim_key in low_dim_keys:
            all_data = [f["data"][ep]["obs"]["low_dim"][low_dim_key][()] for ep in eps_keys]
            all_data = np.concatenate(all_data, axis=0)
            mean["low_dim"][low_dim_key] = np.mean(all_data, axis=0)
            std["low_dim"][low_dim_key] = np.std(all_data, axis=0)
            data_min["low_dim"][low_dim_key] = np.min(all_data, axis=0)
            data_max["low_dim"][low_dim_key] = np.max(all_data, axis=0)

        action_keys = f["data"][eps_keys[0]]["actions"].keys()
        for action_key in action_keys:
            all_data = [f["data"][ep]["actions"][action_key][()] for ep in eps_keys]
            all_data = np.concatenate(all_data, axis=0)
            mean["actions"][action_key] = np.mean(all_data, axis=0)
            std["actions"][action_key] = np.std(all_data, axis=0)
            data_min["actions"][action_key] = np.min(all_data, axis=0)
            data_max["actions"][action_key] = np.max(all_data, axis=0)

        # write to the same file
        # Save the computed mean and std back to the HDF5 file
        if "stats" in f.keys():
            del f["stats"]
        stats_group = f.create_group("stats")

        low_dim_group = stats_group.create_group("low_dim")
        for key, value in mean["low_dim"].items():
            low_dim_group.create_dataset(f"{key}_mean", data=value)
            low_dim_group.create_dataset(f"{key}_std", data=std["low_dim"][key])
            low_dim_group.create_dataset(f"{key}_min", data=data_min["low_dim"][key])
            low_dim_group.create_dataset(f"{key}_max", data=data_max["low_dim"][key])

        actions_group = stats_group.create_group("actions")
        for key, value in mean["actions"].items():
            actions_group.create_dataset(f"{key}_mean", data=value)
            actions_group.create_dataset(f"{key}_std", data=std["actions"][key])
            actions_group.create_dataset(f"{key}_min", data=data_min["actions"][key])
            actions_group.create_dataset(f"{key}_max", data=data_max["actions"][key])
            

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert RLBench dataset to HDF5 format.")
    parser.add_argument("-d", "--dataset_path", type=str, help="Path to the RLBench dataset.")
    parser.add_argument(
        "-c",
        "--conversion_config_path",
        type=str,
        help="Path to the configuration file for the dataset conversion.",
    )

    parser.add_argument("--debug", action="store_true", help="Print or save debug information.")

    args = parser.parse_args()

    conversion_config = dict_to_namespace(load_config(args.conversion_config_path))
    f_name = convert_dataset_to_hdf5(args.dataset_path,  conversion_config, args.debug)
    get_split_indices(f_name, conversion_config.conversion.train_split_ratio)
    compute_low_dim_mean_and_std(f_name)