from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from xarm_quest_teleop.ros import compat as rospy
from scipy.spatial.transform import Rotation as R

from xarm_quest_teleop.configs.ros2_config import package_share_dir
from xarm_quest_teleop.policy.seeker_preprocessing import preprocess_real_rgb_image
from xarm_quest_teleop.utils.conversion_utils import (
    REAL_FRONT_CROP_RATIO,
    adjust_intrinsics_for_image_geometry,
    xyz6g_to_action_abs,
)


class LiveVizRenderer:
    """
    Renders eval debug overlays without owning policy execution.

    This preserves the previous EvalRunner projection and drawing behavior.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._task_meta = self._load_task_meta()

    def _load_task_meta(self) -> dict:
        share_root = str(package_share_dir())
        configured_path = str(getattr(self.cfg, "calibration_path", "all_cams_calib.json"))
        configured_path = os.path.expanduser(configured_path)
        if os.path.isabs(configured_path):
            candidates = [configured_path]
        else:
            candidates = [
                os.path.abspath(configured_path),
                os.path.join(share_root, configured_path),
            ]
        candidates.append(os.path.join(share_root, "config", f"{self.cfg.task_name}.json"))

        seen = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                rospy.logwarn(f"[EvalRunner] failed to load calibration config {path}: {exc}")
                continue

            calibration = payload.get("cameras") or payload.get("calibration") or {}
            if calibration:
                rospy.loginfo(f"[EvalRunner] projection calibration source: {path}")
                return {"calibration": calibration, "calibration_source": path}

        rospy.logwarn("[EvalRunner] no projection calibration found")
        return {}

    def _view_to_camera(self, view: str) -> Optional[str]:
        if view == "agentview":
            return "d435i_front"
        if view == "eye_in_hand":
            return "d405"
        return None

    def _method_kind(self) -> str:
        text = " ".join(
            str(getattr(self.cfg, name, "") or "")
            for name in ("model_name", "model_ckpt_path")
        ).lower()
        if "focuspool" in text or "focus_pool" in text or "focus_refine" in text:
            return "focuspool"
        if "spatial_softmax" in text or "spatial-softmax" in text:
            return "spatial_softmax"
        if "rvt2" in text or "rtv2" in text:
            return "rvt2"
        return "generic"

    def _front_record(self, focus_records):
        for record in list(focus_records):
            if self._view_to_camera(str(getattr(record, "view", ""))) == "d435i_front":
                return record
        return None

    def _camera_intrinsics_for_processed_image(
        self,
        camera_name: str,
        size: int,
        raw_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        try:
            cam_cfg = self._task_meta["calibration"][camera_name]
            intr = cam_cfg.get("intrinsics") or cam_cfg.get("rgb", {}).get("intrinsics")
            K = np.asarray(intr["K"], dtype=np.float32).copy()
            src_w = int(intr["width"])
            src_h = int(intr["height"])
        except Exception:
            return None

        if raw_shape is not None:
            raw_h, raw_w = int(raw_shape[0]), int(raw_shape[1])
            if raw_w > 0 and raw_h > 0 and (raw_w != src_w or raw_h != src_h):
                sx = float(raw_w) / float(src_w)
                sy = float(raw_h) / float(src_h)
                K[0, 0] *= sx
                K[0, 2] *= sx
                K[1, 1] *= sy
                K[1, 2] *= sy
                src_w = raw_w
                src_h = raw_h

        return adjust_intrinsics_for_image_geometry(
            K,
            camera_name=camera_name,
            src_width=src_w,
            src_height=src_h,
            image_size=int(size),
            square_crop=True,
            front_crop_ratio=REAL_FRONT_CROP_RATIO,
        )

    def _pose6_to_matrix_m(self, pose6_mm_rpy: np.ndarray) -> np.ndarray:
        pose6 = np.asarray(pose6_mm_rpy, dtype=np.float32).reshape(6,)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R.from_euler("xyz", pose6[3:6], degrees=False).as_matrix()
        T[:3, 3] = pose6[:3] * 0.001
        return T

    def _camera_pose_world(self, camera_name: str, current_ee_pose6: Optional[np.ndarray]) -> Optional[np.ndarray]:
        try:
            cfg = self._task_meta["calibration"][camera_name]
            setup = str(cfg.get("setup", "")).strip().lower()
            extr = cfg.get("extrinsics") or cfg.get("rgb", {}).get("extrinsics")
            if setup == "eye_to_hand":
                return np.asarray(extr["X_C"], dtype=np.float32)
            if setup == "eye_in_hand":
                if current_ee_pose6 is None:
                    return None
                X_W_E = self._pose6_to_matrix_m(current_ee_pose6)
                X_E_C = np.asarray(extr["X_C"], dtype=np.float32)
                return X_W_E @ X_E_C
        except Exception:
            return None
        return None

    def _project_points(
        self,
        camera_name: str,
        points_world_m: np.ndarray,
        image_size: int,
        current_ee_pose6: Optional[np.ndarray],
        raw_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        K = self._camera_intrinsics_for_processed_image(camera_name, image_size, raw_shape=raw_shape)
        X_W_C = self._camera_pose_world(camera_name, current_ee_pose6)
        if K is None or X_W_C is None:
            return None
        X_C_W = np.linalg.inv(X_W_C)
        pts = np.asarray(points_world_m, dtype=np.float32).reshape(-1, 3)
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        cam = (X_C_W @ pts_h.T).T[:, :3]
        z = cam[:, 2:3]
        valid = z[:, 0] > 1e-6
        if not bool(np.all(valid)):
            return None
        uvw = (K @ cam.T).T
        return uvw[:, :2] / z

    def _draw_focus_record(self, image: np.ndarray, record) -> None:
        pred = getattr(record, "prediction", None)
        self._draw_focus_heatmap(image, record, pred)
        box = getattr(pred, "box_px", None)
        metadata = getattr(record, "metadata", {}) or {}
        draw_box = bool(metadata.get("draw_box", True))
        if box is not None and draw_box:
            box_np = _to_numpy(box).reshape(-1, 4)[-1]
            src_h, src_w = getattr(record, "image_size", (image.shape[0], image.shape[1]))
            sx = float(image.shape[1]) / float(src_w)
            sy = float(image.shape[0]) / float(src_h)
            x1, y1, x2, y2 = box_np
            p1 = (int(round(x1 * sx)), int(round(y1 * sy)))
            p2 = (int(round(x2 * sx)), int(round(y2 * sy)))
            cv2.rectangle(image, p1, p2, (255, 220, 0), 2)
            label = str(getattr(record, "source", "focus"))
            cv2.putText(image, label, (p1[0], max(14, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1)

    def _draw_focus_heatmap(self, image: np.ndarray, record, pred) -> None:
        if pred is None:
            return
        heatmap = getattr(pred, "heatmap", None)
        if heatmap is None:
            heatmap = getattr(pred, "mask_grid", None)
        if heatmap is None:
            return
        mask = _to_numpy(heatmap)
        if mask.size == 0:
            return
        mask = mask.reshape(-1, *mask.shape[-2:])[-1].astype(np.float32, copy=False)
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.maximum(mask, 0.0)
        max_v = float(mask.max()) if mask.size else 0.0
        if max_v <= 1e-8:
            return
        mask = mask / max_v
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        color = cv2.applyColorMap(
            np.clip(mask * 255.0, 0, 255).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32)
        alpha = float((getattr(record, "metadata", {}) or {}).get("mask_alpha", 0.45))
        alpha_map = (mask[..., None] * alpha).astype(np.float32)
        image[:] = np.clip(
            image.astype(np.float32) * (1.0 - alpha_map) + color * alpha_map,
            0,
            255,
        ).astype(np.uint8)

    def _draw_grid_attention_mask(self, image: np.ndarray, record) -> bool:
        pred = getattr(record, "prediction", None)
        metadata = getattr(record, "metadata", {}) or {}
        heatmap = metadata.get("grid_heatmap")
        crop_box = None
        if heatmap is None and pred is not None:
            heatmap = getattr(pred, "heatmap", None)
        if heatmap is None:
            return False

        grid = _to_numpy(heatmap)
        if grid.size == 0:
            return False
        grid = grid.reshape(-1, *grid.shape[-2:])[-1].astype(np.float32, copy=False)
        grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
        grid = np.maximum(grid, 0.0)
        max_v = float(grid.max()) if grid.size else 0.0
        if max_v <= 1e-8:
            return False
        grid = grid / max_v

        src_h, src_w = getattr(record, "image_size", (image.shape[0], image.shape[1]))
        if crop_box is None:
            box = np.asarray([0.0, 0.0, float(src_w), float(src_h)], dtype=np.float32)
        else:
            box = _to_numpy(crop_box).reshape(-1, 4)[-1].astype(np.float32, copy=False)

        sx = float(image.shape[1]) / float(src_w)
        sy = float(image.shape[0]) / float(src_h)
        x1 = int(round(float(box[0]) * sx))
        y1 = int(round(float(box[1]) * sy))
        x2 = int(round(float(box[2]) * sx))
        y2 = int(round(float(box[3]) * sy))
        x1 = max(0, min(image.shape[1] - 1, x1))
        y1 = max(0, min(image.shape[0] - 1, y1))
        x2 = max(x1 + 1, min(image.shape[1], x2))
        y2 = max(y1 + 1, min(image.shape[0], y2))

        mask = cv2.resize(
            grid,
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_NEAREST,
        )
        color = np.asarray([20.0, 120.0, 255.0], dtype=np.float32)
        alpha = (0.12 + 0.58 * mask[..., None]).astype(np.float32)
        roi = image[y1:y2, x1:x2].astype(np.float32)
        roi = roi * (1.0 - alpha) + color[None, None, :] * alpha
        image[y1:y2, x1:x2] = np.clip(roi, 0, 255).astype(np.uint8)

        gh, gw = grid.shape[-2:]
        grid_overlay = image.copy()
        line_color = (255, 255, 255)
        for i in range(gh + 1):
            y = int(round(y1 + (y2 - y1) * float(i) / float(max(1, gh))))
            cv2.line(grid_overlay, (x1, y), (x2, y), line_color, 1)
        for j in range(gw + 1):
            x = int(round(x1 + (x2 - x1) * float(j) / float(max(1, gw))))
            cv2.line(grid_overlay, (x, y1), (x, y2), line_color, 1)
        image[:] = cv2.addWeighted(grid_overlay, 0.35, image, 0.65, 0.0)
        return True

    def _draw_record_box(self, image: np.ndarray, record) -> bool:
        pred = getattr(record, "prediction", None)
        box = None if pred is None else getattr(pred, "box_px", None)
        if box is None:
            return False
        box_np = _to_numpy(box).reshape(-1, 4)[-1]
        src_h, src_w = getattr(record, "image_size", (image.shape[0], image.shape[1]))
        sx = float(image.shape[1]) / float(src_w)
        sy = float(image.shape[0]) / float(src_h)
        x1, y1, x2, y2 = box_np
        p1 = (int(round(x1 * sx)), int(round(y1 * sy)))
        p2 = (int(round(x2 * sx)), int(round(y2 * sy)))
        cv2.rectangle(image, p1, p2, (255, 220, 0), 3)
        return True

    def _draw_spatial_keypoints(self, image: np.ndarray, record) -> bool:
        metadata = getattr(record, "metadata", {}) or {}
        points = metadata.get("points_px")
        if points is None:
            return False
        points = _to_numpy(points).reshape(-1, 2).astype(np.float32, copy=False)
        if points.size == 0:
            return False
        mean_idx = metadata.get("mean_point_index")
        if mean_idx is not None:
            mean_idx = int(mean_idx)
            keep = np.ones((points.shape[0],), dtype=bool)
            if 0 <= mean_idx < points.shape[0]:
                keep[mean_idx] = False
            points = points[keep]

        src_h, src_w = getattr(record, "image_size", (image.shape[0], image.shape[1]))
        sx = float(image.shape[1]) / float(src_w)
        sy = float(image.shape[0]) / float(src_h)
        for p in points:
            x = int(round(float(p[0]) * sx))
            y = int(round(float(p[1]) * sy))
            cv2.circle(image, (x, y), 5, (0, 0, 0), -1)
            cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
        return True

    def _is_attention_record(self, record) -> bool:
        pred = getattr(record, "prediction", None)
        if pred is None:
            return False
        if getattr(pred, "heatmap", None) is None and getattr(pred, "mask_grid", None) is None:
            return False

        source = str(getattr(record, "source", "")).lower()
        pred_source = str(getattr(pred, "source", "")).lower()
        metadata = getattr(pred, "metadata", {}) or {}
        record_metadata = getattr(record, "metadata", {}) or {}
        return (
            source == "stage_pooled"
            or pred_source == "stage_pooled"
            or metadata.get("kind") == "attention_heatmap"
            or record_metadata.get("kind") == "attention_heatmap"
        )

    def _draw_target_frame(
        self,
        image: np.ndarray,
        camera_name: str,
        target_pose6: np.ndarray,
        current_ee_pose6: Optional[np.ndarray],
        raw_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        T_W_T = self._pose6_to_matrix_m(target_pose6)
        origin = T_W_T[:3, 3]
        axis_len = 0.04
        pts = np.stack(
            [
                origin,
                origin + T_W_T[:3, 0] * axis_len,
                origin + T_W_T[:3, 1] * axis_len,
                origin + T_W_T[:3, 2] * axis_len,
            ],
            axis=0,
        )
        uv = self._project_points(
            camera_name,
            pts,
            image.shape[0],
            current_ee_pose6,
            raw_shape=raw_shape,
        )
        if uv is None:
            cv2.putText(image, "target projection unavailable", (8, image.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 80), 1)
            return
        uv = np.round(uv).astype(int)
        o = tuple(uv[0])
        for end, color in [(uv[1], (255, 60, 60)), (uv[2], (60, 255, 60)), (uv[3], (60, 120, 255))]:
            cv2.line(image, o, tuple(end), color, 2)
        cv2.circle(image, o, 4, (255, 255, 255), -1)

    def _draw_action_points(
        self,
        image: np.ndarray,
        camera_name: str,
        acts: np.ndarray,
        current_ee_pose6: Optional[np.ndarray],
        raw_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        if acts.shape[0] == 0 or acts.shape[1] < 3:
            return
        points_world_m = np.asarray(acts[:, :3], dtype=np.float32) * 0.001
        uv = self._project_points(
            camera_name,
            points_world_m,
            image.shape[0],
            current_ee_pose6,
            raw_shape=raw_shape,
        )
        if uv is None:
            return
        uv = np.round(uv).astype(int)
        color_start = np.array([30, 220, 255], dtype=np.float32)
        color_end = np.array([255, 70, 30], dtype=np.float32)
        for i, p in enumerate(uv):
            alpha = 0.0 if uv.shape[0] <= 1 else float(i) / float(uv.shape[0] - 1)
            color = tuple(np.round((1.0 - alpha) * color_start + alpha * color_end).astype(int).tolist())
            cv2.circle(image, tuple(p), 3, color, -1)
            if i > 0:
                cv2.line(image, tuple(uv[i - 1]), tuple(p), color, 1)

    def make_frame(
        self,
        temporal_obs: Dict[str, Any],
        acts: np.ndarray,
        current_exec_start_idx: int,
        focus_records,
    ) -> Optional[np.ndarray]:
        if acts is None:
            return None
        acts = np.asarray(acts, dtype=np.float32)
        if acts.ndim == 3:
            acts = acts[0]
        if acts.ndim == 1:
            acts = acts[None, :]
        if acts.shape[0] == 0 or acts.shape[1] < 9:
            return None

        target_idx = min(int(current_exec_start_idx), acts.shape[0] - 1)
        target_pose6, _ = xyz6g_to_action_abs(acts[target_idx : target_idx + 1, :10])
        target_pose6 = target_pose6[0]
        xyz_span = np.ptp(acts[:, :3], axis=0)
        xyz_delta = acts[-1, :3] - acts[0, :3]
        current_ee = temporal_obs.get("low_dim", {}).get("ee_pose6", [None])[-1]
        current_ee = None if current_ee is None else np.asarray(current_ee, dtype=np.float32)

        record_by_cam = {}
        for record in list(focus_records):
            cam = self._view_to_camera(str(getattr(record, "view", "")))
            if cam is not None:
                record_by_cam[cam] = record

        panels = []
        for cam_name in self.cfg.rgb_cams:
            seq = temporal_obs.get("rgb", {}).get(cam_name)
            if not seq:
                continue
            raw_image = seq[-1]
            raw_shape = raw_image.shape[:2]
            image = preprocess_real_rgb_image(raw_image, camera_name=cam_name).copy()
            if cam_name in record_by_cam:
                self._draw_focus_record(image, record_by_cam[cam_name])
            self._draw_action_points(
                image,
                cam_name,
                acts[:, :10],
                current_ee,
                raw_shape=raw_shape,
            )
            self._draw_target_frame(
                image,
                cam_name,
                target_pose6,
                current_ee,
                raw_shape=raw_shape,
            )
            cv2.putText(image, cam_name, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(
                image,
                f"target[{target_idx}] xyz={target_pose6[:3].round(1)} span={xyz_span.round(1)}",
                (8, image.shape[0] - 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                image,
                f"horizon delta={xyz_delta.round(1)}",
                (8, image.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
            )
            panels.append(image)
        if not panels:
            return None
        return np.concatenate(panels, axis=1)

    def make_attention_frame(
        self,
        temporal_obs: Dict[str, Any],
        focus_records,
    ) -> Optional[np.ndarray]:
        return self.make_method_viz_frame(
            temporal_obs=temporal_obs,
            focus_records=focus_records,
        )

    def make_method_viz_frame(
        self,
        temporal_obs: Dict[str, Any],
        focus_records,
    ) -> Optional[np.ndarray]:
        cam_name = "d435i_front"
        seq = temporal_obs.get("rgb", {}).get(cam_name)
        if not seq:
            return None
        record = self._front_record(focus_records)
        if record is None:
            return None
        image = preprocess_real_rgb_image(seq[-1], camera_name=cam_name).copy()
        method = self._method_kind()
        drew = False
        if method == "focuspool":
            drew = self._draw_grid_attention_mask(image, record)
        elif method == "spatial_softmax":
            drew = self._draw_spatial_keypoints(image, record)
        elif method == "rvt2":
            drew = self._draw_record_box(image, record)
        else:
            pred = getattr(record, "prediction", None)
            if self._is_attention_record(record):
                drew = self._draw_grid_attention_mask(image, record)
            elif pred is not None and getattr(pred, "box_px", None) is not None:
                drew = self._draw_record_box(image, record)

        if not drew:
            return None
        return image


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().to("cpu").numpy()
    return np.asarray(value)
