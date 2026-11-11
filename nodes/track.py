"""人物跟踪节点：YOLO 检测 + IoU 关联，输出 ExMR 需要的框与内参。"""

from pathlib import Path

import numpy as np

from .frames import default_intrinsics
from .loader import models_root


def _yolo_weights(name: str) -> Path:
    root = models_root()
    for candidate in (root / "yolo" / name, root / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"未找到检测权重 {name}，请放到 models/zmocap/yolo/{name}")


class ZMocapTrack:
    """逐帧检测人体并保持同一人物的 track id。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "weights": (["yolov8x.pt", "yolov8s.pt"], {"default": "yolov8x.pt"}),
                "confidence": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.95, "step": 0.05}),
                "fov_deg": ("FLOAT", {"default": 55.0, "min": 20.0, "max": 120.0, "step": 1.0}),
                "person": ("INT", {"default": 0, "min": 0, "max": 7}),
            }
        }

    RETURN_TYPES = ("ZMOCAP_TRACKS",)
    RETURN_NAMES = ("tracks",)
    FUNCTION = "run"
    CATEGORY = "zmocap"

    def run(self, images, weights, confidence, fov_deg, person):
        from ultralytics import YOLO

        frames = images.detach().cpu().numpy() if hasattr(images, "detach") else np.asarray(images)
        if frames.ndim == 3:
            frames = frames[None]
        height, width = frames.shape[1], frames.shape[2]
        model = YOLO(str(_yolo_weights(weights)))
        results = model.predict(
            [np.clip(frame, 0, 1)[..., ::-1] * 255 for frame in frames],
            conf=confidence,
            classes=[0],
            verbose=False,
        )
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((0, 4))
            detections.append(boxes)

        tracks = _associate([self._largest(boxes) for boxes in detections])
        ids = sorted({tid for tid in tracks if tid is not None}, key=lambda tid: -sum(1 for t in tracks if t == tid))
        if not ids:
            raise ValueError("没有检测到人体")
        chosen = ids[min(person, len(ids) - 1)]
        boxes_out = np.zeros((len(tracks), 4), dtype=np.float32)
        observed = np.zeros(len(tracks), dtype=bool)
        for index, (tid, boxes) in enumerate(zip(tracks, detections)):
            if tid == chosen:
                boxes_out[index] = _pick(boxes)
                observed[index] = True
        if observed.sum() == 0:
            raise ValueError("选定人物没有有效帧")
        _fill_gaps(boxes_out, observed)
        intrinsics = default_intrinsics(width, height, fov_deg)
        return (
            {
                "boxes": boxes_out,
                "observed": observed,
                "K": np.repeat(intrinsics[None], len(boxes_out), axis=0),
                "width": width,
                "height": height,
                "track_ids": ids,
                "person": chosen,
            },
        )

    @staticmethod
    def _largest(boxes):
        if len(boxes) == 0:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return boxes[int(np.argmax(areas))]


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def _associate(per_frame, iou_threshold=0.3):
    """贪心 IoU 关联：每帧最多一个人物框，保持 id 稳定。"""
    tracks = []
    active = {}
    next_id = 0
    for box in per_frame:
        if box is None:
            tracks.append(None)
            continue
        best, best_iou = None, iou_threshold
        for tid, last in active.items():
            score = _iou(box, last)
            if score > best_iou:
                best, best_iou = tid, score
        if best is None:
            best = next_id
            next_id += 1
        active[best] = box
        tracks.append(best)
    return tracks


def _pick(boxes):
    return boxes[0] if len(boxes) else np.zeros(4, dtype=np.float32)


def _fill_gaps(boxes, observed):
    indices = np.where(observed)[0]
    if len(indices) == 0:
        return
    for index in range(len(boxes)):
        if not observed[index]:
            nearest = indices[int(np.argmin(np.abs(indices - index)))]
            boxes[index] = boxes[nearest]
