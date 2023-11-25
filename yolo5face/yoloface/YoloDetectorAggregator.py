from typing import Any

import numpy as np
import torch

from yolo5face.yoloface.face_detector import YoloDetector
from yolo5face.yoloface.types import BoxType, KeypointType


class YoloDetectorAggregator:
    def __init__(self, **yolo_args: Any) -> None:
        self.yolo_args = yolo_args
        self.detector = YoloDetector(**self.yolo_args)

    def nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
        """Applies Non-Maximum Suppression (NMS) to filter boxes."""
        return torch.ops.torchvision.nms(boxes.type(torch.float), scores.type(torch.float), iou_threshold)

    def __call__(
        self,
        image: np.ndarray,
        target_size: int | list[int],
    ) -> tuple[list[BoxType], list[KeypointType], list[float]]:
        all_boxes, all_keypoints, all_scores = [], [], []

        if isinstance(target_size, int):
            target_size = [target_size]

        for size in target_size:
            boxes, keypoints, scores = self.detector(image, size)

            all_boxes.extend(boxes)
            all_keypoints.extend(keypoints)
            all_scores.extend(scores)

        if len(target_size) > 1:
            # Perform aggregation with NMS if multiple target sizes are used
            return self.aggregate_predictions(all_boxes, all_keypoints, all_scores)

        return all_boxes, all_keypoints, all_scores

    def aggregate_predictions(
        self,
        all_boxes: list[BoxType],
        all_keypoints: list[KeypointType],
        all_scores: list[float],
    ) -> tuple[list[BoxType], list[KeypointType], list[float]]:
        if not all_boxes or not all(list(all_boxes)):
            # No boxes to process
            return [], [], []

        boxes_tensor = torch.tensor(all_boxes)
        scores_tensor = torch.tensor(all_scores)

        # Apply NMS
        keep_indices = self.nms(boxes_tensor, scores_tensor)

        # Filter boxes, keypoints, and scores
        filtered_boxes = boxes_tensor[keep_indices].tolist()
        filtered_keypoints = [all_keypoints[i] for i in keep_indices]
        filtered_scores = scores_tensor[keep_indices].tolist()

        return filtered_boxes, filtered_keypoints, filtered_scores
