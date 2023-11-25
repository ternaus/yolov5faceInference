from typing import Any

import numpy as np
import torch

from yolo5face.yoloface.face_detector import YoloDetector
from yolo5face.yoloface.types import BoxType, KeypointType


class YoloDetectorAggregator:
    def __init__(self, target_sizes: int | list[int], **yolo_args: Any) -> None:
        self.yolo_args = yolo_args
        self.target_sizes = target_sizes if isinstance(target_sizes, list) else [target_sizes]

    def nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
        """Applies Non-Maximum Suppression (NMS) to filter boxes."""
        return torch.ops.torchvision.nms(boxes.type(torch.float), scores.type(torch.float), iou_threshold)

    def __call__(self, image: np.ndarray) -> tuple[BoxType, KeypointType]:
        all_boxes, all_keypoints = [], []

        for size in self.target_sizes:
            detector = YoloDetector(target_size=size, **self.yolo_args)

            boxes, keypoints = detector(image)

            all_boxes.extend(boxes)
            all_keypoints.extend(keypoints)

            print(all_boxes, all_keypoints)

        if len(self.target_sizes) > 1:
            # Perform aggregation with NMS if multiple target sizes are used
            return self.aggregate_predictions(all_boxes, all_keypoints)

        return all_boxes[0], all_keypoints[0]

    def aggregate_predictions(
        self,
        all_boxes: list[BoxType],
        all_keypoints: list[KeypointType],
    ) -> tuple[BoxType, KeypointType]:
        if not all_boxes or not all(list(all_boxes)):
            # No boxes to process
            return [], []

        boxes_tensor = torch.tensor([box for sublist in all_boxes for box in sublist])
        scores = torch.ones(len(boxes_tensor))

        # Apply NMS
        keep_indices = self.nms(boxes_tensor, scores)

        # Filter boxes and keypoints
        filtered_boxes = boxes_tensor[keep_indices].tolist()
        filtered_keypoints = [all_keypoints[0][i] for i in keep_indices]

        return filtered_boxes, filtered_keypoints
