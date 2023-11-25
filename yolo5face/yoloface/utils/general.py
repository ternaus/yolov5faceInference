import math
import time

import numpy as np
import torch
import torchvision

from yolo5face.yoloface.types import ShapeLike, TensorLike

MAX_NMS_COMPARISONS = 3000


def check_img_size(img_size: int, s: int = 32) -> int:
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, s)  # ceil gs-multiple
    if new_size != img_size:
        print(f"WARNING: --img-size {img_size:g} must be multiple of max stride {s:g}, updating to {new_size:g}")
    return new_size


def make_divisible(x: int, divisor: int) -> int:
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def xyxy2xywh(x: TensorLike) -> TensorLike:
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x: TensorLike) -> TensorLike:
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(
    img1_shape: ShapeLike,
    coords: torch.Tensor,
    img0_shape: tuple[int, int],
) -> torch.Tensor:
    height_0, width_0 = img0_shape[:2]
    height_1, width_1 = img1_shape[:2]

    gain = min(height_1 / height_0, width_1 / width_0)  # gain  = old / new
    pad = (width_1 - width_0 * gain) / 2, (height_1 - height_0 * gain) / 2  # wh padding

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes: torch.Tensor, img_shape: ShapeLike) -> None:
    # Clip bounding xyxy bounding boxes to image shape (height, width)

    height, width = img_shape[:2]

    boxes[:, 0].clamp_(0, width)  # x1
    boxes[:, 1].clamp_(0, height)  # y1
    boxes[:, 2].clamp_(0, width)  # x2
    boxes[:, 3].clamp_(0, height)  # y2


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    # Return intersection-over-union (Jaccard index) of boxes
    def box_area(box: torch.Tensor) -> torch.Tensor:
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression_face(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> list[torch.Tensor]:
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # (pixels) maximum box width and height
    max_wh = 4096
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 16), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 15:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, landmarks, cls)

        conf, j = x[:, 15:].max(1, keepdim=True)
        x = torch.cat((box, conf, x[:, 5:15], j.float()), 1)[conf.view(-1) > conf_thres]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 15:16] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS

        if merge and (1 < n < MAX_NMS_COMPARISONS):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def scale_coords_landmarks(
    source_shape: ShapeLike,
    coords: TensorLike,
    target_shape: ShapeLike,
) -> TensorLike:
    """
    Rescales coordinates from one image shape to another.

    Parameters:
        source_shape (tuple[int, int]): Shape (height, width) of the source image.
        coords (TensorLike): Coordinates to be scaled.
        target_shape (tuple[int, int]): Shape (height, width) of the target image.

    Returns:
        TensorLike: Scaled coordinates.
    """
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]

    # Calculate scaling gain and padding
    gain = min(source_width / target_width, source_height / target_height)
    pad_x = (source_width - target_width * gain) / 2
    pad_y = (source_height - target_height * gain) / 2

    # Adjust x and y coordinates and clamp them
    for i in range(0, 10, 2):  # Iterating over x coordinates (even indices)
        coords[:, i] = (coords[:, i] - pad_x) / gain
        coords[:, i].clamp_(0, target_width)

    for i in range(1, 10, 2):  # Iterating over y coordinates (odd indices)
        coords[:, i] = (coords[:, i] - pad_y) / gain
        coords[:, i].clamp_(0, target_height)

    return coords
