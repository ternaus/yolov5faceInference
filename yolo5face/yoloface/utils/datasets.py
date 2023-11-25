import cv2
import numpy as np


def letterbox(
    img: np.ndarray,
    new_shape: int | tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    scaleup: bool = True,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """
    Resizes an image to a new shape while maintaining aspect ratio.

    Args:
        img: The original image.
        new_shape: The new shape as a single int or a tuple of (height, width).
        color: The color used for padding.
        auto: Flag to apply minimum rectangle padding.
        scale_fill: Flag to stretch image to fill new_shape.
        scaleup: Flag to allow scaling up the image.

    Returns:
        A tuple containing:
            - The resized image.
            - The scaling ratio for width and height.
            - The padding applied on width and height.
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = float(new_shape[1] - new_unpad[0]), float(new_shape[0] - new_unpad[1])  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0, 0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
