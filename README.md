# Wrapper over YoloV5Face

![](https://habrastorage.org/webt/gy/-1/xd/gy-1xdtfz3_i7xxt-nqzl4mfhuw.jpeg)

A Python wrapper for the YoloV5Face model, providing easy-to-use functionalities for face detection in images.

## Installation

Install the YoloV5Face wrapper using pip:

```bash
pip install -U yolo5face
```

## Inference

Use the wrapper to quickly deploy face detection in your projects:

```bash
from yolo5face.get_model import get_model
import cv2

model = get_model("yolov5n", device=-1, target_size=512, min_face=24)

image = cv2.imread(<IMAGE_PATH>)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes, key_points, scores = model(image)
```

* **device**: Specify device `cpu`, `cuda`, `mps` or integer for the number of cuda device.
* **target_size**: The minimum size of the target image for detection.
* **min_face**: The minimum face size in pixels. Faces smaller than this value will be ignored.

## Enhanced Detection with Aggregated Target Sizes

In addition to standard detection, this wrapper supports enhanced detection capabilities by aggregating results over multiple target sizes. This feature is especially useful in scenarios where face sizes vary significantly within the same image.

To use this feature:

```bash
from yolo5face.get_model import get_model

model = get_model("yolov5n", device=-1, target_size=[320, 640, 1280], min_face=24)

# Aggregate detections over the specified target sizes
boxes, key_points, scores = aggregator(image)
```

This approach leverages multiple detections at different scales, followed by Non-Maximum Suppression, to provide a more comprehensive set of detections.

## License

This YoloV5Face wrapper is released under the [MIT License](LICENSE)
