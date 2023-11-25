# Wrapper over YoloV5Face

![](https://habrastorage.org/webt/gy/-1/xd/gy-1xdtfz3_i7xxt-nqzl4mfhuw.jpeg)

A user-friendly Python wrapper for the YoloV5Face model, designed to simplify face detection in images. This wrapper offers straightforward functionalities for quick integration into Python projects, along with customization options for handling various face detection scenarios.

## Installation

Install the YoloV5Face wrapper using pip to easily incorporate it into your projects:

```bash
pip install -U yolo5face
```

## Face Detection: Standard and Enhanced

The YoloV5Face wrapper supports both standard and enhanced face detection. The standard detection is suitable for most use cases, while the enhanced detection, which aggregates results over multiple target sizes, is ideal for images with faces of varying sizes.

### Getting Started

To detect faces in an image:

```bash
from yolo5face.get_model import get_model
import cv2

# Initialize the model
model = get_model("yolov5n", device=-1, min_face=24)

# Load your image
image = cv2.imread(<IMAGE_PATH>)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Standard Detection
boxes, key_points, scores = model(image, target_size=512)

# Enhanced Detection (aggregating over multiple target sizes)
enhanced_boxes, enhanced_key_points, enhanced_scores = model(image, target_size=[320, 640, 1280])
```

Parameters:

* **device**: Set the processing device (cpu, cuda, mps, or CUDA device number).
* **target_size**: For standard detection, it's the minimum size of the target image. For enhanced detection, provide a list of sizes for better accuracy.
* **min_face**: Minimum size of faces to detect in pixels. Smaller faces will be ignored.

This approach, especially the enhanced detection, uses multiple scales for improved accuracy and is followed by Non-Maximum Suppression to refine the results.

## License

This YoloV5Face wrapper is released under the [MIT License](LICENSE).
