# Wrapper over YoloV5Face

![](https://habrastorage.org/webt/gy/-1/xd/gy-1xdtfz3_i7xxt-nqzl4mfhuw.jpeg)

### Installation

```
pip install -U yolo5face
```

### Inference
```
from yolo5face.get_model import get_model

model = get_model("yolov5n", gpu=-1, target_size=512, min_face=24)

image = cv2.imread(<IMAGE_PATH>)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes, key_points = model(image)

```

* `gpu` - GPU number, `-1` or `cpu` for CPU
* `target_size` - min size of the target_image
* `min_face` - minimum face size in pixels. All faces that have side smaller than `min_face` will be ignored.
