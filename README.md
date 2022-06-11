# Wrapper over YoloV5Face

Wrapper over YoloV5Face for a better user experience.

```
from yolo5face.get_model import get_model

model = get_model("yolov5n", gpu=-1, target_size=512)
```

* `gpu` - GPU number, `-1` or `cpu` for CPU
* `target_size` - min size of the target_image
* `min_face` - minimum face size in pixels.
