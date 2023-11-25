from deepdiff import DeepDiff
from pytest import mark

from tests.conftest import test_images as images
from yolo5face.get_model import get_model

model = get_model("yolov5n", gpu=-1, target_size=512)


@mark.parametrize(
    ["image", "face"],
    [
        (images[image_name]["image"], images[image_name]["faces"])
        for image_name in ["with_faces_rescaled", "with_no_faces"]
    ],
)
def test_face_detection(image, face):
    boxes, points = model(image)

    for box_id, box in enumerate(boxes):
        assert len(DeepDiff(box, face[box_id]["box"])) == 0
        assert len(DeepDiff(points[box_id], face[box_id]["keypoints"])) == 0
