from pathlib import Path

import albumentations as albu
import cv2
import numpy as np

TARGET_IMAGE_SIZE = 1280


def load_rgb(image_path: Path) -> np.ndarray:
    image = cv2.imread(image_path.as_posix())
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


image_with_faces_path = Path(__file__).parent / "data" / "13.jpg"

image_with_faces = load_rgb(image_with_faces_path)

image_with_faces_rescaled = albu.Compose(
    [albu.LongestMaxSize(max_size=TARGET_IMAGE_SIZE, p=1, interpolation=cv2.INTER_CUBIC)]
)(image=image_with_faces)["image"]

image_with_no_face_path = Path(__file__).parent / "data" / "no_face.jpg"
image_with_no_face = load_rgb(image_with_no_face_path)

test_images = {
    "with_faces_rescaled": {
        "image_path": image_with_faces_path,
        "image": image_with_faces_rescaled,
        "url": "https://ternaustests.s3.amazonaws.com/13.jpg",
        "gcp_path": "gs://monday-frontend-test.appspot.com/test/images/user_id/13.jpg",
        "faces": [
            {
                "box": [353, 128, 467, 277],
                "keypoints": [[392, 185], [442, 186], [420, 219], [390, 230], [438, 232]],
            },
            {
                "box": [599, 158, 702, 294],
                "keypoints": [[633, 213], [676, 212], [657, 239], [636, 259], [672, 258]],
            },
            {
                "box": [896, 213, 1004, 342],
                "keypoints": [[915, 259], [954, 270], [922, 290], [915, 303], [948, 311]],
            },
        ],
        "faces_aggregate": [
            {
                "box": [349, 123, 467, 278],
                "keypoints": [[391, 183], [442, 187], [420, 220], [386, 230], [438, 234]],
            },
            {
                "box": [598, 159, 704, 292],
                "keypoints": [[632, 214], [678, 213], [656, 241], [635, 260], [673, 259]],
            },
            {
                "box": [896, 213, 1004, 342],
                "keypoints": [[915, 259], [954, 270], [922, 290], [915, 303], [948, 311]],
            },
        ],
    },
    "with_no_faces": {
        "image": image_with_no_face,
        "sha256hash": "034fa2852f6715cd214c1d22970112b1e9c1ad61c1de43231747106300d5b405",
        "url": "https://ternaustests.s3.amazonaws.com/no_face.jpg",
        "gcp_path": "gs://monday-frontend-test.appspot.com/test/images/user_id/no_face.jpg",
        "image_path": image_with_no_face_path,
        "faces": [{"box": [], "keypoints": []}],
        "faces_aggregate": [{"box": [], "keypoints": []}],
    },
}
