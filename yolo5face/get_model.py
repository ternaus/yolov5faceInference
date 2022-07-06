import os
from collections import namedtuple
from pathlib import Path

from torch.hub import download_url_to_file

from yolo5face.yoloface.face_detector import YoloDetector

model = namedtuple("model", ["config", "weights", "model"])

models = {
    "yolov5n": model(
        config="https://github.com/ternaus/yolov5faceInference/releases/download/v0.0.1/yolov5n.yaml",
        weights="https://github.com/ternaus/yolov5faceInference/releases/download/v0.0.1/yolov5n_state_dict.pt",
        model=YoloDetector,
    )
}


def get_file_name(url: str) -> str:
    return url.split("/")[-1]


def get_model(
    model_name: str, gpu: int, target_size: int, min_face: int = 24, weights_path: str = "~/.torch/models"
) -> YoloDetector:
    """

    Args:
        model_name: Name of the model. The only model that is supported is "yolov5n"
        gpu: gpu number (int) or -1 or string for cpu
        target_size: target size of smaller image axis (choose lower for faster work). e.g. 480, 720, 1080.
                    None for original resolution.
        min_face: minimal face size in pixels.
        weights_path: Path on the disk to store weights

    Returns:

    """

    cache_path = Path(weights_path).expanduser().absolute()

    cache_path.mkdir(exist_ok=True, parents=True)

    weight_file_path = cache_path / get_file_name(models[model_name].weights)
    config_file_path = cache_path / get_file_name(models[model_name].config)

    if not os.path.exists(weight_file_path):
        download_url_to_file(models[model_name].weights, weight_file_path.as_posix(), progress=True)

    if not os.path.exists(config_file_path):
        download_url_to_file(models[model_name].config, config_file_path.as_posix(), progress=True)

    return models[model_name].model(
        target_size=target_size,
        gpu=gpu,
        min_face=min_face,
        weights_name=weight_file_path,
        config_name=config_file_path,
    )
