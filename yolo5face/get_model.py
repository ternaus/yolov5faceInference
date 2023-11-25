from pathlib import Path
from typing import NamedTuple

from torch.hub import download_url_to_file

from yolo5face.yoloface.YoloDetectorAggregator import YoloDetectorAggregator


class Model(NamedTuple):
    config: str
    weights: str
    model: str


models = {
    "yolov5n": {
        "config_name": "https://github.com/ternaus/yolov5faceInference/releases/download/v0.0.1/yolov5n.yaml",
        "weights_name": "https://github.com/ternaus/yolov5faceInference/releases/download/v0.0.1/yolov5n_state_dict.pt",
    },
}


def get_file_name(url: str) -> str:
    return url.split("/")[-1]


def get_model(
    model_name: str,
    gpu: int,
    target_size: int,
    min_face: int = 24,
    weights_path: str = "~/.torch/models",
) -> YoloDetectorAggregator:
    cache_path = Path(weights_path).expanduser().absolute()
    cache_path.mkdir(exist_ok=True, parents=True)

    weights_name = models[model_name]["weights_name"]
    config_name = models[model_name]["config_name"]

    weight_file_path = cache_path / get_file_name(weights_name)
    config_file_path = cache_path / get_file_name(config_name)

    if not weight_file_path.exists():
        download_url_to_file(weights_name, weight_file_path.as_posix(), progress=True)

    if not config_file_path.exists():
        download_url_to_file(config_name, config_file_path.as_posix(), progress=True)

    return YoloDetectorAggregator(
        target_sizes=target_size,
        min_face=min_face,
        gpu=gpu,
        weights_name=weight_file_path,
        config_name=config_file_path,
    )
