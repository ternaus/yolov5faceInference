import copy
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from yolo5face.yoloface.models.yolo import Model
from yolo5face.yoloface.types import BoxType, KeypointType
from yolo5face.yoloface.utils.datasets import letterbox
from yolo5face.yoloface.utils.general import (
    check_img_size,
    non_max_suppression_face,
    scale_coords,
    scale_coords_landmarks,
)


class YoloDetector:
    def __init__(
        self,
        weights_name: str = "yolov5n_state_dict.pt",
        config_name: str = "yolov5n.yaml",
        gpu: int | str = 0,
        min_face: int = 100,
        target_size: int | None = None,
    ):
        """
        weights_name: name of file with network weights in weights/ folder.
        config_name: name of .yaml config with network configuration from models/ folder.
        gpu : gpu number (int) or -1 or string for cpu.
        min_face : minimal face size in pixels.
        target_size : target size of smaller image axis (choose lower for faster work). e.g. 480, 720, 1080.
                    None for original resolution.

        """
        self._class_path = Path(__file__).parent.absolute()
        self.gpu = gpu
        self.target_size = target_size
        self.min_face = min_face

        self.detector = self.init_detector(weights_name, config_name)

    def init_detector(self, weights_name: str, config_name: str) -> nn.Module:
        # Check for MPS availability (specific to macOS with Apple Silicon)
        if torch.backends.mps.is_available():
            print("Using MPS (Apple Metal Performance Shaders)")
            self.device = torch.device("mps")
        # Check for CUDA availability
        elif isinstance(self.gpu, int) and self.gpu >= 0 and torch.cuda.is_available():
            print("Using CUDA")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
            self.device = torch.device("cuda:0")
        else:
            print("Using CPU")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self.device = torch.device("cpu")

        state_dict = torch.load(weights_name)
        detector = Model(cfg=config_name)
        detector.load_state_dict(state_dict)
        return detector.to(self.device).float().eval()

    def _preprocess(self, imgs: list[np.ndarray]) -> torch.Tensor:
        """
        Preprocessing image before passing through the network. Resize and conversion to torch tensor.
        """
        pp_imgs = []
        for img in imgs:
            h0, w0 = img.shape[:2]  # orig hw
            if self.target_size:
                r = self.target_size / min(h0, w0)  # resize image to img_size
                if r < 1:
                    img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)

            imgsz = check_img_size(max(img.shape[:2]), s=int(self.detector.stride.max()))  # check img_size
            img = letterbox(img, new_shape=imgsz)[0]
            pp_imgs.append(img)
        pp_imgs_n = np.array(pp_imgs).transpose(0, 3, 1, 2)

        return torch.from_numpy(pp_imgs_n).to(self.device).float() / 255

    def _postprocess(
        self,
        imgs: list[np.ndarray],
        origimgs: list[np.ndarray],
        pred: torch.Tensor,
        conf_thres: float,
        iou_thres: float,
    ) -> tuple[list[BoxType], list[KeypointType], list[float]]:
        """
        Postprocessing of raw pytorch model output.
        Returns:
            bboxes: list of arrays with 4 coordinates of bounding boxes with format x1,y1,x2,y2.
            points: list of arrays with coordinates of 5 facial keypoints (eyes, nose, lips corners).
            scores: list of objectness confidence scores for each detection.
        """
        bboxes: list = [[] for _ in range(len(origimgs))]
        landmarks: list = [[] for _ in range(len(origimgs))]
        scores: list = [[] for _ in range(len(origimgs))]  # Initialize list for scores

        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        for image_id, origimg in enumerate(origimgs):
            img_shape = origimg.shape
            image_height, image_width = img_shape[:2]
            gn = torch.tensor(img_shape)[[1, 0, 1, 0]]  # normalization gain whwh
            gn_lks = torch.tensor(img_shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain landmarks
            det = pred[image_id].cpu()
            scale_coords(imgs[image_id].shape[1:], det[:, :4], img_shape).round()
            scale_coords_landmarks(imgs[image_id].shape[1:], det[:, 5:15], img_shape).round()

            for j in range(det.size()[0]):
                score = det[j, 4].item()  # Objectness confidence score
                box = (det[j, :4].view(1, 4) / gn).view(-1).tolist()
                box = list(
                    map(
                        int,
                        [box[0] * image_width, box[1] * image_height, box[2] * image_width, box[3] * image_height],
                    ),
                )
                if box[3] - box[1] < self.min_face:
                    continue
                lm = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                lm = list(map(int, [i * image_width if j % 2 == 0 else i * image_height for j, i in enumerate(lm)]))
                lm = [lm[i : i + 2] for i in range(0, len(lm), 2)]
                bboxes[image_id].append(box)
                landmarks[image_id].append(lm)
                scores[image_id].append(score)

        return bboxes[0], landmarks[0], scores[0]

    def predict(
        self,
        imgs: np.ndarray | list[np.ndarray],
        conf_thres: float = 0.7,
        iou_thres: float = 0.5,
    ) -> tuple[list[BoxType], list[KeypointType], list[float]]:
        """
        Get bbox coordinates and keypoints of faces on original image.
        Params:
            imgs: image or list of images to detect faces on
            conf_thres: confidence threshold for each prediction
            iou_thres: threshold for NMS (filtering of intersecting bboxes)
        Returns:
            bboxes: list of arrays with 4 coordinates of bounding boxes with format x1,y1,x2,y2.
            points: list of arrays with coordinates of 5 facial keypoints (eyes, nose, lips corners).
        """
        # Pass input images through face detector
        images = imgs if isinstance(imgs, list) else [imgs]

        origimgs = copy.deepcopy(images)

        images = self._preprocess(images)

        with torch.inference_mode():  # change this with torch.no_grad() for pytorch <1.8 compatibility
            pred = self.detector(images)[0]

        return self._postprocess(images, origimgs, pred, conf_thres, iou_thres)

    def __call__(
        self,
        imgs: np.ndarray | list[np.ndarray],
        conf_thres: float = 0.7,
        iou_thres: float = 0.5,
    ) -> tuple[list[BoxType], list[KeypointType], list[float]]:
        return self.predict(imgs, conf_thres, iou_thres)
