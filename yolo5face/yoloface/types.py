from typing import Union

import numpy as np
import torch

BoxType = list[list[int]]
KeypointType = list[list[int]]

TensorLike = Union[np.ndarray, torch.Tensor]  # noqa: UP007

ShapeLike = Union[np.ndarray, tuple[int, int, int], tuple[int, int]]  # noqa: UP007
