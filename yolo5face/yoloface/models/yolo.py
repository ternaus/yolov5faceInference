from copy import deepcopy
from pathlib import Path

import torch
import yaml
from torch import nn

from yolo5face.yoloface.models.common import (
    C3,
    SPP,
    Bottleneck,
    BottleneckCSP,
    Concat,
    Conv,
    Focus,
    ShuffleV2Block,
    StemBlock,
    dwconv,
)
from yolo5face.yoloface.utils.general import make_divisible


class Detect(nn.Module):
    stride: torch.Tensor | None = None  # strides computed during build

    def __init__(self, nc: int = 80, anchors: list[tuple[float, ...]] = (), ch: list[int] = ()):  # type: ignore[assignment]
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5 + 10  # number of outputs per anchor

        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]] | list[torch.Tensor]:
        z = []  # inference output

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                stride_i = self.stride[i]  # type: ignore[index]
                device = x[i].device
                grid_i = self.grid[i].to(device)
                anchor_grid_i = self.anchor_grid[i]

                y = torch.full_like(x[i], 0)
                y[..., [0, 1, 2, 3, 4, 15]] = x[i][..., [0, 1, 2, 3, 4, 15]].sigmoid()
                y[..., 5:15] = x[i][..., 5:15]

                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + grid_i) * stride_i  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid_i  # wh

                y[..., 5:7] = y[..., 5:7] * anchor_grid_i + grid_i * stride_i
                y[..., 7:9] = y[..., 7:9] * anchor_grid_i + grid_i * stride_i
                y[..., 9:11] = y[..., 9:11] * anchor_grid_i + grid_i * stride_i
                y[..., 11:13] = y[..., 11:13] * anchor_grid_i + grid_i * stride_i
                y[..., 13:15] = y[..., 13:15] * anchor_grid_i + grid_i * stride_i

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx: int = 20, ny: int = 20) -> torch.Tensor:
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg: str = "yolov5s.yaml", ch: int = 3, nc: int | None = None):
        super().__init__()
        self.yaml_file = Path(cfg).name
        with Path(cfg).open(encoding="utf8") as f:
            self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc  # override yaml value

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: list[torch.Tensor] = []  # outputs
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x


def parse_model(d: dict, ch: list[int]) -> tuple[nn.Sequential, list[int]]:  # model_dict, input_channels(3)
    anchors, nc, gd, gw = d["anchors"], d["nc"], d["depth_multiple"], d["width_multiple"]
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers: list = []
    save: list = []
    c2 = ch[-1]

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [
            Conv,
            Bottleneck,
            SPP,
            dwconv,
            Focus,
            BottleneckCSP,
            C3,
            ShuffleV2Block,
            StemBlock,
        ]:
            c1, c2 = ch[f], args[0]

            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[-1 if x == -1 else x + 1] for x in f)
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
