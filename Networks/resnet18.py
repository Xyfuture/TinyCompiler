from typing import Optional, List

from TinyGraph.DSL import DepTensor
from TinyGraph.Module import DepModule
from TinyGraph.Module import *
from utils import create_input_tensor


def conv1x1(in_planes: int, out_planes: int, stride: Tuple[int, int] = (1, 1)):
    return DepConv2d(
        in_planes,
        out_planes,
        kernel_size=(1, 1),
        stride=stride,
        padding=1
    )


def conv3x3(in_planes: int, out_planes: int, stride: Tuple[int, int] = (1, 1), padding: int = 1):
    return DepConv2d(
        in_planes,
        out_planes,
        kernel_size=(3, 3),
        stride=stride,
        padding=padding
    )


class BasicBlock(DepModule):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: Tuple[int, int] = (1, 1),
                 downsample: Optional[DepModule] = None):
        super().__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = DepReLU()
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

        self.element_add = DepElementAdd()

    def forward(self, x: DepTensor) -> DepTensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.element_add(out, identity)
        out = self.relu(out)
        return out


class ResNet(DepModule):
    def __init__(self,
                 layers: List[int],
                 num_classes: int = 10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3, self.in_planes)  # for 32x32x3 input image
        self.relu = DepReLU()
        self.maxpool = DepMaxpool2d(kernel_size=(2, 2), stride=(2, 2))  # for 3x3

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(256, layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(512, layers[3], stride=(2, 2))

        self.linear1 = DepLinear(2048, 512)
        self.linear2 = DepLinear(512, num_classes)

    def forward(self, input_tensor: DepTensor):

        x = self.conv1(input_tensor)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

    def _make_layer(self,
                    planes: int,
                    blocks: int,
                    stride: Tuple[int, int] = (1, 1)):
        downsample = None
        if stride[0] != 1:
            downsample = conv1x1(self.in_planes, planes, stride)

        layers = [BasicBlock(self.in_planes, planes, stride, downsample, )]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.in_planes,
                    planes,
                )
            )

        return DepSequential(*layers)


def resnet18():
    return ResNet([2, 2, 2, 2], 10)


def get_resnet18():
    return resnet18(),create_input_tensor((32, 32), 3)