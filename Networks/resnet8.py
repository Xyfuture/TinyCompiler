from typing import Optional, List


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


class ResNet8(DepModule):
    def __init__(self,
                 layers: List[int],
                 num_classes: int = 10):
        super(ResNet8, self).__init__()
        self.in_planes = 16

        self.conv1 = conv3x3(3, self.in_planes)  # for 32x32x3 input image
        self.relu = DepReLU()

        self.layer1 = self._make_layer(16, layers[0])
        self.layer2 = self._make_layer(32, layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(64, layers[2], stride=(2, 2))

        self.maxpool1 = DepMaxpool2d(kernel_size=(2, 2), stride=(2, 2))  # for 3x3
        self.maxpool2 = DepMaxpool2d(kernel_size=(4, 4), stride=(4, 4))
        self.linear = DepLinear(64, num_classes)

    def forward(self, input_tensor: DepTensor):

        x = self.conv1(input_tensor)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.maxpool2(x)

        x = self.linear(x)

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


def resnet8():
    return ResNet8([1, 1, 1,], 10)


def get_resnet8():
    return resnet8(),create_input_tensor((32, 32), 3)