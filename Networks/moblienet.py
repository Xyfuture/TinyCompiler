from typing import Optional, List

from TinyGraph.DSL import DepTensor
from TinyGraph.Module import DepModule
from TinyGraph.Module import *
from utils import create_input_tensor

class DW(DepModule):
    def __init__(self, in_channels, out_channels, stride=(1,1)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dw_block = DepSequential(
            # 3*3 dw conv
            # DepConv2d(self.in_channels, self.in_channels, kernel_size=(3,3), stride=self.stride, padding=1, ),
            # DepReLU(),
            # 1*1 pw conv
            DepConv2d(self.in_channels, self.out_channels, kernel_size=(1,1), stride=stride),
            DepReLU(),
        )

    def forward(self, x):
        out = self.dw_block(x)
        return out


class MobileNet(DepModule):
    def __init__(self, num_classes=2):
        super().__init__()

        self.conv1 = DepSequential(
            DepConv2d(3, 8, kernel_size=(3,3), stride=(2,2), padding=1),
            DepReLU()
        )
        self.conv_dw2 = DW(8, 16, stride=(1,1))
        self.conv_dw3 = DW(16, 32, stride=(2,2))
        self.conv_dw4 = DW(32, 32, stride=(1,1))
        self.conv_dw5 = DW(32, 64, stride=(2,2))
        self.conv_dw6 = DW(64, 64, stride=(1,1))
        self.conv_dw7 = DW(64, 128, stride=(2,2))
        self.conv_dw8 = DW(128, 128, stride=(1,1))
        self.conv_dw9 = DW(128, 128, stride=(1,1))
        self.conv_dw10 = DW(128, 128, stride=(1,1))
        self.conv_dw11 = DW(128, 128, stride=(1,1))
        self.conv_dw12 = DW(128, 128, stride=(1,1))
        self.conv_dw13 = DW(128, 256, stride=(2,2))
        self.conv_dw14 = DW(256, 256, stride=(1,1))
        self.gap = DepMaxpool2d((3,3),(3,3))
        self.fc = DepLinear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv_dw2(out)
        out = self.conv_dw3(out)
        out = self.conv_dw4(out)
        out = self.conv_dw5(out)
        out = self.conv_dw6(out)
        out = self.conv_dw7(out)
        out = self.conv_dw8(out)
        out = self.conv_dw9(out)
        out = self.conv_dw10(out)
        out = self.conv_dw11(out)
        out = self.conv_dw12(out)
        out = self.conv_dw13(out)
        out = self.conv_dw14(out)
        # out = self.gap(out)
        out = self.fc(out)
        return out


def get_mobilenet(num_classes=2):
    return MobileNet(),create_input_tensor((32,32),3)


