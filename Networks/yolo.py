from typing import Optional, List

from TinyGraph.DSL import DepTensor
from TinyGraph.Module import DepModule
from TinyGraph.Module import *
from utils import create_input_tensor


class YoloVgg(DepModule):
    def __init__(self, num_class: int = 10):
        # input image 128x128x3
        # output 4x4x20 (2x5+10)
        # vgg like network

        super().__init__()
        self.num_class = num_class

        self.features = DepSequential(
            DepConv2d(3, 64, kernel_size=(7, 7), stride=(4, 4), padding=3),
            DepReLU(),
            DepConv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            DepReLU(),
            DepMaxpool2d(kernel_size=(2, 2), stride=(2, 2)),
            DepConv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            DepReLU(),
            DepConv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            DepReLU(),
            DepConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            DepReLU(),
            DepMaxpool2d(kernel_size=(2, 2), stride=(2, 2)),
            DepConv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            DepReLU(),
            DepConv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            DepReLU(),
            DepConv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1),
            DepMaxpool2d(kernel_size=(2, 2), stride=(2, 2)),
        )  # (4,4,512)

        self.detection = DepSequential(
            DepLinear(8192, 1024),
            DepReLU(),
            DepLinear(1024, 4 * 4 * (10 + num_class))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.detection(x)
        return x


def get_yolovgg():
    return YoloVgg(),create_input_tensor((128,128),reduce_dim_size=3)