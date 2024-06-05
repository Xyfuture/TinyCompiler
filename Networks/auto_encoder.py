from typing import Optional, List

from TinyGraph.DSL import DepTensor
from TinyGraph.Module import DepModule
from TinyGraph.Module import *
from utils import create_input_tensor


class AutoEncoder(DepModule):
    def __init__(self, in_features: int = 1024):
        super().__init__()

        self.in_features = in_features

        self.encoder = DepSequential(
            DepLinear(self.in_features, 512),
            DepReLU(),
            DepLinear(512, 512),
            DepReLU(),
            DepLinear(512, 128),
            DepReLU(),
            DepLinear(128, 128),
            DepReLU(),
        )

        self.decoder = DepSequential(
            DepLinear(128, 128),
            DepReLU(),
            DepLinear(128, 512),
            DepReLU(),
            DepLinear(512, 512),
            DepReLU(),
            DepLinear(512, self.in_features),
            DepReLU(),
        )

    def forward(self, x: DepTensor):
        out = self.encoder(x)
        out = self.decoder(out)

        return out


def get_autoencoder():
    return AutoEncoder(), create_input_tensor((1,), 1024)
