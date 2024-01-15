from TinyGraph.Graph import MicroNode, DepTensor


class DepModule:
    def __init__(self):
        pass

    def forward(self, *args, **kwargs):
        pass


class DepConv2d(DepModule):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1),
                 padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def forward(self, input_tensor: DepTensor):
        pass




class TransferNode(MicroNode):
    pass


class ConvComputeNode(MicroNode):
    pass
