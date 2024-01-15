from TinyNet.wrapper.convpipe import Conv
from TinyNet.wrapper.linearpipe import Linear
from TinyNet.wrapper.pooling import MaxPooling


class auto_encoder:
    def __init__(self):

        self.linear1 = Linear(in_features=4096,out_features=2048)
        self.linear2 = Linear(in_features=2048,out_features=1024)
        self.linear3 = Linear(in_features=1024,out_features=1024)

        self.linear4 = Linear(in_features=1024,out_features=1024)
        self.linear5 = Linear(in_features=1024,out_features=2048)
        self.linear6 = Linear(in_features=2048,out_features=4096)

        self.stage_list = [
            [self.linear1],[self.linear2],[self.linear3],
            [self.linear4],[self.linear5],[self.linear6]
        ]