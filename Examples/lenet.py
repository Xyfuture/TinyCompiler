from TinyNet.wrapper.convpipe import Conv
from TinyNet.wrapper.linearpipe import Linear
from TinyNet.wrapper.pooling import MaxPooling

class lenet:
    def __init__(self):

        self.conv1 = Conv(in_channels=3,out_channels=6,kernel_size=5,activation_func='relu')
        self.max_pool1 = MaxPooling(kernel_size=2,stride=2)

        self.conv2 = Conv(in_channels=6,out_channels=16,kernel_size=5,activation_func='relu')
        self.max_pool2 = MaxPooling(kernel_size=2,stride=2)

        self.conv3 = Conv(in_channels=16,out_channels=120,kernel_size=5,activation_func='relu')

        self.linear1 = Linear(in_features=120,out_features=84,activation_func='relu')

        self.linear2 = Linear(in_features=84,out_features=10)

        self.stage_list = [
            [self.conv1,self.max_pool1],[self.conv2,self.max_pool2],
            [self.conv3],[self.linear1],[self.linear2]
        ]