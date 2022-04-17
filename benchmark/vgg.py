from  collections import OrderedDict

from TinyNet.wrapper.conv import Conv
from TinyNet.wrapper.linear import Linear
from TinyNet.wrapper.pooling import MaxPooling
from TinyNet.wrapper.base import module

class Vgg8:
    def __init__(self):
        super(Vgg8, self).__init__()
        self.conv1 = Conv(in_channels=3,out_channels=128,kernel_size=3,padding=1,activation_func='relu')
        self.conv2 = Conv(in_channels=128,out_channels=128,kernel_size=3,padding=1,activation_func='relu')
        self.max_pool1= MaxPooling(kernel_size=2,stride=2)
        self.conv3 = Conv(in_channels=128,out_channels=256,kernel_size=3,padding=1,activation_func='relu')
        self.conv4 = Conv(in_channels=256,out_channels=256,kernel_size=3,padding=1,activation_func='relu')
        self.max_pool2 = MaxPooling(kernel_size=2,stride=2)
        self.conv5 = Conv(in_channels=256,out_channels=512,kernel_size=3,padding=1,activation_func='relu')
        self.conv6 = Conv(in_channels=512,out_channels=512,kernel_size=3,padding=1,activation_func='relu')
        self.max_pool3 = MaxPooling(kernel_size=2,stride=2)
        self.conv7 = Conv(in_channels=512,out_channels=1024,kernel_size=3,padding=0,activation_func='relu')
        self.max_pool4 = MaxPooling(kernel_size=2,stride=2)
        self.linear = Linear(in_features=1024,out_features=10)

        self.module_list = [
            self.conv1,self.conv2,self.max_pool1,
            self.conv3,self.conv4,self.max_pool2,
            self.conv5,self.conv6,self.max_pool3,
            self.conv7,self.max_pool3,self.linear
        ]


    def forward(self,torch_tensor,pim_tensor):
        ret = (torch_tensor,pim_tensor)

        for layer in self.module_list:
            ret = layer(*ret)







