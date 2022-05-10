import torch
import torch.nn as nn
import time


class vgg11(nn.Module):
    def __init__(self):
        super(vgg11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,128,3,padding=1),
            nn.Conv2d(128,128,3,padding=1),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128,256,3,padding=1),
            nn.Conv2d(256,256,3,padding=1),
            nn.MaxPool2d(2,2),

            nn.Conv2d(256,512,3,padding=1),
            nn.Conv2d(512,512,3,padding=1),
            nn.MaxPool2d(2,2),

            nn.Conv2d(512,1024,3),
            nn.MaxPool2d(2,2)
        )

        self.classifier = nn.Linear(1024,10)

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)





# if __name__ == "__main__":
net = vgg11()
in_ten = torch.randn([1,3,32,32])

t1 = time.time()
out = net(in_ten)
t2 = time.time()

print(t2-t1)
