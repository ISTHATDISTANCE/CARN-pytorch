import torch
import torch.nn as nn
import torch.nn.functional as F
import model.ops as ops

class DenseLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(nChannels)
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=3,padding=1)
        
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = torch.cat((x, out), dim=1)
        return out
        
class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, group):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(nChannels)
        self.conv = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, groups=group)
    
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return out


def DenseBlock(growthRate, nChannels, nOutChannels, nDenseLayers, group):
    layers = []
    for i in range(nDenseLayers):
        layers.append(DenseLayer(nChannels, growthRate))
        nChannels += growthRate
    layers.append(Transition(nChannels, nOutChannels, group))
    return nn.Sequential(*layers)


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        scale = kwargs.get("scale")
        multi_scale = kwargs.get("multi_scale")
        group = kwargs.get("group", 1)

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = DenseBlock(16, 64, 64, 4, group)
        self.b2 = DenseBlock(16, 64, 64, 4, group)
        self.b3 = DenseBlock(16, 64, 64, 4, group)
        # self.b4 = DenseBlock(16, 64, 64, 4)
        self.c1 = ops.BasicBlock(64*2, 64, 1, 1, 0)
        self.c2 = ops.BasicBlock(64*3, 64, 1, 1, 0)
        self.c3 = ops.BasicBlock(64*4, 64, 1, 1, 0)
        # self.c4 = ops.BasicBlock(64*5, 64, 1, 1, 0)
        self.upsample = ops.UpsampleBlock(64, scale=scale, 
                                          multi_scale=multi_scale,
                                          group=group)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)
                
    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        # b4 = self.b3(o3)
        # c4 = torch.cat([c3, b4], dim=1)
        # o4 = self.c3(c4)

        # out = self.upsample(o4, scale=scale)
        out = self.upsample(o3, scale=scale)
        out = self.exit(out)
        out = self.add_mean(out)

        return out
