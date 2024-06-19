# pyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Multi Layer Perceptron
class MLP_CONV(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.conv = nn.Conv1d(self.input_size, self.output_size, 1)
        self.bn = nn.BatchNorm1d(self.output_size)

    def forward(self, input):
        return F.relu(self.bn(self.conv(input)))

# Fully Connected with Batch Normalization
class FC_BN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lin = nn.Linear(self.input_size, self.output_size)
        self.bn = nn.BatchNorm1d(self.output_size)

    def forward(self, input):
        return F.relu(self.bn(self.lin(input)))

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.conv_layers = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.conv_layers.append(MLP_CONV(last_channel, out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        B, C, N = xyz.shape
        S = self.npoint
        new_xyz, new_points = self.sample_and_group(xyz, points)
        for conv in self.conv_layers:
            new_points = conv(new_points)
        new_points = torch.max(new_points, 2)[0]
        return new_xyz, new_points

    def sample_and_group(self, xyz, points):
        # Implementar agrupamiento de puntos aquí
        pass

class PointNet2(nn.Module):
    def __init__(self):
        super(PointNet2, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024])

    def forward(self, xyz):
        B, N, C = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz.permute(0, 2, 1), None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        return l3_points

class PointNet2Seg(nn.Module):
    def __init__(self, classes=3):
        super(PointNet2Seg, self).__init__()
        self.pointnet2 = PointNet2()
        self.mlp1 = MLP_CONV(1024, 512)
        self.mlp2 = MLP_CONV(512, 256)
        self.mlp3 = MLP_CONV(256, 128)
        self.conv = nn.Conv1d(128, classes, 1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        inputs = self.pointnet2(input)
        x = self.mlp1(inputs)
        x = self.mlp2(x)
        x = self.mlp3(x)
        output = self.conv(x)
        return self.logsoftmax(output)
