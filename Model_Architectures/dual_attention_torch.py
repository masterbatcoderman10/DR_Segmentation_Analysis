import torch
from torch import nn

class SpatialAttention(nn.Module):

    def __init__(self, in_channels):

        super(SpatialAttention, self).__init__()

        self.C = in_channels

        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.conv1 = nn.Conv2d(in_channels=self.C, out_channels=self.C, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=self.C, out_channels=self.C, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=self.C, out_channels=self.C, kernel_size=1, stride=1)

    
    def forward(self, x):
        

        H = x.shape[2]
        W = x.shape[3]

        N = H * W

        a = x
        b = self.conv1(x)
        c = self.conv2(x)
        d = self.conv3(x)

        b = b.view(-1, self.C, N)
        c = c.view(-1, self.C, N)
        d = d.view(-1, self.C, N)

        c = torch.bmm(c.transpose(1, 2), b)
        S = nn.Softmax(dim=1)(c)
        S = S.transpose(1, 2)

        d = self.alpha * torch.bmm(d, S)
        d = d.view(-1, self.C, H, W)
        E = a + d

        return E


class ChannelAttention(nn.Module):

    def __init__(self, in_channels):

        super(ChannelAttention, self).__init__()
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.C = in_channels
    
    def forward(self, x):

        a1=a2=a3=a4 = x
        H = x.shape[2]
        W = x.shape[3]
        N = H * W

        a2 = a2.view(-1, self.C, N)
        a3 = a3.view(-1, self.C, N)
        a4 = a4.view(-1, self.C, N)
        a4 = a4.transpose(1, 2)

        aa_T = torch.bmm(a3, a4)
        X = nn.Softmax(dim=1)(aa_T)
        X = X.transpose(1, 2)

        a2_pass = torch.bmm(X, a2) * self.beta
        a2_pass = a2_pass.view(-1, self.C, H, W)

        E = a1 + a2_pass

        return E

class DualAttention(nn.Module):

    def __init__(self, in_channels):

        self.C = in_channels

        self.conv1 = nn.Conv2d(self.C, self.C, 1)
        self.conv2 = nn.Conv2d(self.C, self.C, 1)

        self.sam = SpatialAttention(in_channels)
        self.cam = ChannelAttention(in_channels)

    def forward(self, x):

        e1 = self.sam(x)
        e2 = self.sam(x)

        e1 = self.conv1(e1)
        e2 = self.conv2(e2)

        F = e1 + e2
        return F






