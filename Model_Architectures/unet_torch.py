import torch
import torchvision

from torch import nn
#from torchsummary import summary

class EncoderBlock(nn.Module):

    def __init__(self, d_in, d_out):

        super(EncoderBlock, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(d_in, d_out, 3, 1, "same"),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.conv_2 = nn.Sequential(
            nn.Conv2d(d_out, d_out, 3, 1, "same"),
            nn.ReLU()
        )
    
    def forward(self, inputs):

        a = self.conv_1(inputs)
        a = self.conv_2(a)
        x = self.pool(a)
        x = self.conv_2(x)

        return x, a


class LastEncoder(nn.Module):

    def __init__(self, d_in, d_out):

        super(LastEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(d_in, d_out, 3, 1, "same"),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(d_out, d_out, 3, 1, "same"),
            nn.ReLU()
        )

    def forward(self, inputs):

        x = self.conv2(self.conv1(inputs))

        return x

class FullEncoder(nn.Module):

    def __init__(self, d_in, filters):

        super(FullEncoder, self).__init__()

        self.encoder_blocks = []
        for f in filters[:-1]:

            encoder = EncoderBlock(d_in, f)
            self.encoder_blocks.append(encoder)
            d_in = f
        
        self.last_encoder = LastEncoder(f, filters[-1])


    def forward(self, inputs):

        activations = []
        x = inputs
        for eb in self.encoder_blocks:
            x, a = eb(x)
            activations.append(a)
        
        x = self.last_encoder(x)

        return x, activations

class UNet(nn.Module):

    def __init__(self, d_in, num_classes, filters):

        super(UNet, self).__init__()
        self.encoder = FullEncoder(d_in, filters[:-1])

        self.decoder = Decoder(d_in, filters[:-1][::-1], num_classes)
    
    def forward(self,inputs):

        x, activations = self.encoder(inputs)

        o = self.decoder(x, activations[::-1])