import torch
import torchvision

from torch import nn

class DecoderBlock(nn.Module):

    def __init__(self, d_in, d_out):

        super(DecoderBlock, self).__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(d_in, d_out, 2, 2, padding="same"),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(d_out, d_out, 3, 1, padding="same"),
            nn.ReLU()
        )

    def forward(self, inp, a):

        x = self.upconv(inp)
        if a is not None:
            x = torch.cat([a, x], axis=-1)
        
        x = self.conv(self.conv(x))
    
        return x


class Decoder(nn.Module):

    def __init__(self, d_in, filters, num_classes):

        super(Decoder, self).__init__()

        self.decoder_blocks = []

        for f in filters:

            self.db = DecoderBlock(d_in, f)
            self.decoder_blocks.append(self.db)
            d_in = f
        
        self.output = nn.Conv2d(f, num_classes, 1, 1)
    
    def forward(self, inputs, activations):

        x = inputs
        for db, a in zip(self.decoder_blocks, activations):

            x = db(x, a)
        
        output = self.output(x)

        return output
        
