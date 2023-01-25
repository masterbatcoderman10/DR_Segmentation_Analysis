import torch
import torchvision

from torch import nn

class DecoderBlock(nn.Module):

    def __init__(self, d_in, d_out):


        super().__init__()
        self.upconv = nn.ConvTranspose2d(d_in, d_out, 3, 2, padding=1)
        self.conv_1 = nn.Conv2d(d_out*2, d_out, 3, 1, padding=1)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(d_out, d_out, 3, 1, padding=1)
        

    def forward(self, inp, a):
        
        x = self.relu(self.upconv(inp))
        #x = self.upconv(inp)
        if a is not None:
            x = torch.cat([a, x], axis=1)
            x = self.relu(self.conv_1(x))
            
        x = self.relu(self.conv_2(x))

        return x


class Decoder(nn.Module):

    def __init__(self, d_in, filters, num_classes, simple=False, sigmoid=False):

        super().__init__()

        self.decoder_blocks = []

        for f in filters:
            
            if simple:
                db = SimpleDecoderBlock(d_in, f)
            else:
                db = DecoderBlock(d_in, f)

            self.decoder_blocks.append(db)
            d_in = f
        
        self.output = nn.Conv2d(f, num_classes, 1, 1)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)
        self.sig = nn.Sigmoid() if sigmoid else None
    
    def forward(self, inputs, activations):

        x = inputs
        for db, a in zip(self.decoder_blocks, activations):

            x = db(x, a)
        
        output = self.output(x)
        if self.sig is not None:
            output = self.sig(output)


        return output
        

        