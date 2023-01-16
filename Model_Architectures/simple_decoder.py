class SimpleDecoderBlock(nn.Module):

    def __init__(self, d_in, d_out):

        super().__init__()

        self.upconv = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_1 = nn.Conv2d(d_in, d_out, 1, 1)
        self.relu = nn.ReLU()
        self.conv_2 = nn.Conv2d(d_out*2, d_out, 3, 1, "same")
        self.conv_3 = nn.Conv2d(d_out, d_out, 3, 1, "same")

    def forward(self, inp, a):

        x = self.upconv(inp)
        x = self.relu(self.conv_1(x))

        if a is not None:
            x = torch.cat([a, x], axis=1)
            x = self.relu(self.conv_2(x))
            
        x = self.relu(self.conv_3(x))

        return x


        