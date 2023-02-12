class CSA(nn.Module):

    def __init__(self, in_channels):
        
        super(CSA, self).__init__()

        self.C = in_channels
        self.C_by_2 = int(in_channels / 2)
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_3x3_1 = nn.Conv2d(in_channels=self.C_by_2, out_channels=self.C_by_2, kernel_size=3, stride=1, padding="same")
        self.conv_3x3_2 = nn.Conv2d(in_channels=self.C_by_2, out_channels=self.C_by_2, kernel_size=3, stride=1, padding="same")
        self.conv_3x3_3 = nn.Conv2d(in_channels=in_channels, out_channels=self.C_by_2, kernel_size=3, stride=1, padding="same")

        self.group_1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.group_2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)
        self.final_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)

    def forward(self, input):

        H = input.shape[2]
        W = input.shape[3]

        N = H * W

        F = self.conv_1(input)
        F_1, F_2 = F.split(int(self.C / 2), dim=1)

        F_1 = self.conv_3x3_1(F_1)
        F_2 = self.conv_3x3_2(F_2)
        F_2 = torch.concat([F_1, F_2], dim=1)
        F_2 = self.conv_3x3_3(F_2)

        F = torch.concat([F_1, F_2], dim=1)

        #Global average pooling
        F = nn.AdaptiveAvgPool2d((H, W))(F)

        F = self.group_1(F)
        F = self.bn(F)
        F = self.relu(F)
        F = self.group_2(F)

        F_1_s, F_2_s = F.split(int(self.C / 2), dim=1)

        F_1_s = self.softmax(F_1)
        F_2_s = self.softmax(F_2)

        F_1_final = F_1 * F_1_s
        F_2_final = F_2 * F_2_s

        F_final = torch.concat([F_1_final, F_2_final], dim=1)
        F_final = self.final_conv(F_final)

        output = F_final + input

        return output