from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

effnet_b4 = efficientnet_b4(weights=EfficientNet_B4_Weights)

class EfficientNetUNet(nn.Module):

    def __init__(self, num_classes, simple=False, sigmoid=False):

        super().__init__()
        self.activations = [None]

        effnet_b4 = efficientnet_b4(weights=EfficientNet_B4_Weights)
        self.effnet_b4_backbone = nn.Sequential(*(list(effnet_b4.children())[0]))
        for param in self.effnet_b4_backbone.parameters():
            param.requires_grad = False
        
        filters = [160, 56, 32, 48, 64]
        self.decoder = Decoder(filters[0], filters, num_classes, simple, sigmoid)
    
    def getActivations(self):
        def hook(model, input, output):
            self.activations.append(output)
        return hook
    
    def forward(self, input):

        self.activations = [None]

        e1 = self.effnet_b4_backbone[0].register_forward_hook(self.getActivations())
        e2 = self.effnet_b4_backbone[2][-1].register_forward_hook(self.getActivations())
        e3 = self.effnet_b4_backbone[3][-1].register_forward_hook(self.getActivations())
        e4 = self.effnet_b4_backbone[5][-1].register_forward_hook(self.getActivations())
        e5 = self.effnet_b4_backbone[7][-1].register_forward_hook(self.getActivations())

        _ = self.effnet_b4_backbone(input)
        effnet_output = self.activations.pop()

        final_output = self.decoder(effnet_output, self.activations[::-1])

        e1.remove()
        e2.remove()
        e3.remove()
        e4.remove()
        e5.remove()

        return final_output

        
        

        