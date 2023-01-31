from torchvision.models import resnet50, ResNet50_Weights

resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)

class ResNetUNet(nn.Module, BaseModel):

    def __init__(self, num_classes, simple=False, sigmoid=False, attention=False):

        super().__init__()

        self.activations = [None]
        resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.resnet_backbone = nn.Sequential(*(list(resnet_model.children())[0:7]))
        for param in self.resnet_backbone.parameters():
            param.requires_grad = False

        filters = [512, 256, 64, 64]
        self.decoder = Decoder(1024, filters, num_classes, simple, sigmoid)
        self.attention = attention
        if attention:
            self.dual_attention = DualAttention(1024)
            self.dual_attention_2 = DualAttention(filters[0])
    
    def getActivations(self):
        def hook(model, input, output):
            self.activations.append(output)
        return hook
    
    def forward(self, input):

        self.activations = [None]

        hr1 = self.resnet_backbone[2].register_forward_hook(self.getActivations())
        hr2 = self.resnet_backbone[4][2].register_forward_hook(self.getActivations())
        hr3 = self.resnet_backbone[5][-1].register_forward_hook(self.getActivations())

        resnet_output = self.resnet_backbone(input)
        
        if self.attention:
            resnet_output = self.dual_attention(resnet_output)
            self.activations[-1] = self.dual_attention_2(self.activations[-1])

        final_output = self.decoder(resnet_output, self.activations[::-1])

        hr1.remove()
        hr2.remove()
        hr3.remove()

        return final_output