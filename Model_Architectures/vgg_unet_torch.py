from torchvision.models import vgg19, VGG19_Weights


class VGGUNet(nn.Module, BaseModel):

    def __init__(self, num_classes, simple=False, sigmoid=False, attention=False, attention_indices=[]):

        super().__init__()
        
        vgg19_m = vgg19(weights=VGG19_Weights.DEFAULT)
        self.vgg = nn.Sequential(*(list(vgg19_m.children())[0][:-1]))
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.vgg = self.vgg.to(device)
        self.activations = []

        self.filters = [512, 256, 128, 64]
        self.decoder = Decoder(self.filters[0], self.filters, num_classes, simple=simple, sigmoid=sigmoid)

        self.attention = attention
        self.attention_indices = attention_indices
    
    def getActivations(self):
        def hook(model, input, output):
            self.activations.append(output)
        return hook
    
    def forward(self, input):

        self.activations = []

        h1 = self.vgg[3].register_forward_hook(self.getActivations())
        h2 = self.vgg[8].register_forward_hook(self.getActivations())
        h3 = self.vgg[17].register_forward_hook(self.getActivations())
        h4 = self.vgg[26].register_forward_hook(self.getActivations())

        vgg_output = self.vgg(input)

        if self.attention:
            in_channels = vgg_output.shape[1]
            vgg_output = DualAttention(in_channels)(vgg_output)

            for ix in self.attention_indices:
                self.activations[ix] = DualAttention(self.filters[ix])(self.activations[ix])

        final_output = self.decoder(vgg_output, self.activations[::-1])

        h1.remove()
        h2.remove()
        h3.remove()
        h4.remove()

        return final_output