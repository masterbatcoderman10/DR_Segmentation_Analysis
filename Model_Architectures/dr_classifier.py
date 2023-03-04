class DRClassifier(nn.Module):
    
    def __init__(self, num_classes=2):
        
        effnet_b4 = efficientnet_b4(weights=EfficientNet_B4_Weights)
        self.effnet_b4_backbone = nn.Sequential(*(list(effnet_b4.children())[:2]))
        self.output_layer = nn.Linear(in_features=1792, out_features=num_classes)
    
    def forward(self, input):
        
        x = self.effnet_b4_backbone(input)
        x = x.view(x.size(0), -1)
        
        output = self.output_layer(x)
        
        return output
        
        
        