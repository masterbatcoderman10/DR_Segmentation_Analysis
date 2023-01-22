class DiceCE(nn.Module):
    def __init__(self, num_classes, dice_weight=1, ce_weight=1, log_dice=False):
        super(DiceCE, self).__init__()
        self.dice_loss = DiceLoss(num_classes, log_dice)
        self.ce_loss = CrossEntropy4D()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
    def forward(self, prediction, target):
        dice_loss = self.dice_loss(prediction, target)
        ce_loss = self.ce_loss(prediction, target)
        return dice_weight*dice_loss + ce_weight*ce_loss