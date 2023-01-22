class DiceLoss(nn.Module):
    def __init__(self, num_classes, log_loss=False):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.log_loss = log_loss

    def forward(self, prediction, target):
        prediction = torch.softmax(prediction, dim=1)
        prediction = prediction.view(prediction.size(0), self.num_classes, -1)
        target = target.view(target.size(0), self.num_classes, -1)
        intersection = (prediction * target).sum(dim=-1)
        union = prediction.sum(dim=-1) + target.sum(dim=-1)
        dice = 2 * intersection / union

        if self.log_loss:
            dice = torch.log(dice)

        return 1 - dice.mean()