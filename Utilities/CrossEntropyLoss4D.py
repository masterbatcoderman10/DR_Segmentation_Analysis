class CrossEntropy4D(nn.Module):
    def __init__(self):
        super(CrossEntropy4D, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        # Reshape predictions and targets to (batch_size * height * width, num_classes)
        predictions_2d = prediction.permute(0, 2, 3, 1).reshape(-1, prediction.size()[1])
        targets_2d = target.permute(0, 2, 3, 1).reshape(-1, target.size()[1])
        loss = self.cross_entropy_loss(predictions_2d, targets_2d.argmax(1))
        return loss