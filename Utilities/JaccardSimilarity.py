class JaccardSimilarity(nn.Module):
    """This is a metric for multi-label classification"""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def forward(self, predictions, targets):

        binary_predictions = torch.where(predictions > self.threshold, 1, 0)
        #Calculate the intersection between predictions and targets, sum in the class dimension
        intersection = torch.sum(binary_predictions * targets, dim=1)
        union = torch.sum((binary_predictions | targets), dim=1)

        jaccard_score = torch.mean(intersection.float() / union.float())
        return jaccard_score