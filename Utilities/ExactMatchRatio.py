class ExactMatchRatio(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, predictions, targets):
        """
        :param predictions: tensor of shape (batch_size, num_classes) with predicted scores
        :param targets: tensor of shape (batch_size, num_classes) with target scores
        :return: exact match ratio
        """
        # binarize predictions using threshold
        binary_predictions = torch.where(predictions > self.threshold, 1, 0)
        # calculate element-wise equality between binary predictions and targets
        equality = torch.eq(binary_predictions, targets)
        # calculate row-wise sums of element-wise equality
        row_sums = torch.sum(equality, dim=1)
        # calculate exact match ratio
        exact_match_ratio = torch.mean(torch.eq(row_sums, targets.shape[1]).float())
        return exact_match_ratio
