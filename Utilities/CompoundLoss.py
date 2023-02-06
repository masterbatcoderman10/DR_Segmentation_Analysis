import torch
import torch.nn as nn

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, prediction, ground_truth):
        # apply softmax on the prediction
        prediction = torch.softmax(prediction, dim=1)

        # calculate the ground truth marginal probabilities
        #y = torch.sum(ground_truth, dim=1) / 512 * 512
        y = torch.mean(ground_truth, dim=(2,3))

        # calculate the predicted marginal probabilities
        #p = torch.sum(predictionction, dim=1) / 512 * 512
        p = torch.mean(prediction, dim=(2,3))

        # calculate L1 loss
        loss = torch.mean(torch.sum(torch.abs(y - p), dim=1))

        return loss

class KLLoss(torch.nn.Module):
    
    def __init__(self):
        
        super(KLLoss, self).__init__()
        
    def forward(self, prediction, ground_truth):
        y = torch.mean(ground_truth, dim=(2,3))
        p = torch.mean(torch.nn.functional.softmax(prediction, dim=1), dim=(2,3))
        
        loss = torch.sum(y * torch.log(y / p), dim=1)
        return torch.mean(loss)

class CompoundLoss(nn.Module):
    
    def __init__(self, l1=True, lamb=0.1):
        
        self.seg_loss = CrossEntropy4D()
        self.reg_loss = L1Loss() if l1 else KLLoss()
        self.lamb = lamb
    
    def forward(self, prediction, target):
        
        ce_loss = self.seg_loss(prediction, target)
        reg_loss = self.lamb * self.reg_loss(prediction, target)
        
        loss = ce_loss + reg_loss
        
        return loss
    