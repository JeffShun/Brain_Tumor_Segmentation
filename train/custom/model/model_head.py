import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model_Head(nn.Module):

    def __init__(
        self,
        in_channels,
        num_class
    ):
        super(Model_Head, self).__init__()
        # TODO: 定制Head模型
        self.conv = nn.Conv3d(in_channels, num_class, 1)
        self.diceloss = DiceLoss()
        self.ssloss = Sensitivity_SpecificityLoss(alpha_sen=0.4)

    def forward(self, inputs):
        # TODO: 定制forward网络
        out = self.conv(inputs)
        return out

    def loss(self, inputs, targets):
        wt_diceloss = self.diceloss(inputs[:,0],targets[:,0])
        tc_diceloss = self.diceloss(inputs[:,1],targets[:,1])
        et_diceloss = self.diceloss(inputs[:,2],targets[:,2])

        wt_ssloss = self.ssloss(inputs[:,0],targets[:,0])
        tc_ssloss = self.ssloss(inputs[:,1],targets[:,1])
        et_ssloss = self.ssloss(inputs[:,2],targets[:,2])
        return {"wt_diceloss": wt_diceloss , "tc_diceloss": tc_diceloss, "et_diceloss": et_diceloss,
                "wt_ssloss": wt_ssloss,"tc_ssloss": tc_ssloss, "et_ssloss": et_ssloss}


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, inputs, targets):
        """soft dice loss"""
        inputs = torch.sigmoid(inputs)
        eps = 1e-7
        iflat = torch.flatten(inputs,start_dim=1,end_dim=-1)
        tflat = torch.flatten(targets,start_dim=1,end_dim=-1)
        intersection = (iflat * tflat).sum(1)
        loss =  1 - 2. * intersection / ((iflat ** 2).sum(1) + (tflat ** 2).sum(1) + eps)
        return loss.mean()

class Sensitivity_SpecificityLoss(nn.Module):
    def __init__(self, alpha_sen=0.5):
        super(Sensitivity_SpecificityLoss, self).__init__()
        self.alpha_sen = alpha_sen
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        # 平滑变量
        smooth = 1e-5
        input_flat = torch.flatten(inputs,start_dim=1,end_dim=-1)
        targets_flat = torch.flatten(targets,start_dim=1,end_dim=-1)
        
        sensitivity_loss = ((input_flat - targets_flat)**2 * targets_flat).sum()/(targets_flat.sum()+smooth)
        specificity_loss = ((input_flat - targets_flat)**2 * (1-targets_flat)).sum()/((1-targets_flat).sum()+smooth)
        loss = self.alpha_sen*sensitivity_loss + (1-self.alpha_sen)*specificity_loss
        
        return loss
