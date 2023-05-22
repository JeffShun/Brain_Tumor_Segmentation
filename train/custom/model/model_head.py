import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Model_Head(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_class: int
    ):
        super(Model_Head, self).__init__()
        # TODO: 定制Head模型
        self.conv1 = nn.Conv3d(in_channels, 1, 1)
        self.conv2 = nn.Conv3d(in_channels, num_class, 1)
        self.sp_wt_loss = SamplingBCELoss(neg_ratio=4, min_sampling=2000)
        self.sp_tc_loss = SamplingBCELoss(neg_ratio=6, min_sampling=1000)
        self.sp_et_loss = SamplingBCELoss(neg_ratio=8, min_sampling=500)
        

    def forward(self, inputs):
        # TODO: 定制forward网络
        input1, input2 = inputs
        out1 = self.conv1(input1)
        out2 = self.conv2(input2)
        return out1, out2

    def loss(self, inputs, targets):
        input1, input2 = inputs
        target1 = targets[:,0]
        weight1 = torch.ones_like(target1,dtype=target1.dtype,device=target1.device)
        target2 = targets[:,1]
        weight2 = target1
        target3 = targets[:,2]
        weight3 = target2
        sp_wt_loss = self.sp_wt_loss(input2[:,0],target1,weight1)
        sp_wt_aux_loss = self.sp_wt_loss(input1[:,0],target1,weight1)
        sp_tc_loss = self.sp_tc_loss(input2[:,1],target2,weight2)
        sp_et_loss = self.sp_et_loss(input2[:,2],target3,weight3)
        return {"sp_wt_aux_loss": sp_wt_aux_loss, "sp_wt_loss": sp_wt_loss , "sp_tc_loss": sp_tc_loss, "sp_et_loss": sp_et_loss}

class SamplingBCELoss(nn.Module):
    def __init__(self, neg_ratio=8, min_sampling=1000):
        super(SamplingBCELoss, self).__init__()
        self.neg_ratio = neg_ratio
        self.min_sampling = min_sampling
        self.bceloss = torch.nn.BCEWithLogitsLoss(reduce=False)
    
    def forward(self, inputs, targets, weight):
        N = targets.shape[0]
        inputs = inputs.view(N, -1)
        targets = targets.view(N, -1)
        weight = weight.view(N, -1)
        loss_all = self.bceloss(inputs, targets)
        pos_weight = targets

        neg_p = weight * (1-targets)
        loss_neg = loss_all * neg_p
        # softmax_func with weight
        exp_inputs = torch.exp(loss_neg)
        exp_inputs = exp_inputs * neg_p
        exp_sum = torch.sum(exp_inputs, 1, keepdim=True)
        loss_neg_normed = exp_inputs / (exp_sum+1e-12)

        n_pos = torch.sum(targets, 1, keepdim=True)
        sampling_prob = torch.max(self.neg_ratio * n_pos, torch.zeros_like(n_pos)+self.min_sampling) * loss_neg_normed
        random_map = torch.rand_like(sampling_prob)
        neg_weight = (random_map < sampling_prob).int()
        weight = neg_weight + pos_weight
        # print(torch.sum(neg_weight*neg_p,1)/torch.sum(pos_weight,1))
        loss = (loss_all * weight).sum()/(weight.sum())
        return loss

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