import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from collections import OrderedDict

def generate_random_orthogonal_matrix(feat_in, num_classes):
    rand_mat = np.random.random(size=(feat_in, num_classes))
    orth_vec, _ = np.linalg.qr(rand_mat)
    orth_vec = torch.tensor(orth_vec).float()
    assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
        "The max irregular value is : {}".format(
            torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
    return orth_vec

class DRLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 reg_lambda=0.
                 ):
        super(DRLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda
    
    def aux_forward(self,feat,target):
        dot = torch.matmul(feat,feat.T)
        labels = target.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(labels.shape[0]).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask 
        dot1 = (dot - 1) ** 2
        dot2 = (dot1*mask).sum(1)
        return torch.mean(dot2)


    def l2_forward(
            self,
            feat,
            target,
            gt_label,
            weight=None,
            h_norm2=None,
            m_norm2=None,
            avg_factor=None,
    ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)
        #weight = (1-dot).detach()


        dot1 = ((m_norm2 * h_norm2) - dot)**2
        if weight is None:
            loss = 0.5 * torch.mean((dot1 ) / h_norm2)
        else:
            loss = 0.5 * torch.mean(weight*(dot1 ) / h_norm2)

        # dot1 = ((m_norm2 * h_norm2) - dot)
        # if weight is None:
        #     loss = torch.mean((dot1 ) / h_norm2)
        # else:
        #     loss = torch.mean(weight*(dot1 ) / h_norm2)


        #loss += 0.05*self.aux_forward(feat,gt_label)
        return loss * self.loss_weight
    
    def l1_forward(
            self,
            feat,
            target,
            gt_label,
            weight=None,
            h_norm2=None,
            m_norm2=None,
            avg_factor=None,
    ):
        assert avg_factor is None
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)
        #weight = (1-dot).detach()

        dot1 = ((m_norm2 * h_norm2) - dot)
        if weight is None:
            loss = torch.mean((dot1 ) / h_norm2)
        else:
            loss = torch.mean(weight*(dot1 ) / h_norm2)


        #loss += 0.05*self.aux_forward(feat,gt_label)
        return loss * self.loss_weight

obj_drloss = DRLoss()

class ETFHead(nn.Module):
    """Classification head for Baseline.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
    """

    def __init__(self, num_classes: int, in_channels: int, *args, **kwargs) -> None:
        super(ETFHead,self).__init__()
        assert num_classes > 0, f'num_classes={num_classes} must be a positive integer'
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss = obj_drloss

        orth_vec = generate_random_orthogonal_matrix(self.in_channels, self.num_classes)
        i_nc_nc = torch.eye(self.num_classes)
        one_nc_nc: torch.Tensor = torch.mul(torch.ones(self.num_classes, self.num_classes), (1 / self.num_classes))
        etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(self.num_classes / (self.num_classes - 1)))
        self.etf_vec = etf_vec
        

        etf_rect = torch.ones((1, num_classes), dtype=torch.float32)
        self.etf_rect = etf_rect
        self.etf_vec = self.etf_vec.cuda()

    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> Dict:
        """Forward training data."""
        x = self.pre_logits(x)
        cls_score = x @ self.etf_vec
        return cls_score
    
    def dr_loss_l2(self, x: torch.Tensor, gt_label: torch.Tensor,weight=None):
        x = self.pre_logits(x)
        target = self.etf_vec[:, gt_label].t()
        loss1 = self.loss.l2_forward(x,target,gt_label,weight)
        return loss1
    
    def dr_loss_l1(self, x: torch.Tensor, gt_label: torch.Tensor,weight=None):
        x = self.pre_logits(x)
        target = self.etf_vec[:, gt_label].t()
        loss1 = self.loss.l1_forward(x,target,gt_label,weight)
        return loss1
    
    def ce_loss(self, x: torch.Tensor, gt_label: torch.Tensor,weight=None):
        x = self.pre_logits(x)
        cls_score = x @ self.etf_vec
        ce_loss = nn.CrossEntropyLoss()
        loss1 = ce_loss(cls_score,gt_label)
        #target = self.etf_vec[:, gt_label].t()
        #loss1 = self.loss(x,target,gt_label,weight)
        return loss1
    
    def dr_loss2(self, x: torch.Tensor, gt_label: torch.Tensor,weight=None):
        x = self.pre_logits(x)
        cls_score = x @ self.etf_vec
        for i in range(cls_score.shape[0]):
            cls_score[i,gt_label[i]] = cls_score[i,gt_label[i]] * (-1)
        cls_score += cls_score + 1
        cls1 = cls_score.sum(1)
        loss1 = torch.mean(cls1)
        #target = self.etf_vec[:, gt_label].t()
        #loss1 = self.loss(x,target,gt_label,weight)
        return loss1

    def dr_loss3(self, x: torch.Tensor, gt_label: torch.Tensor,weight=None):
        x = self.pre_logits(x)
        cls_score = ((x @ self.etf_vec - 1)**2)*(-1)
        for i in range(cls_score.shape[0]):
            cls_score[i,gt_label[i]] = cls_score[i,gt_label[i]] * (-1)
        cls_score = cls_score + 1
        cls1 = cls_score.sum(1)
        loss1 = torch.mean(cls1)
        #target = self.etf_vec[:, gt_label].t()
        #loss1 = self.loss(x,target,gt_label,weight)
        return loss1

    def dr_constrative(self, x: torch.Tensor, gt_label: torch.Tensor,weight=None):
        x = self.pre_logits(x)
        cls_score = 0.5*((1 - x @ x.t())**2)
        #cls_score = 1 - x @ x.t()
        #neg_score = x @ x.t()
        labels = gt_label.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        logits_mask=torch.scatter(torch.ones_like(mask),1,torch.arange(mask.shape[0]).view(-1,1).to(gt_label.device),0)
        pos_mask = mask * logits_mask
        pos_mat = pos_mask * cls_score
        pos_sum = pos_mat.mean()
        neg_mask = 1 - mask
        neg_mat = neg_mask * cls_score
        neg_sum = neg_mat.mean()
        loss = pos_sum / (neg_sum + pos_sum)
        return loss
        







class MLPFFNNeck(nn.Module):
    def __init__(self, in_channels=512, out_channels=512):
        super().__init__()
        #self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.ln1 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_channels, in_channels * 2)),
            ('ln',nn.LayerNorm(in_channels * 2)),
            #('ln', build_norm_layer(dict(type='LN'), in_channels * 2)[1]),
            ('relu', nn.LeakyReLU(0.1))
        ]))
        self.ln2 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_channels * 2, in_channels * 2)),
            ('ln',nn.LayerNorm(in_channels * 2)),
            #('ln', build_norm_layer(dict(type='LN'), in_channels * 2)[1]),
            ('relu', nn.LeakyReLU(0.1))
        ]))
        self.ln3 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(in_channels * 2, out_channels, bias=False)),
        ]))
        if in_channels == out_channels:
            # self.ffn = nn.Identity()
            self.ffn = nn.Sequential(OrderedDict([
                ('proj', nn.Linear(in_channels, out_channels, bias=False)),
            ]))
        else:
            self.ffn = nn.Sequential(OrderedDict([
                ('proj', nn.Linear(in_channels, out_channels, bias=False)),
            ]))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs = inputs[-1]
        #x = self.avg(inputs)
        x = inputs
        x = x.view(inputs.size(0), -1)
        identity = x
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = x + self.ffn(identity)
        return x
    
