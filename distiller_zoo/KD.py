from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class DistillKL_logit_stand(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self):
        super(DistillKL_logit_stand, self).__init__()

    def forward(self, y_s, y_t, temp):
        T = temp.cuda()

        KD_loss = 0
        KD_loss += nn.KLDivLoss(reduction='batchmean')(F.log_softmax(normalize(y_s) / T, dim=1),
                                                       F.softmax(normalize(y_t) / T, dim=1)) * T * T

        return KD_loss


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)