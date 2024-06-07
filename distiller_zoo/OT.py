import torch
from torch import nn
import torch.nn.functional as F
import ot
import time


class OTLoss(torch.nn.Module):

    def __init__(self, opt):
        super(OTLoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.method = opt.ot_method
        self.embed_type = opt.ot_embed
        self.ot_gamma = opt.ot_gamma
        self.ot_eps = opt.ot_eps
        self.ot_iter = opt.ot_iter
        self.device = opt.device

    def forward(self, f_t, f_s):
        # Concatenate teacher and student embeddings
        if self.embed_type == 'use':
            f_es = self.embed_s(f_s)
            f_et = self.embed_t(f_t)
            X = torch.cat((f_et, f_es), 0).to(self.device)
        else:
            X = torch.cat((f_t, f_s), 0).to(self.device)

        if self.method == 'pcc':
            C = PCC(X)
        elif self.method == 'cos':
            C = cosine_similarity(X)
        elif self.method == 'edu':
            C = euclidean_dist(X)
        else:
            raise ValueError("Invalid method specified.")

        n = C.shape[0] // 2
        Nst = C[0:n, n:]  # Node matrix

        M = 1 - Nst
        # M = minmax_normalize(M)
        # M = zscore_normalize(M)
        P = sinkhorn(M.unsqueeze(0), gamma=self.ot_gamma, eps=self.ot_eps, maxiters=self.ot_iter)
        P = P.squeeze(0)
        P = column_normalize(P)

        loss_ot = torch.norm((P - torch.eye(P.shape[1], device=self.device)), 2)
        # loss_ot = -torch.log(torch.diag(P)).mean()

        return loss_ot, P, M


def sinkhorn(M, r=None, c=None, gamma=1.0, eps=1.0e-6, maxiters=20, logspace=False):  # maxiters=1000

    B, H, W = M.shape
    assert r is None or r.shape == (B, H) or r.shape == (1, H)
    assert c is None or c.shape == (B, W) or c.shape == (1, W)
    assert not logspace or torch.all(M > 0.0)

    r = 1.0 / H if r is None else r.unsqueeze(dim=2)
    c = 1.0 / W if c is None else c.unsqueeze(dim=1)

    if logspace:
        P = torch.pow(M, gamma)
    else:
        P = torch.exp(-1.0 * gamma * (M - torch.amin(M, 2, keepdim=True)))  # 对输入的M进行gamma倍的指数运算，同时进行数值平移以避免数值不稳定性

    for i in range(maxiters):
        alpha = torch.sum(P, 2)
        # Perform division first for numerical stability
        P = P / alpha.view(B, H, 1) * r

        beta = torch.sum(P, 1)
        if torch.max(torch.abs(beta - c)) <= eps:
            break
        P = P / beta.view(B, 1, W) * c

    return P


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x

class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def PCC(m):
    """Compute the Pearson’s correlation coefficients."""
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)  # 减均值
    mt = m.t()
    # 计算协方差矩阵
    c = fact * m.matmul(mt).squeeze()
    d = torch.diag(c, 0)  # 提取对角线元素
    std = torch.sqrt(d)  # 计算标准差
    c /= std[:, None]  # 对每行除以标准差
    c /= std[None, :]  # 对每列除以标准差
    return c


def cosine_similarity(m):
    """Compute the cosine similarity matrix."""
    m_norm = F.normalize(m, p=2, dim=1)  # 对输入进行L2范数归一化
    return torch.matmul(m_norm, m_norm.T)


def euclidean_dist(m):
    """Compute the Euclidean distance matrix."""
    n = m.size(0)
    m = m.view(n, -1)
    dist = torch.cdist(m, m, p=2.0)  # 计算欧氏距离
    return dist


def minmax_normalize(C):
    max_val = C.max()
    min_val = C.min()
    return (C - min_val) / (max_val - min_val)


def zscore_normalize(C):
    mean = C.mean()
    std = C.std()
    return (C - mean) / std


def row_normalize(P):
    row_sums = P.sum(dim=1, keepdim=True)
    return P / row_sums


def column_normalize(P):
    col_sums = P.sum(dim=0, keepdim=True)
    return P / col_sums


def doubly_normalize(P):
    P = row_normalize(P)
    P = column_normalize(P)
    return P
