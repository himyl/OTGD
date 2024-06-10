import torch
from torch import nn
import math
import torch.nn.functional as F
import dgl
import dgl.backend as B
from dgl import DGLGraph
from scipy import sparse
from dgl.nn.pytorch import TAGConv

eps = 1e-7
knn = 8

""" HKD embed method add OT method """


class EMDOTLoss(nn.Module):
    def __init__(self, opt):  #, method='pcc', ot_gamma=1, ot_eps=1e-6, ot_iter=20, device='cuda'):
        super(EMDOTLoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.contrast = NCEAverage(opt.feat_dim, opt.n_data, opt.nce_k).to(opt.device)
        self.criterion = NCESoftmaxLoss()
        self.feat_size = opt.feat_dim
        self.hkd_weight = opt.hkd_weight
        self.ot_weight = opt.ot_weight
        self.method = opt.ot_method
        self.embed_type = opt.ot_embed
        self.ot_gamma = opt.ot_gamma
        self.ot_eps = opt.ot_eps
        self.ot_iter = opt.ot_iter
        self.device = opt.device

        u = torch.tensor([i for i in range(opt.batch_size * opt.nce_k)]).cuda()
        v = torch.tensor([i for i in range(opt.batch_size * opt.nce_k)]).cuda()
        self.G_neg = dgl.graph((u, v)).to(self.device)

    def forward(self, epoch, f_s, f_t, idx, contrast_idx=None):
        batchSize = f_s.size(0)
        K = self.contrast.K
        T = 0.07

        weight_t, weight_s = self.contrast(batchSize, idx, contrast_idx)

        # graph independent part
        f_es = self.embed_s(f_s)
        f_et = self.embed_t(f_t)
        f_us, f_ut = self.contrast.get_pos(idx)
        ls_pos = torch.einsum('nc,nc->n', [f_ut, f_es]).unsqueeze(-1)
        lt_pos = torch.einsum('nc,nc->n', [f_us, f_et]).unsqueeze(-1)

        ls_neg = torch.bmm(weight_t, f_es.view(batchSize, self.feat_size, 1)).squeeze()
        lt_neg = torch.bmm(weight_s, f_et.view(batchSize, self.feat_size, 1)).squeeze()

        out_s = torch.cat([ls_pos, ls_neg], dim=1)
        out_s = torch.div(out_s, T)
        out_s = out_s.contiguous()

        out_t = torch.cat([lt_pos, lt_neg], dim=1)
        out_t = torch.div(out_t, T)
        out_t = out_t.contiguous()

        loss_e = self.criterion(out_s) + self.criterion(out_t)

        if self.embed_type == 'use':
            X = torch.cat((f_et, f_es), 0).to(self.device)
        else:
            X = torch.cat((f_t, f_s), 0).to(self.device)

        if self.method == 'pcc':
            C = PCC(X)
        elif self.method == 'cos':
            C = cosine_similarity(X)
        else:
            raise ValueError("Invalid method specified.")

        n = C.shape[0] // 2
        Nst = C[0:n, n:]  # Node matrix
        M = 1 - Nst
        # M = minmax_normalize(M)
        M = zscore_normalize(M)

        P = sinkhorn(M.unsqueeze(0), gamma=self.ot_gamma, eps=self.ot_eps, maxiters=self.ot_iter)
        P = P.squeeze(0)
        # P = row_normalize(P)
        # P = column_normalize(P)
        P = doubly_normalize(P)

        loss_ot = torch.norm((P - torch.eye(P.shape[1], device=self.device)), 2)

        self.contrast.update(f_es, f_et, idx)

        loss_tol = self.hkd_weight * loss_e + self.ot_weight * loss_ot

        return loss_tol, loss_e, loss_ot, P, M


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


def cos_distance_softmax(x):
    soft = F.softmax(x, dim=2)
    w = soft.norm(p=2, dim=2, keepdim=True)
    return 1 - soft @ B.swapaxes(soft, -1, -2) / (w @ B.swapaxes(w, -1, -2)).clamp(min=eps)


class NCEAverage(nn.Module):
    def __init__(self, inputSize, outputSize, K):
        super(NCEAverage, self).__init__()
        self.K = K
        self.momentum = 0.9
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def update(self, l, ab, y):
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(self.momentum)
            l_pos.add_(torch.mul(l, 1 - self.momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(self.momentum)
            ab_pos.add_(torch.mul(ab, 1 - self.momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)

    def get_smooth(self, l, ab, y):
        momentum = 0.75
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
        return updated_l.detach(), updated_ab.detach()

    def get_pos(self, y):
        l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
        ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
        return l_pos.detach(), ab_pos.detach()

    def forward(self, batchSize, y, idx=None):
        K = self.K

        weight_t = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_t = weight_t.view(batchSize, K, -1)

        weight_s = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_s = weight_s.view(batchSize, K, -1)

        return weight_t, weight_s


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        # label 指定每个采样的正样本 idx = 0
        label = torch.zeros([bsz], device=x.device).long()
        loss = self.criterion(x, label)
        return loss


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.conv1 = TAGConv(in_dim, hidden_dim, k=1)
        self.l2norm = Normalize(2)

    def forward(self, g):
        h = g.ndata['h']
        h = self.l2norm(self.conv1(g, h))
        return h


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
