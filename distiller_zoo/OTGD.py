import torch
from torch import nn
import torch.nn.functional as F
import dgl.backend as B
from dgl import DGLGraph
from scipy import sparse
from dgl.nn.pytorch import TAGConv
import matplotlib.pyplot as plt
import wandb
import numpy as np

eps = 1e-7
knn = 8

""" HKD method without InfoNCE estimator + OT loss"""


class OTGDLoss(nn.Module):
    def __init__(self, opt):
        super(OTGDLoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.gnn_s = Encoder(opt.feat_dim, opt.feat_dim)
        self.gnn_t = Encoder(opt.feat_dim, opt.feat_dim)
        self.feat_size = opt.feat_dim
        self.e_weight = opt.e_weight
        self.g_weight = opt.g_weight
        self.ge_weight = opt.ge_weight
        self.eg_weight = opt.eg_weight
        self.loss_ot = OTLoss(opt)
        self.device = opt.device

    def forward(self, epoch, f_s, l_s, f_t, l_t):
        batchSize = f_s.size(0)

        # graph independent part
        f_es = self.embed_s(f_s)
        f_et = self.embed_t(f_t)
        loss_e, P, cost = self.loss_ot(f_et, f_es)

        if batchSize < knn:
            return loss_e, P, cost

        # graph nn
        G_pos_s = knn_graph(l_s.detach(), knn).to(self.device)
        G_pos_s.ndata['h'] = f_es
        f_gs = self.gnn_s(G_pos_s)

        G_pos_t = knn_graph(l_t.detach(), knn).to(self.device)
        G_pos_t.ndata['h'] = f_et
        f_gt = self.gnn_t(G_pos_t)

        loss_ge, _, _ = self.loss_ot(f_gt, f_es)
        loss_eg, _, _ = self.loss_ot(f_et, f_gt)
        loss_g, P, cost = self.loss_ot(f_gt, f_gs)

        loss = self.g_weight * loss_g + self.e_weight * loss_e + self.ge_weight * loss_ge + self.eg_weight * loss_eg

        return loss, loss_e, loss_g, loss_ge, loss_eg, P, cost


class OTLoss(torch.nn.Module):
    def __init__(self, opt):
        super(OTLoss, self).__init__()
        self.ot_gamma = opt.ot_gamma
        self.ot_eps = opt.ot_eps
        self.ot_iter = opt.ot_iter
        self.device = opt.device

    def forward(self, ft, fs):
        X = torch.cat((ft, fs), 0).to(self.device)
        C = cosine_similarity(X)
        n = C.shape[0] // 2
        M = C[0:n, n:]

        M = zscore_normalize(M)
        P = sinkhorn(1 - M.unsqueeze(0), gamma=self.ot_gamma, eps=self.ot_eps, maxiters=self.ot_iter).squeeze(0)
        P = doubly_normalize(P)

        loss_ot = torch.norm((P - torch.eye(P.shape[1], device=self.device)), 2)
        return loss_ot, P, M


def sinkhorn(M, r=None, c=None, gamma=1.0, eps=1.0e-6, maxiters=20, logspace=False):

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

def cosine_similarity(m):
    """Compute the cosine similarity matrix."""
    m_norm = F.normalize(m, p=2, dim=1)
    return torch.matmul(m_norm, m_norm.T)


def zscore_normalize(C):
    return (C - C.mean()) / C.std()


def doubly_normalize(P):
    P = P / P.sum(dim=1, keepdim=True)
    P = P / P.sum(dim=0, keepdim=True)
    return P


def cos_distance_softmax(x):
    soft = F.softmax(x, dim=2)
    w = soft.norm(p=2, dim=2, keepdim=True)
    return 1 - soft @ B.swapaxes(soft, -1, -2) / (w @ B.swapaxes(w, -1, -2)).clamp(min=eps)


def knn_graph(x, k):
    if B.ndim(x) == 2:
        x = B.unsqueeze(x, 0)
    n_samples, n_points, _ = B.shape(x)

    dist = cos_distance_softmax(x)

    fil = 1 - torch.eye(n_points, n_points, device=x.device)
    dist = dist * B.unsqueeze(fil, 0)
    dist = dist - B.unsqueeze(torch.eye(n_points, n_points, device=x.device), 0)

    k_indices = B.argtopk(dist, k, 2, descending=False)

    dst = B.copy_to(k_indices, B.cpu())
    src = B.zeros_like(dst) + B.reshape(B.arange(0, n_points), (1, -1, 1))

    per_sample_offset = B.reshape(B.arange(0, n_samples) * n_points, (-1, 1, 1))
    dst += per_sample_offset
    src += per_sample_offset
    dst = B.reshape(dst, (-1,))
    src = B.reshape(src, (-1,))
    adj = sparse.csr_matrix((B.asnumpy(B.zeros_like(dst) + 1), (B.asnumpy(dst), B.asnumpy(src))))

    g = DGLGraph(adj)  # g = DGLGraph(adj, readonly=True)
    return g


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
        return x  / norm