import torch
import torch.nn.functional as F
import ot
import time
from helper import gromov_parameter
import numpy as np


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


# rng = np.random.RandomState(42)


class GWLoss(torch.nn.Module):

    def __init__(self, method='pcc', gw_reg=1.0, gw_iter=10, gw_tol=1e-5, device='cuda'):
        super(GWLoss, self).__init__()

        self.method = method
        self.gw_reg = gw_reg
        self.gw_iter = gw_iter
        self.gw_tol = gw_tol
        self.device = device

    def forward(self, ft, fs):
        # Concatenate teacher and student embeddings
        X = torch.cat((ft, fs), 0).to(self.device)
        if self.method == 'pcc':
            C = PCC(X)
        elif self.method == 'cos':
            C = cosine_similarity(X)
        elif self.method == 'edu':
            C = euclidean_dist(X)
        else:
            raise ValueError("Invalid method specified.")\

        n = C.shape[0] // 2
        Et = C[0:n, 0:n]  # Teacher edge matrix
        Es = C[n:, n:]  # Student edge matrix
        Nst = C[0:n, n:]  # Node matrix

        Ct = 1 - Et
        Cs = 1 - Es

        # a0 = rng.rand(Ct.shape[0])
        # a0 /= a0.sum()
        # a1_torch = torch.tensor(a0).requires_grad_(True)
        # a2_torch = torch.tensor(a2)
        # Ct = minmax_normalize(Ct)
        # Cs = minmax_normalize(Cs)

        res = gromov_parameter.solve_gromov(Ca=Ct, Cb=Cs, M=None, alpha=1, reg=self.gw_reg, max_iter=self.gw_iter, tol=self.gw_tol)
        # res = ot.solve_gromov(Ca=Ct, Cb=Cs, M=None, alpha=1, reg=self.gw_reg, max_iter=self.gw_iter,tol=self.gw_tol)
        T = res.plan
        # T = doubly_normalize(T)
        loss_edge = res.value_quad
        # loss_edge = torch.norm((T - torch.eye(T.shape[1], device=self.device)), p=2)
        GM_loss = loss_edge

        return GM_loss, T
