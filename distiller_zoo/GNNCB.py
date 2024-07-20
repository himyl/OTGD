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

""" HKD method with InfoNCE estimator Loss + OT loss"""


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
        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, x):
        bsz = x.shape[0]
        # label 指定每个采样的正样本 idx = 0
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


class GNNComLoss(nn.Module):
    def __init__(self, opt):
        super(GNNComLoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.gnn_s = Encoder(opt.feat_dim, opt.feat_dim)
        self.gnn_t = Encoder(opt.feat_dim, opt.feat_dim)
        self.contrast = NCEAverage(opt.feat_dim, opt.n_data, opt.nce_k).cuda()
        self.criterion = NCESoftmaxLoss()
        self.feat_size = opt.feat_dim
        self.loss_ot = OTLoss(opt)
        self.device = opt.device

        u = torch.tensor([i for i in range(opt.batch_size * opt.nce_k)]).cuda()
        v = torch.tensor([i for i in range(opt.batch_size * opt.nce_k)]).cuda()
        self.G_neg = dgl.graph((u, v)).to('cuda:0')

    def forward(self, epoch, f_s, l_s, f_t, l_t, idx, contrast_idx=None):
        batchSize = f_s.size(0)
        K = self.contrast.K
        T = 0.07

        weight_t, weight_s = self.contrast(batchSize, idx, contrast_idx)

        # graph indepandent
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

        if batchSize < knn:
            return loss_e

        # graph nn
        G_pos_s = knn_graph(l_s.detach(), knn)
        G_pos_s = G_pos_s.to('cuda:0')
        G_pos_s.ndata['h'] = f_es
        f_gs = self.gnn_s(G_pos_s)

        G_pos_t = knn_graph(l_t.detach(), knn)
        G_pos_t = G_pos_t.to('cuda:0')
        G_pos_t.ndata['h'] = f_et
        f_gt = self.gnn_t(G_pos_t)

        f_sgs, f_sgt = self.contrast.get_smooth(f_gs, f_gt, idx)

        gs_pos = torch.einsum('nc,nc->n', [f_sgt, f_gs]).unsqueeze(-1)
        gt_pos = torch.einsum('nc,nc->n', [f_sgs, f_gt]).unsqueeze(-1)

        gs_neg = torch.bmm(weight_t, f_gs.view(batchSize, self.feat_size, 1)).squeeze()
        gt_neg = torch.bmm(weight_s, f_gt.view(batchSize, self.feat_size, 1)).squeeze()

        out_gs = torch.cat([gs_pos, gs_neg], dim=1)
        out_gs = torch.div(out_gs, T)
        out_gs = out_gs.contiguous()

        out_gt = torch.cat([gt_pos, gt_neg], dim=1)
        out_gt = torch.div(out_gt, T)
        out_gt = out_gt.contiguous()

        loss_g = self.criterion(out_gs) + self.criterion(out_gt)

        loss_e_ot, P, M = self.loss_ot(f_et, f_gs)
        loss_g_ot, P, M = self.loss_ot(f_gt, f_gs)
        loss = loss_e + loss_g + loss_e_ot + loss_g_ot

        self.contrast.update(f_es, f_et, idx)
        return loss, loss_e + loss_g, loss_e_ot + loss_g_ot, P, M


#     def forward(self, epoch, f_s, l_s, f_t, l_t):
#         batchSize = f_s.size(0)
#
#         # graph independent part
#         f_es = self.embed_s(f_s)
#         f_et = self.embed_t(f_t)
#         Pe = self.loss_ot(f_et, f_es)
#         P = self.normalize_P(Pe)
#         loss_e = torch.norm((P - torch.eye(P.shape[1], device=self.device)), 2)
#
#         if batchSize < knn:
#             return loss_e
#
#         # graph nn
#         G_pos_s = knn_graph(l_s.detach(), knn).to(self.device)
#         G_pos_s.ndata['h'] = f_es
#         f_gs = self.gnn_s(G_pos_s)
#
#         G_pos_t = knn_graph(l_t.detach(), knn).to(self.device)
#         G_pos_t.ndata['h'] = f_et
#         f_gt = self.gnn_t(G_pos_t)
#
#         Pg = self.loss_ot(f_gt, f_gs)
#         Psum = Pg + Pe
#         Psum = self.normalize_P(Psum)
#         loss = torch.norm((Psum - torch.eye(Psum.shape[1], device=self.device)), 2)
#
#         return loss, loss_e, Psum
#
#     def normalize_P(self, P):
#         if self.P_norm == 'Pr':
#             return row_normalize(P)
#         elif self.P_norm == 'Pc':
#             return column_normalize(P)
#         elif self.P_norm == 'Prc':
#             return doubly_normalize(P)
#         elif self.P_norm == 'SC':
#             return softmax_scaling(P, self.tau)
#         return P
#
#
class OTLoss(torch.nn.Module):
    def __init__(self, opt):
        super(OTLoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.method = opt.ot_method
        self.ot_gamma = opt.ot_gamma
        self.ot_eps = opt.ot_eps
        self.ot_iter = opt.ot_iter
        self.device = opt.device
        self.M_norm = opt.M_norm
        self.P_norm = opt.P_norm
        self.tau = opt.tau

    def forward(self, ft, fs):
        X = torch.cat((ft, fs), 0).to(self.device)
        C = self.compute_similarity(X)
        n = C.shape[0] // 2
        M = C[0:n, n:]

        M = self.normalize_M(M)
        P = sinkhorn(1 - M.unsqueeze(0), gamma=self.ot_gamma, eps=self.ot_eps, maxiters=self.ot_iter).squeeze(0)
        P = self.normalize_P(P)

        loss_ot = torch.norm((P - torch.eye(P.shape[1], device=self.device)), 2)
        return loss_ot, P, M

    def compute_similarity(self, X):
        if self.method == 'pcc':
            return PCC(X)
        elif self.method == 'cos':
            return cosine_similarity(X)
        raise ValueError("Invalid method specified.")

    def normalize_M(self, M):
        if self.M_norm == 'Mm':
            return minmax_normalize(M)
        elif self.M_norm == 'Mz':
            return zscore_normalize(M)
        elif self.M_norm == 'Mmz':
            return mmzs_normalize(M)
        return M

    def normalize_P(self, P):
        if self.P_norm == 'Pr':
            return row_normalize(P)
        elif self.P_norm == 'Pc':
            return column_normalize(P)
        elif self.P_norm == 'Prc':
            return doubly_normalize(P)
        elif self.P_norm == 'SC':
            return softmax_scaling(P, self.tau)
        return P



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
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    c = fact * m.matmul(mt).squeeze()
    d = torch.diag(c, 0)  # 提取对角线元素
    std = torch.sqrt(d)  # 计算标准差
    c /= std[:, None]
    c /= std[None, :]
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


def mmzs_normalize(C):
    C = minmax_normalize(C)
    C = zscore_normalize(C)
    return C


def row_normalize(P):
    row_sums = P.sum(dim=1, keepdim=True)
    return P / row_sums


def softmax_scaling(P, Tau):
    m = torch.nn.Softmax(dim=1)
    P = m(P / Tau)
    return P


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
        out = x.div(norm)
        return out
