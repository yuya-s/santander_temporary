import torch
import torch.nn as nn
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../' )
from utils.setting_param import Dataset_adj_shape

adj_shape = Dataset_adj_shape

# coo_numpy形式からtorch.sparseに変換
def coo_numpy2sparse_pytorch(coo_numpy, matrix_size):
    values = coo_numpy[0]
    indices = np.vstack((coo_numpy[1], coo_numpy[2]))
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    shape = matrix_size
    return torch.sparse.DoubleTensor(i, v, torch.Size(shape))


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propagator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim, n_node, n_edge_types, L):
        super(Propagator, self).__init__()

        self.n_node = n_node
        self.n_edge_types = n_edge_types
        self.L = L

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def bmm_sparse(self, A, state):
        """
        バッチ、タイムステップごとに2次元×2次元のスパース行列積を演算する
        A          :(batch_size, 3, max_nnz)
        state      :(batch_size, n_edge_types * n_node, L, state_dim)
        A_sparse   :sparse of (n_node, n_edge_types * n_node)
        a_ts       :list of L          * (n_node, state_dim)
        a_batch    :list of batch_size * (n_node, L, state_dim)
        """
        a_batch = []
        for batch in range(A.shape[0]): # batchごとに疎行列×密行列の行列積を計算したい
            a_ts = []
            for time_step in range(self.L): # time_stepごとに疎行列×密行列の行列積を計算したい
                A_sparse = coo_numpy2sparse_pytorch(A[batch], adj_shape) # torch.sparseに変換 (COOはスライス操作ができないので演算直前に変換する必要がある)
                a_ts.append(torch.sparse.mm(A_sparse, state[batch, :, time_step, :])) # 疎行列×密行列の行列積の演算
            a_batch.append(torch.stack(a_ts, 1)) # axis=1 (time_step方向)にstack
        a = torch.stack(a_batch, 0) # axis=0 (batch方向)にstack
        return a

    def forward(self, state_in, state_out, state_cur, A):
        """
        state_in      :(batch_size, n_edge_types * n_node, L, state_dim)
        state_out     :(batch_size, n_edge_types * n_node, L, state_dim)
        state_cur     :(batch_size, n_node, L, state_dim)
        A             :(batch_size, 2, 3, max_nnz)
        A_in          :(batch_size, 3, max_nnz)
        A_out         :(batch_size, 3, max_nnz)
        a_in          :(batch_size, n_node, L, state_dim)
        a_out         :(batch_size, n_node, L, state_dim)
        a             :(batch_size, n_node, L, state_dim * 3)
        r             :(batch_size, n_node, L, state_dim)
        z             :(batch_size, n_node, L, state_dim)
        joined_input  :(batch_size, n_node, L, state_dim * 3)
        h_hat         :(batch_size, n_node, L, state_dim)
        output        :(batch_size, n_node, L, state_dim)
        """
        A_in = A[:, 0]
        A_out = A[:, 1]

        a_in = self.bmm_sparse(A_in, state_in)
        a_out = self.bmm_sparse(A_out, state_out)
        a = torch.cat((a_in, a_out, state_cur), 3)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 3)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat
        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt):
        super(GGNN, self).__init__()

        assert opt.state_dim >= opt.annotation_dim, 'state_dim must be no less than annotation_dim'

        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.n_edge_types = opt.n_edge_types
        self.n_node = opt.n_node
        self.n_steps = opt.n_steps
        self.L = opt.L
        self.batchSize = opt.batchSize

        for i in range(self.n_edge_types):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propagator = Propagator(self.state_dim, self.n_node, self.n_edge_types, self.L)

        # Output Model
        self.out = nn.Sequential(
            nn.Linear((self.state_dim + self.annotation_dim), self.state_dim),
            nn.Tanh()
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        """"
        prop_state  :(batch_size, n_node, L, state_dim)
        annotation  :(batch_size, n_node, L, annotation_dim)
        A           :(batch_size, 2, 3, max_nnz)
        in_states   :(batch_size, n_edge_types * n_node, L, state_dim)
        out_states  :(batch_size, n_edge_types * n_node, L, state_dim)
        join_state  :(batch_size, n_node, L, state_dim + annotation_dim)
        output_data :(batch_size, n_node, L, state_dim)
        """
        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.n_edge_types):
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous() # (batch_size, n_edge_types, n_node, L, state_dim)
            in_states = in_states.view(-1, self.n_edge_types*self.n_node, self.L, self.state_dim) # (batch_size, n_edge_types * n_node, L, state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.n_edge_types*self.n_node, self.L, self.state_dim)

            prop_state = self.propagator(in_states, out_states, prop_state, A)

        join_state = torch.cat((prop_state, annotation), 3)
        output_data = self.out(join_state)
        return output_data