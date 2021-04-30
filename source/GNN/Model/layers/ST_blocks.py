import torch.nn as nn
from utils.Model.layers.GGNN import GGNN
from utils.Model.layers.GatedCNN import GatedCNN

class AttrProxy(object):
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

class ST_blocks(nn.Module):
    def __init__(self, opt, kernel_size=2, n_blocks=1, state_dim_bottleneck=64, annotation_dim_bottleneck=64):
        super(ST_blocks, self).__init__()

        assert opt.L - 2 * n_blocks * (kernel_size - 1) >= 1, 'L length is insufficient'

        self.kernel_size = kernel_size
        self.n_blocks = n_blocks

        # bottleneck strategy (ST_blocks have residual connection)
        self.state_dim_bottleneck = state_dim_bottleneck
        self.annotation_dim_bottleneck = annotation_dim_bottleneck
        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        opt.state_dim = self.state_dim_bottleneck
        opt.annotation_dim = self.annotation_dim_bottleneck

        for i in range(self.n_blocks):
            gcnn_tp = GatedCNN(opt, in_channels=self.state_dim, out_channels=self.state_dim_bottleneck, kernel_size=self.kernel_size)
            gcnn_ta = GatedCNN(opt, in_channels=self.annotation_dim, out_channels=self.annotation_dim_bottleneck, kernel_size=self.kernel_size)
            opt.L = opt.L - kernel_size + 1

            ggnn = GGNN(opt)

            gcnn_bp = GatedCNN(opt, in_channels=self.state_dim_bottleneck, out_channels=self.state_dim, kernel_size=self.kernel_size)
            gcnn_ba = GatedCNN(opt, in_channels=self.annotation_dim_bottleneck, out_channels=self.annotation_dim, kernel_size=self.kernel_size)
            opt.L = opt.L - kernel_size + 1

            self.add_module("gcnn_tp_{}".format(i), gcnn_tp)
            self.add_module("gcnn_ta_{}".format(i), gcnn_ta)
            self.add_module("ggnn_{}".format(i), ggnn)
            self.add_module("gcnn_bp_{}".format(i), gcnn_bp)
            self.add_module("gcnn_ba_{}".format(i), gcnn_ba)

        opt.L = opt.init_L
        opt.state_dim = self.state_dim
        opt.annotation_dim = self.annotation_dim

        self.gcnn_tps = AttrProxy(self, "gcnn_tp_")
        self.gcnn_tas = AttrProxy(self, "gcnn_ta_")
        self.ggnns = AttrProxy(self, "ggnn_")
        self.gcnn_bps = AttrProxy(self, "gcnn_bp_")
        self.gcnn_bas = AttrProxy(self, "gcnn_ba_")


    def forward(self, prop_state, annotation, A):
        """"
        prop_state  :(batch_size, n_node, L, state_dim)
        annotation  :(batch_size, n_node, L, annotation_dim)
        A           :(batch_size, 2, 3, max_nnz)
        output_data :(batch_size, n_node, L-2*n_blocks(kernel_size-1), state_dim)
        """
        output_data = prop_state
        for i in range(self.n_blocks):
            # GatedCNN top
            output_data = self.gcnn_tps[i](output_data)
            annotation = self.gcnn_tas[i](annotation)
            # GGNN
            output_data = self.ggnns[i](output_data, annotation, A)
            # GatedCNN bottom
            output_data = self.gcnn_bps[i](output_data)
            annotation = self.gcnn_bas[i](annotation)
        return output_data