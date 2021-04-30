import torch.nn as nn
from utils.Model.layers.ST_blocks import ST_blocks
from utils.Model.layers.GatedCNN import GatedCNN

class Encoder(nn.Module):

    def __init__(self, opt, kernel_size=2, n_blocks=1, state_dim_bottleneck=64, annotation_dim_bottleneck=64):
        super(Encoder, self).__init__()

        self.batchSize = opt.batchSize
        self.state_dim = opt.state_dim
        self.n_node = opt.n_node

        # ST-Block層
        self.st_blocks = ST_blocks(opt, kernel_size=kernel_size, n_blocks=n_blocks, state_dim_bottleneck=state_dim_bottleneck, annotation_dim_bottleneck=annotation_dim_bottleneck)
        opt.L = opt.L - 2 * n_blocks * (kernel_size - 1)

        # GCNN1
        k = int(opt.L / 2)
        self.gcnn1 = GatedCNN(opt, in_channels=self.state_dim, out_channels=self.state_dim*2, kernel_size=k)
        opt.L = opt.L - (k - 1)

        # GCNN2
        k = int(opt.L / 2)
        self.gcnn2 = GatedCNN(opt, in_channels=self.state_dim*2, out_channels=self.state_dim*4, kernel_size=k)
        opt.L = opt.L - (k - 1)

        # GCNN3
        k = opt.L
        self.gcnn3 = GatedCNN(opt, in_channels=self.state_dim*4, out_channels=self.state_dim*8, kernel_size=k)
        opt.L = opt.L - (k - 1)

        # 出力層 GCNN (時間軸を単一ステップへマッピング)
        # self.gcnn = GatedCNN(opt, in_channels=self.state_dim, out_channels=5, kernel_size=opt.L)
        # opt.L = 1

        opt.L = opt.init_L
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        """"
        prop_state :(batch_size, n_node, L, state_dim)
        annotation :(batch_size, n_node, L, annotation_dim)
        A          :(batch_size, 2, 3, max_nnz)
        output     :(batch_size, n_node, state_dim)
        """
        output = self.st_blocks(prop_state, annotation, A)                 # (batch_size, n_node, L-2*n_blocks(kernel_size-1), state_dim)
        if output.shape[2] > 1:
            #output = self.gcnn(output)                                     # (batch_size, n_node, 1, state_dim)
            output = self.gcnn1(output)
            output = self.gcnn2(output)
            output = self.gcnn3(output)
        output = output.view(self.batchSize, self.n_node, 40)  # (batch_size, n_node, 40)
        return output