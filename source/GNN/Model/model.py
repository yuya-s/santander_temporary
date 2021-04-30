import torch.nn as nn

class STGGNN(nn.Module):

    def __init__(self, opt, kernel_size=2, n_blocks=1, state_dim_bottleneck=64, annotation_dim_bottleneck=64):
        super(STGGNN, self).__init__()

        self.batchSize = opt.batchSize
        self.state_dim = opt.state_dim
        self.n_node = opt.n_node

        # ST-Block層
        self.st_blocks = ST_blocks(opt, kernel_size=kernel_size, n_blocks=n_blocks, state_dim_bottleneck=state_dim_bottleneck, annotation_dim_bottleneck=annotation_dim_bottleneck)
        opt.L = opt.L - 2 * n_blocks * (kernel_size - 1)

        # 出力層 GCNN (時間軸を単一ステップへマッピング)
        self.gcnn = GatedCNN(opt, in_channels=self.state_dim, out_channels=self.state_dim, kernel_size=opt.L)
        opt.L = 1

        # 出力層 FC (回帰用)
        self.out = nn.Sequential(
            nn.Linear(self.state_dim, opt.output_dim),
            #nn.Sigmoid() 使ったら精度落ちる
        )

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
        output     :(batch_size, n_node, output_dim)
        """
        output = self.st_blocks(prop_state, annotation, A)                 # (batch_size, n_node, L-2*n_blocks(kernel_size-1), state_dim)
        if output.shape[2] > 1:
            output = self.gcnn(output)                                     # (batch_size, n_node, 1, state_dim)
        output = output.view(self.batchSize, self.n_node, self.state_dim)  # (batch_size, n_node, state_dim)
        output = self.out(output)                                          # (batch_size, n_node, output_dim)
        return output