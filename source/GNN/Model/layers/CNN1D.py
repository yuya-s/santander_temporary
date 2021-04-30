import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, opt, in_channels=16, out_channels=32, kernel_size=2):
        super(CNN1D, self).__init__()

        self.n_node = opt.n_node
        self.L = opt.L
        self.batchSize = opt.batchSize

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1d = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=1, padding=0)

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input_data):
        """"
        input_data  :(batch_size, n_node, L, in_channels)
        output_data :(batch_size, n_node, L-kernel_size+1, out_channels)

        The input for conv1d() is (N, C_in, L)
        N    :batch size
        C_in :a number of channels (e.g. state_dim, annotation_dim)
        L    :length of signal sequence
        """
        input_data = input_data.view(self.batchSize * self.n_node, self.L, self.in_channels) # (batch_size * n_node, L, in_channels)
        input_data = input_data.transpose(1, 2) # (batch_size * n_node, in_channels, L)
        output_data = self.conv1d(input_data) # (batch_size * n_node, out_channels, L-kernel_size+1)
        output_data = output_data.view(self.batchSize, self.n_node, self.out_channels, self.L-self.kernel_size+1).transpose(2, 3)
        return output_data