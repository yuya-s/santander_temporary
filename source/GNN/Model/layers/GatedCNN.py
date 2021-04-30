import torch.nn as nn
from utils.Model.layers.CNN1D import CNN1D

class GatedCNN(nn.Module):
    def __init__(self, opt, in_channels=16, out_channels=32, kernel_size=2):
        super(GatedCNN, self).__init__()

        self.cnn1d_0 = CNN1D(opt, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.cnn1d_1 = CNN1D(opt, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        """"
        input_data  :(batch_size, n_node, L, in_channels)
        input_data0 :(batch_size, n_node, L-kernel_size+1, out_channels)
        input_data1 :(batch_size, n_node, L-kernel_size+1, out_channels)
        output_data :(batch_size, n_node, L-kernel_size+1, out_channels)
        """
        input_data0 = self.cnn1d_0(input_data)
        input_data1 = self.cnn1d_1(input_data)
        output_data = input_data0 * self.sigmoid(input_data1)
        return output_data