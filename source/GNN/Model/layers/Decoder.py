import torch.nn as nn
import torch

class Decoder(nn.Module):

    def __init__(self, opt, hidden_state):
        super(Decoder, self).__init__()

        self.batchSize = opt.batchSize
        self.state_dim = opt.state_dim
        self.n_node = opt.n_node
        self.H = opt.H
        self.hidden_dim = hidden_state
        self.output_dim = opt.output_dim

        self.lstm = nn.LSTM(input_size=45, # peeky
                             hidden_size=self.hidden_dim,
                             batch_first=True)

        self.out = nn.Linear(80, opt.output_dim) # peeky

        """
        self.lstm = nn.LSTM(input_size=5,
                             hidden_size=self.hidden_dim,
                             batch_first=True)

        self.out = nn.Linear(40, opt.output_dim)
        """

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input_state, hidden_state, is_train):
        """"
        input_state :(batch_size, n_node, H, output_dim)
        hidden_state:((num_layers * num_directions, batch_size * n_node, hidden_dim), (num_layers * num_directions, batch_size * n_node, hidden_dim))
        h_t         :(batch_size * n_node, H, hidden_dim) 最後の層の各tにおける隠れ状態
        h_n         :(num_layers * num_directions, batch_size * n_node, hidden_dim) 時系列の最後の隠れ状態
        c_n         :(num_layers * num_directions, batch_size * n_node, hidden_dim) 時系列の最後のセル状態
        num_layersはLSTMの層数、スタック数。
        num_directionsはデフォルト1、双方向で2。
        """
        if is_train:
            input_state = input_state.view(self.batchSize * self.n_node, self.H, 45) # peeky
            #input_state = input_state.view(self.batchSize * self.n_node, self.H, 5)
        else:
            input_state = input_state.view(self.batchSize * self.n_node,      1, 45) # peeky
            #input_state = input_state.view(self.batchSize * self.n_node,      1, 5)

        h_t, (h_n, c_n) = self.lstm(input_state, hidden_state)
        # Many to Manyなので、第１戻り値を使う。
        # 第２戻り値は推論時に使う。
        output = self.out(torch.cat((h_t, input_state[:,:,5:]), dim=2)) # peeky
        # output = self.out(h_t)
        state = (h_n, c_n)
        return output, state