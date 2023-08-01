from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from torch.nn import Module, Parameter


class DeepLearnableModel(ABC, Module):
    mean: Parameter
    std_dev: Parameter

    def __init__(self, in_size):
        super().__init__()
        self.mean = Parameter(torch.ones(in_size), requires_grad=False)
        self.std = Parameter(torch.ones(in_size), requires_grad=False)

    @abstractmethod
    def sequential_eval(self, x: Tensor, carry: Tensor) -> Tensor:
        pass

    def set_normalization(self, x: Tensor) -> None:
        # flatten x into shape num_points * in_size
        x = torch.flatten(x, end_dim=-2)
        self.mean = Parameter(x.mean(dim=0), requires_grad=False)
        self.std = Parameter(x.std(dim=0), requires_grad=False)

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class DeepRecurrentModel(DeepLearnableModel):

    def __init__(self, in_size: int, hidden_size: int, out_size: int,
                 layers: int, nonlinearity: Module) -> None:
        super().__init__(in_size)
        encode = True
        if encode:
            self.encoder = _mlp(in_size, hidden_size, hidden_size, layers // 2,
                                nonlinearity)
        else:
            self.encoder = lambda x: x
        self.decoder = _mlp(hidden_size, hidden_size, out_size,
                            layers - (layers // 2), nonlinearity)
        rnn_in_size = hidden_size if encode else in_size
        self.recurrent = nn.GRU(input_size=rnn_in_size,
                                hidden_size=hidden_size,
                                num_layers=1,
                                batch_first=True)

    def forward(self, x: Tensor, carry: Tensor) -> Tensor:
        (next_recurrent_output, carry) = self.sequential_eval(x, carry)
        return self.decoder(next_recurrent_output), carry

    def sequential_eval(self, x: Tensor, carry: Tensor) -> Tensor:
        # pdb.set_trace()
        # x is B x L x N
        carry = carry.transpose(0, 1)
        for i in range(x.shape[1]):
            xi = self.normalize(x[:, i:(i + 1), :])
            recurrent_output, carry = self.recurrent(self.encoder(xi), carry)
        return recurrent_output, carry.transpose(0, 1)


def _mlp(in_size: int, hidden_size: int, out_size: int, layers: int,
         nonlinearity: Module) -> Module:
    modules = []
    if layers == 0:
        return nn.Linear(in_size, out_size)
    modules.append(nn.Linear(in_size, hidden_size))
    for i in range(layers - 1):
        modules.append(nonlinearity())
        modules.append(nn.Linear(hidden_size, hidden_size))
    modules.append(nonlinearity())
    modules.append(nn.Linear(hidden_size, out_size))
    return nn.Sequential(*modules)


class MLP(DeepLearnableModel):

    def __init__(self, in_size: int, hidden_size: int, out_size: int,
                 layers: int, nonlinearity: Module) -> None:
        super().__init__(in_size)
        self.net = _mlp(in_size, hidden_size, out_size, layers, nonlinearity)

    def forward(self, x: Tensor, carry: Tensor) -> Tensor:
        return self.sequential_eval(x, carry)

    def sequential_eval(self, x: Tensor, carry: Tensor) -> Tensor:
        # x is B x L x N
        return self.net(self.normalize(x[:, -1, :])).unsqueeze(1), carry


class ZeroModel(DeepLearnableModel):

    def __init__(self, in_size: int, hidden_size: int, out_size: int,
                 layers: int, nonlinearity: Module) -> None:
        super().__init__(in_size)
        self.out_size = out_size
        self.dummy_param = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, x: Tensor, carry: Tensor) -> Tensor:
        return self.sequential_eval(x, carry)

    def sequential_eval(self, x: Tensor, carry: Tensor) -> Tensor:
        # x is B x L x N
        return self.dummy_param * torch.zeros(
            (x.shape[0], 1, self.out_size)), carry

class LSTMModel(DeepLearnableModel):
    def __init__(self, in_size: int, hidden_size: int, out_size: int, num_layers: int, nonlinearity: Module, dropout_keep_prob: float, cell_type: str) -> None:
        super().__init__(in_size)
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.dropout_keep_prob = dropout_keep_prob
        self.cell_type = cell_type
        # self.fc_layers = nn.ModuleList([nn.Linear(12, hidden_size) for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(p=1 - dropout_keep_prob)
        if self.cell_type == 'rnn':
            self.rnn = nn.RNN(input_size=12, hidden_size=hidden_size, num_layers=num_layers, dropout=1-dropout_keep_prob)
        elif self.cell_type == 'gru':
            self.rnn = nn.GRU(input_size=12, hidden_size=hidden_size, num_layers=num_layers, dropout=1-dropout_keep_prob)
        elif self.cell_type == 'lstm':
            self.rnn = nn.LSTM(input_size=12, hidden_size=hidden_size, num_layers=num_layers, dropout=1-dropout_keep_prob)
            
    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.nonlinearity(x)
        x = self.fc(x)
        return x