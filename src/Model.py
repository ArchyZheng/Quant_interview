# used for create torch model.
# %%
import torch
import torch.nn as nn
from torch import tensor


class BaseGRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, with_attention: bool):
        """
        Only use one GRU layer, the size of input is 5 * 320, which 320 is mentioned by 20 days data point.
        We also use BN to normalize the output
        Finally, we use FC to predict the output, which size is 1

        :param with_attention: Does it add attention mechanism
        :param input_size:
        :param hidden_size:
        :param num_layers:
        """
        super(BaseGRUModel, self).__init__()
        self.hidden_dim = hidden_size
        self.n_layers = num_layers

        self.GRU = nn.GRU(batch_first=True, input_size=input_size, hidden_size=self.hidden_dim,
                          num_layers=self.n_layers)
        self.BN = nn.BatchNorm1d(num_features=self.hidden_dim, affine=False)
        self.attention = with_attention
        self.FC = nn.Linear(self.hidden_dim, 1)

    def forward(self, previous_data: tensor, h) -> tensor:
        output, h = self.GRU(previous_data, h)
        final_output = output[:, -1, :]
        output = self.BN(final_output)
        output = self.FC(output).reshape(-1, )
        return output, h

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
