# Define the MLP class
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()

        if dropout > 0.0:
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
            )
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x):
        return self.net(self.ln(x))


class InputLayer(nn.Module):
    def __init__(self, input_dim, n_embd):
        super().__init__()
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(input_dim, n_embd)

    def forward(self, x):
        out = self.flatten(x)
        return self.proj(out)


class MultiLayerPercetronLayer(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([FeedFoward(input_dim) for _ in range(num_layers)])

        self.ln_f = nn.LayerNorm(input_dim)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = out + layer(out)
        out = self.ln_f(out)
        return out


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers, n_embd):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input_layer = InputLayer(input_dim, n_embd)
        self.layers = nn.ModuleList([FeedFoward(n_embd) for _ in range(num_layers)])

        self.ln_f = nn.LayerNorm(n_embd)
        self.output_layer = nn.Linear(n_embd, self.output_dim)

    def forward(self, x):
        out = self.input_layer(x)
        for layer in self.layers:
            out = out + layer(out)
        out = self.ln_f(out)
        out = self.output_layer(out)
        return out
