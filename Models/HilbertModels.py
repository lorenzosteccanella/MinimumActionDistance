# This file is based on the original implementation of the Hilbert representation model
# produced by Park et al. (2024). More specifically, it is based directly on the implementation
# details specified in the paper and code published by the authors.
#
# The following code files were particularly useful in helping us produce an
# accurate and faithful implementation of the Hilbert representation model:
# - https://github.com/seohongpark/HILP/blob/master/hilp_gcrl/src/special_networks.py
# - https://github.com/seohongpark/HILP/blob/master/hilp_gcrl/main.py
# - https://github.com/seohongpark/HILP/blob/master/hilp_gcrl/src/agents/hilp.py

import torch


class HilbertEmbeddingModel(torch.nn.Module):
    def __init__(
        self,
        obs_dim: int,
        embedding_dim: int,
        hidden_dims: list[int] = [512],
        use_layer_norm: bool = True,
        force_positive: bool = False,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.use_layer_norm = use_layer_norm

        # Input layer
        self.layers.append(torch.nn.Linear(obs_dim, hidden_dims[0]))
        if self.use_layer_norm:
            self.layers.append(torch.nn.LayerNorm(hidden_dims[0]))
        self.layers.append(torch.nn.GELU())

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if self.use_layer_norm:
                self.layers.append(torch.nn.LayerNorm(hidden_dims[i + 1]))
            self.layers.append(torch.nn.GELU())

        # Output layer
        self.out = torch.nn.Linear(hidden_dims[-1], embedding_dim)

        if force_positive:
            self.final_activation = torch.nn.ReLU()
        else:
            self.final_activation = torch.nn.Identity()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = s
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return self.final_activation(x)
