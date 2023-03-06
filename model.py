import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class GeneralMatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users + 1, embedding_dim)
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch) * 3

        users = x[:, 0]
        items = x[:, 1]

        user_vectors = self.user_embedding(users)
        item_vectors = self.item_embedding(items)

        return torch.mul(user_vectors, item_vectors)


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        mlp_layers: Tuple[int, ...],
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users + 1, embedding_dim)
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim)

        self.MLP = self._make_layers(layers=mlp_layers, embedding_dim=embedding_dim)

        self._init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, 3)
        # 3 means user, item, rating (0 or 1)
        users = x[:, 0]
        items = x[:, 1]

        user_vectors = self.user_embedding(users)
        item_vectors = self.item_embedding(items)

        concat_vectors = torch.concat((user_vectors, item_vectors), dim=1)

        return self.MLP(concat_vectors)

    def _make_layers(self, layers: Tuple[int, ...], embedding_dim: int) -> nn.Module:
        if not layers:
            raise ValueError("should have at least one mlp layer.")

        mlp_layers = nn.Sequential()
        past_dims = embedding_dim * 2
        for layer in layers:
            mlp_layers.append(nn.Linear(past_dims, layer))
            mlp_layers.append(nn.ReLU())
            past_dims = layer

        return mlp_layers

    def _init_weight(self):
        pass


class NeuralMatrixFactorization(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        GMF_embedding_dim: int,
        MLP_embedding_dim: int,
        mlp_layers: Tuple[int, ...],
    ) -> None:
        super().__init__()
        self.GMF = GeneralMatrixFactorization(
            n_users=n_users, n_items=n_items, embedding_dim=GMF_embedding_dim
        )
        self.MLP = MultiLayerPerceptron(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=MLP_embedding_dim,
            mlp_layers=mlp_layers,
        )

        self.predict_layer = nn.Linear(GMF_embedding_dim + mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch) * 3
        gmf_vectors = self.GMF(x)
        mlp_vectors = self.MLP(x)

        concat_vectors = torch.concat((gmf_vectors, mlp_vectors), dim=1)
        output = self.predict_layer(concat_vectors)
        return self.sigmoid(output).squeeze()
