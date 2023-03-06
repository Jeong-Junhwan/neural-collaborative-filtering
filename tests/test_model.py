import pytest
from model import (
    GeneralMatrixFactorization,
    MultiLayerPerceptron,
    NeuralMatrixFactorization,
)
import torch


def test_GeneralMatrixFactorization():
    model = GeneralMatrixFactorization(n_users=10, n_items=15, embedding_dim=16)
    data = torch.LongTensor([[1, 5, 1], [5, 7, 0], [5, 1, 1], [4, 2, 1]])

    assert model.forward(data).shape == (4, 16)  # (batch, embedding_dim)


def test_MultiLayerPerceptron():
    model = MultiLayerPerceptron(
        n_users=10, n_items=15, embedding_dim=16, mlp_layers=(30, 10, 5)
    )
    data = torch.LongTensor([[1, 5, 1], [2, 5, 1], [3, 5, 0], [4, 5, 0]])

    output = model.forward(data)
    assert output.shape == (4, 5)  # (batch, last mlp_layers)


def test_NeuralMatrixFactorization():
    model = NeuralMatrixFactorization(
        n_users=10,
        n_items=15,
        GMF_embedding_dim=16,
        MLP_embedding_dim=4,
        mlp_layers=(9, 3, 6),
    )

    data = torch.LongTensor([[1, 5, 1], [2, 5, 1], [3, 5, 0], [4, 5, 0]])

    output = model.forward(data)

    assert output.shape == (4,)  # (batch)
