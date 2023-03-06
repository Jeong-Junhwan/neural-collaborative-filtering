import pytest
from typing import Tuple, Dict
import pandas as pd
import numpy as np

from preprocess import (
    load_raw_data,
    leave_one_out,
    generate_negative_samples,
    load_train_data,
)


@pytest.fixture
def data() -> Tuple[pd.DataFrame, Dict[str, int]]:
    inter_data, data_info = load_raw_data()
    return inter_data, data_info


def test_load_raw_data(data: Tuple[pd.DataFrame, Dict[str, int]]):
    inter_data, data_info = data

    assert "users" in data_info
    assert "items" in data_info
    assert "ratings" in data_info
    assert inter_data.shape == (data_info["ratings"], 4)

    assert inter_data["user_id"].max() == data_info["users"]
    assert inter_data["item_id"].max() == data_info["items"]


def test_leave_one_out(data: Tuple[pd.DataFrame, Dict[str, int]]):
    train_data, valid_data = leave_one_out(data[0])

    assert len(train_data) > 5 * len(valid_data)
    assert train_data.shape[1] == 3
    assert valid_data.shape[1] == 3


def test_generate_negative_samples(
    data: Tuple[pd.DataFrame, Dict[str, int]], negative_ratio: int = 10
):
    inter_data, data_info = data
    train_data_1, _ = leave_one_out(inter_data)
    train_data_2 = generate_negative_samples(train_data_1, data_info, negative_ratio)

    assert len(train_data_2) == len(train_data_1) + data_info["users"] * negative_ratio


def test_load_train_data(negative_ratio: int = 10):
    train_data, valid_data, data_info = load_train_data(negative_ratio)

    assert train_data["user_id"].max() == data_info["users"]
    assert train_data["item_id"].max() == data_info["items"]
    assert valid_data["user_id"].max() == data_info["users"]
    assert valid_data["item_id"].max() <= data_info["items"]
