import pandas as pd
import numpy as np
from typing import Tuple, Dict


col_names = ["user_id", "item_id", "rating", "timestamp"]
dtypes = {
    "user_id": np.int64,
    "item_id": np.int64,
    "rating": np.int64,
    "timestamp": np.int64,
}


def load_raw_data() -> Tuple[pd.DataFrame, Dict[str, int]]:
    # read raw data
    inter_data = pd.read_csv("ml-100k/u.data", sep="\t", names=col_names, dtype=dtypes)

    data_info = dict()
    with open("ml-100k/u.info", "r") as file:
        for line in file:
            n, what = line.split()
            data_info[what] = int(n)

    # convert to implicit feedback
    inter_data = inter_data.query("rating >= 3")
    inter_data.loc[:, "rating"] = 1

    data_info["ratings"] = len(inter_data)

    return inter_data, data_info


def leave_one_out(inter_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inter_data["time_order"] = inter_data.groupby("user_id")["timestamp"].rank(
        method="first", ascending=False
    )

    train_data = inter_data.query("time_order > 1")
    valid_data = inter_data.query("time_order == 1")

    train_data = train_data[["user_id", "item_id", "rating"]].reset_index(drop=True)
    valid_data = valid_data[["user_id", "item_id", "rating"]].reset_index(drop=True)

    return train_data, valid_data


def generate_negative_samples(
    inter_data: pd.DataFrame, data_info: Dict[str, int], negative_ratio: int
) -> pd.DataFrame:
    # find negative samples (don't have interaction with each user)
    gb = inter_data.groupby("user_id")["item_id"].apply(set)
    non_inter_data = gb.apply(
        lambda x: list(set(range(1, data_info["items"] + 1)) - set(x))
    )

    non_inter_data = (
        non_inter_data.apply(
            lambda x: np.random.choice(x, size=negative_ratio, replace=False)
        )
        .explode()
        .reset_index()
    )
    non_inter_data["rating"] = 0

    train_data = pd.concat([inter_data, non_inter_data]).reset_index(drop=True)

    return train_data


def load_train_data(
    negative_ratio: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    inter_data, data_info = load_raw_data()
    train_data, valid_data = leave_one_out(inter_data)
    train_data = generate_negative_samples(train_data, data_info, negative_ratio)
    return train_data, valid_data, data_info
