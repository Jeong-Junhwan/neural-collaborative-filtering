from preprocess import load_train_data
from dataset import NCFDataset
from model import NeuralMatrixFactorization
from trainer import NCFTrainer
from torch.utils.data import DataLoader, default_collate


def main(config):
    train_data, valid_data, data_info = load_train_data()

    train_dataset = NCFDataset(train_data)
    valid_dataset = NCFDataset(valid_data)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=default_collate,
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=default_collate,
    )

    model = NeuralMatrixFactorization(
        n_users=data_info["users"],
        n_items=data_info["items"],
        GMF_embedding_dim=config["GMF_embedding_dim"],
        MLP_embedding_dim=config["MLP_embedding_dim"],
        mlp_layers=config["mlp_layers"],
    )
    trainer = NCFTrainer(
        model=model,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        device=config["device"],
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    trainer.train()


if __name__ == "__main__":
    import torch

    config = dict()

    config["batch_size"] = 256
    config["GMF_embedding_dim"] = 64
    config["MLP_embedding_dim"] = 64
    config["mlp_layers"] = (128, 64, 32)

    config["epochs"] = 100
    config["learning_rate"] = 0.001
    config["weight_decay"] = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device

    main(config)
