import torch

from src.config import BSParams
from src.models.pinn_call import PINNCall, PINNConfig
from src.training.trainer import TrainerEuropeanCall, TrainConfig


if __name__ == "__main__":

    print("Lancement du training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bs_params = BSParams()

    model_config = PINNConfig(
        input_dim=2, hidden_dim=64, num_hidden_layers=4, activation="tanh"
    )
    model = PINNCall(model_config)
    train_config = TrainConfig()
    trainer = TrainerEuropeanCall(model, bs_params, train_config)

    history = trainer.train()
