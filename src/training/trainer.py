import torch
from dataclasses import dataclass

from src.training.loss import loss_bvp1_call, loss_bvp2_call, loss_ivp_call, loss_pde


@dataclass
class TrainConfig:

    learning_rate: float = 1e-3
    beta: float = 1.0
    num_iterations: int = 50_000
    print_every: int = 1_000


class TrainerEuropeanCall:

    def __init__(self, model, bs_params, train_config: TrainConfig):

        self.model = model
        self.bs_params = bs_params
        self.config = train_config

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

    def train_step(self):

        self.model.train()

        loss_ivp = loss_ivp_call(self.model, self.bs_params)
        loss_bvp1 = loss_bvp1_call(self.model, self.bs_params)
        loss_bvp2 = loss_bvp2_call(self.model, self.bs_params)
        loss_edp = loss_pde(self.model, self.bs_params)

        loss_total = loss_ivp + loss_bvp1 + loss_bvp2 + self.config.beta * loss_edp

        self.optimizer.zero_grad()

        loss_total.backward()
        self.optimizer.step()

        return {
            "loss_total": loss_total,
            "loss_ivp": loss_ivp,
            "loss_bvp1": loss_bvp1,
            "loss_bvp2": loss_bvp2,
            "loss_edp": loss_edp,
        }

    def train(self):

        history = []

        for it in range(self.config.num_iterations):

            stats = self.train_step()
            history.append(stats)

            if it % self.config.print_every == 0:

                print(stats)

        return history
