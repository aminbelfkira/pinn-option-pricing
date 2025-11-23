import torch

from src.training.sampler_tensor import (
    sample_ivp_call,
    sample_bvp1_call,
    sample_bvp2_call,
    sample_int_call,
)

from src.bs_pde import bs_pde_call


def loss_ivp_call(model, params):
    S_ivp, t_ivp, V_ivp = sample_ivp_call(params)
    V_pred = model(t_ivp, S_ivp)
    return torch.mean((V_pred - V_ivp) ** 2)


def loss_bvp1_call(model, params):
    S_bvp1, t_bvp1, _ = sample_bvp1_call(params)
    V_pred = model(t_bvp1, S_bvp1)
    return torch.mean(V_pred**2)


def loss_bvp2_call(model, params):
    S_bvp2, t_bvp2, V_bvp2 = sample_bvp2_call(params)
    V_pred = model(t_bvp2, S_bvp2)
    return torch.mean((V_pred - V_bvp2) ** 2)


def loss_pde(model, params):
    S_int, t_int, _ = sample_int_call(params)
    t_int.requires_grad_(True)
    S_int.requires_grad_(True)
    V = model(t_int, S_int)
    dV_dt = torch.autograd.grad(
        V, t_int, grad_outputs=torch.ones_like(V), create_graph=True
    )[0]
    dV_dS = torch.autograd.grad(
        V, S_int, grad_outputs=torch.ones_like(V), create_graph=True
    )[0]
    d2V_dS2 = torch.autograd.grad(
        dV_dS, S_int, grad_outputs=torch.ones_like(dV_dS), create_graph=True
    )[0]
    f = bs_pde_call(t_int, S_int, V, params.r, params.sigma, dV_dt, dV_dS, d2V_dS2)
    return torch.mean(f**2)
