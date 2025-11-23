import torch
import numpy as np

from src.config import BSParams
from src.constants import N_IVP, N_BVP1, N_BVP2, N_INT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = BSParams()

# IVP


def sample_ivp_call(params):
    S_ivp = params.S_min + (params.S_max - params.S_min) * torch.rand(
        N_IVP, device=device
    )
    t_ivp = torch.full((N_IVP,), params.T, device=device)
    V_ivp = torch.maximum(S_ivp - params.K, torch.zeros(N_IVP, device=device))

    return S_ivp, t_ivp, V_ivp


# BVP1


def sample_bvp1_call(params):
    S_bvp1 = torch.full((N_BVP1,), params.S_min, device=device)
    t_bvp1 = params.T * torch.rand(N_BVP1, device=device)
    V_bvp1 = torch.zeros((N_BVP1,), device=device)

    return S_bvp1, t_bvp1, V_bvp1


# BVP2
def sample_bvp2_call(params):
    S_bvp2 = torch.full((N_BVP2,), params.S_max, device=device)
    t_bvp2 = params.T * torch.rand(N_BVP2, device=device)
    V_bvp2 = S_bvp2 - params.K * torch.exp(-params.r * t_bvp2)

    return S_bvp2, t_bvp2, V_bvp2


# intérieur


def sample_int_call(params):
    S_int = params.S_min + (params.S_max - params.S_min) * torch.rand(
        N_INT, device=device
    )
    t_int = params.T * torch.rand(N_INT, device=device)
    V_int = torch.full((N_INT,), np.nan, device=device)

    return S_int, t_int, V_int
