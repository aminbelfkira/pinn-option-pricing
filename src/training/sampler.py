import torch
import numpy as np

from src.config import BSParams
from src.constants import N_IVP, N_BVP1, N_BVP2, N_INT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")s
params = BSParams()

# IVP


def sample_ivp_call():
    S_ivp = np.random.uniform(params.S_min, params.S_max, size=N_IVP)
    t_ivp = np.array([params.T for _ in range(N_IVP)])
    V_ivp = np.maximum(S_ivp - params.K, 0)

    return S_ivp, t_ivp, V_ivp


# BVP1


def sample_bvp1_call():
    S_bvp1 = np.array([params.S_min for _ in range(N_BVP1)])
    t_bvp1 = np.random.uniform(0, params.T, size=N_BVP1)
    V_bvp1 = np.zeros(N_BVP1)

    return S_bvp1, t_bvp1, V_bvp1


# BVP2
def sample_bvp2_call():
    S_bvp2 = np.array([params.S_max for _ in range(N_BVP2)])
    t_bvp2 = np.random.uniform(0, params.T, size=N_BVP2)
    V_bvp2 = S_bvp2 - params.K * np.exp(-params.r * t_bvp2)

    return S_bvp2, t_bvp2, V_bvp2


# intérieur


def sample_int_call():
    S = np.random.uniform(params.S_min, params.S_max, size=N_INT)
    t = np.random.uniform(0, params.T, size=N_INT)
    V = np.array([np.nan for _ in range(N_INT)])

    return S, t, V
