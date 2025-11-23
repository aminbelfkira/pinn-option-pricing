import numpy as np

from src.config import BSParams
from src.constants import N_IVP, N_BVP1, N_BVP2, N_INT


if __name__ == "__main__":

    params = BSParams()

    # IVP

    S_ivp = np.random.uniform(params.S_min, params.S_max, size=N_IVP)
    t_ivp = np.array([params.T for _ in range(N_IVP)])
    v_ivp = np.maximum(S_ivp - params.K, 0)

    # BVP1

    S_bvp1 = np.array([params.S_min for _ in range(N_BVP1)])
    t_bvp1 = np.random.uniform(0, params.T, size=N_BVP1)
    V_bvp1 = np.zeros(N_BVP1)

    # BVP2

    S_bvp2 = np.array([params.S_max for _ in range(N_BVP2)])
    t_bvp2 = np.random.uniform(0, params.T, size=N_BVP2)
    V_bvp2 = S_bvp2 - params.K * np.exp(-params.r * t_bvp2)

    # intérieur

    S = np.random.uniform(params.S_min, params.S_max, size=N_INT)
    t = np.random.uniform(0, params.T, size=N_INT)
    V = np.array([np.nan for _ in range(N_INT)])
