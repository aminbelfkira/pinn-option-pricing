def bs_pde_call(
    t: float,
    S: float,
    V: float,
    r: float,
    sigma: float,
    dV_t: float,
    dV_S: float,
    d2V_S2: float,
):

    return dV_t + 0.5 * sigma**2 * S**2 * d2V_S2 + r * S * dV_S - r * V

