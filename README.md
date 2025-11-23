# Physics-Informed Neural Network for Black–Scholes European Call Pricing

## Overview

This project implements a **Physics-Informed Neural Network (PINN)** to solve the **Black–Scholes Partial Differential Equation (PDE)** for pricing a **European call option**.

The PINN learns the pricing function  
V(t, S)  
by enforcing:

- the Black–Scholes PDE (via automatic differentiation),
- the terminal condition,
- the boundary conditions.

This approach integrates financial theory directly inside neural network training.

---

# Mathematical Background

We solve the Black–Scholes PDE:

∂V/∂t
+ (1/2) σ² S² ∂²V/∂S²
+ r S ∂V/∂S
– r V = 0

with:

Terminal condition (IVP):
V(T, S) = max(S – K, 0)

Boundary condition 1 (BVP1):
V(t, S_min) = 0

Boundary condition 2 (BVP2):
V(t, S_max) = S_max – K e^{–rt}

PINN training minimizes a weighted sum of:
- IVP loss,
- boundary losses,
- PDE residual loss.

---

# PINN Model

The PINN is a fully connected MLP:

Input: (t, S)
Output: V(t, S)

Configurable via dataclass:

input_dim = 2  
hidden_dim = 64  
num_hidden_layers = 4  
activation = "tanh"

The PINNCall class constructs the architecture and performs the forward pass.

---

# Sampling Strategy

Sampling domains:

1. IVP (t = T) → enforce payoff  
2. BVP1 (S = S_min) → V = 0  
3. BVP2 (S = S_max) → V = S – K e^{–rt}  
4. Interior points (random t,S) → enforce PDE residual

---

# Loss Functions

IVP loss: MSE(V_pred, max(S-K,0))  
BVP losses: MSE(V_pred, V_true)  
PDE loss: MSE(f_theta(t,S), 0)

Total loss = IVP + BVP + beta * PDE

---

# Training

The Trainer handles:

- losses
- autodiff
- Adam optimizer
- gradient steps
- logging
- history export

Usage:

trainer = TrainerEuropeanCall(model, bs_params, config)  
trainer.train()

---

# How to Run

pip install torch numpy  
python src/main.py

---

# Expected Output

After training:

- PDE residual approx. zero  
- boundary and terminal conditions satisfied  
- pricing surface close to analytical Black–Scholes  
- smooth V(t,S) surface

---
