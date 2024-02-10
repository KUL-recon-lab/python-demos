""" minimal example of ADMM for motion correction 
    simple 1D "denoising" problem with 3 shifted signals
    solutions to the subproblems can be computed analytically
    (quadratic regularization)

Learnings:
1. align "z to referenece z" to estimate motion
2. in the first iterations, rho controls whether the z's are close 
to the individual reconstructions or close to the average
-> to be able to estimate motion, we need "small" rho (e.g. 0.1)
-> rho too small also suboptimal, 
   because then the z's are too close to the individual reconstructions
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_powell


def cost_function(
    recon: np.ndarray,
    diag_fwd_op: np.ndarray,
    shifts: list[int],
    d: list[np.ndarray],
    bet: float,
) -> float:

    data_fidelities = np.zeros(len(shifts))

    for k, s in enumerate(shifts):
        exp_data = diag_fwd_op * np.roll(recon, s)
        data_fidelities[k] = 0.5 * ((exp_data - d[k]) ** 2).sum()

    recon_grad = np.roll(recon, -1) - recon
    prior = 0.5 * bet * (recon_grad**2).sum()

    return data_fidelities.sum() + prior


# %%
# input parameters
np.random.seed(1)

n = 201
noise_level = 0.15
num_iter = 800
beta = 1e0  # weight of the quad. prior
alignment_strategy = 1  # 1: z to z, 2: lam to z, 3: z+u to z+u, 4: lam to z+u
motion_update_period = 1

use_sub2_approx = False

# very big rho means that information between z is heavily shared (z more lambda like)
# -> not good when we want to align the z's to get a motion update
# very small row means that the z's stay very close to the ind. recons of the data
# which is better for motion estimation, but noise gets a problem
# -> there should be a sweet spot for rho, here this is around 1e-1
rho = 1e-2

# %%

# true shifts for the 3 gates
s1 = (n // 4) % n
s2 = (-n // 5) % n
s3 = 0

# errors for the shifts during recon
# 0 means use of true shifts during recon
s1_error = -s1  # 0 + n // 8  # set to -s1 for no shift modeling during recon
s2_error = -s2  # 0 + n // 16  # set to -s2 for no shift modeling during recon
s3_error = -s3  # 0  #  # set to -s3 for no shift modeling during recon

# x coordinate
x = np.linspace(-n // 2, n // 2, n)
# setup a simple diagonal operator A which is a multiplication with a triangle profile
# this can be inverted / transposed easily
diag_A = 1.2 * (x.max() - np.abs(x)) / x.max() + 0.1

# %%

# setup the true object
# simple rectangle
f = (np.abs(x) < (n / 8)).astype(float)
# gaussian profile
# f = np.exp(-x**2 / (n/16)**2)

f1 = np.roll(f, s1)
f2 = np.roll(f, s2)
f3 = np.roll(f, s3)

d1 = diag_A * f1 + np.random.normal(0, noise_level, n)
d2 = diag_A * f2 + np.random.normal(0, noise_level, n)
d3 = diag_A * f3 + np.random.normal(0, noise_level, n)


# %%
# setup the operator we need to analytically solve the ind. recons
# argmin_x (beta/2) || grad x||^2 + 1/2 ||x - d||^2

op = np.zeros((n, n))
i = np.arange(n)

op[i, i] = diag_A[i] ** 2 + 2 * beta
op[i, (i - 1) % n] = -beta
op[i, (i + 1) % n] = -beta

op_inv = np.linalg.inv(op)

# calculate the individual reconstructions
r1 = op_inv @ (diag_A * d1)
r2 = op_inv @ (diag_A * d2)
r3 = op_inv @ (diag_A * d3)


# %%
# setup the operators we need to analytically solve subproblem 2 (with or without approximation)
# argmin_lam (beta/2) || grad lambda||^2 + \sum_k rho/2 ||S_k lambda - (z_k + u_k)||^2

num_gates = 3

op2 = np.zeros((n, n))

op2[i, i] = rho * num_gates + 2 * beta
op2[i, (i - 1) % n] = -beta
op2[i, (i + 1) % n] = -beta

op2_inv = np.linalg.inv(op2)

op3 = np.zeros((n, n))
op3[i, i] = 1 + 2 * beta / rho
op3[i, (i - 1) % n] = -beta / rho
op3[i, (i + 1) % n] = -beta / rho

op3_inv = np.linalg.inv(op3)

# %%
# ADMM recons

# recon shifts
sr1 = s1 + s1_error
sr2 = s2 + s2_error
sr3 = s3 + s3_error

sr1s = [sr1]
sr2s = [sr2]
sr2s = [sr3]

# init variables
# lam = (np.roll(r1, -sr1) + np.roll(r2, -sr2) + np.roll(r3, -sr3)) / 3
lam = np.zeros(n)

u1 = np.zeros(n)
u2 = np.zeros(n)
u3 = np.zeros(n)

cost = np.zeros(num_iter)

for i in range(num_iter):
    # 1st sub-problem
    e1 = np.roll(lam, sr1) - u1
    e2 = np.roll(lam, sr2) - u2
    e3 = np.roll(lam, sr3) - u3

    # analytic solution of sub problem 1
    z1 = (diag_A * d1 + rho * e1) / (diag_A**2 + rho)
    z2 = (diag_A * d2 + rho * e2) / (diag_A**2 + rho)
    z3 = (diag_A * d3 + rho * e3) / (diag_A**2 + rho)

    # analytic solution of sub problem 2 - with or without approximation
    if use_sub2_approx:
        w = (
            np.roll(z1 + u1, -sr1) + np.roll(z2 + u2, -sr2) + np.roll(z3 + u3, -sr3)
        ) / num_gates
        lam = op3_inv @ w

    else:
        v = rho * (
            np.roll(z1 + u1, -sr1) + np.roll(z2 + u2, -sr2) + np.roll(z3 + u3, -sr3)
        )
        lam = op2_inv @ v

    # update of u
    u1 = u1 + z1 - np.roll(lam, sr1)
    u2 = u2 + z2 - np.roll(lam, sr2)
    u3 = u3 + z3 - np.roll(lam, sr3)

    # calculate the cost function
    cost[i] = cost_function(lam, diag_A, [sr1, sr2, sr3], [d1, d2, d3], beta)

    # update the shifts
    if (i + 1) % motion_update_period == 0:
        if alignment_strategy == 1:
            sr1 = np.argmin([((np.roll(z3, i) - z1) ** 2).sum() for i in range(n)])
            sr2 = np.argmin([((np.roll(z3, i) - z2) ** 2).sum() for i in range(n)])
        elif alignment_strategy == 2:
            sr1 = np.argmin([((np.roll(lam, i) - z1) ** 2).sum() for i in range(n)])
            sr2 = np.argmin([((np.roll(lam, i) - z2) ** 2).sum() for i in range(n)])
        elif alignment_strategy == 3:
            sr1 = np.argmin(
                [((np.roll(z3 + u3, i) - (z1 + u1)) ** 2).sum() for i in range(n)]
            )
            sr2 = np.argmin(
                [((np.roll(z3 + u3, i) - (z2 + u2)) ** 2).sum() for i in range(n)]
            )
        elif alignment_strategy == 4:
            sr1 = np.argmin(
                [((np.roll(lam, i) - (z1 + u1)) ** 2).sum() for i in range(n)]
            )
            sr2 = np.argmin(
                [((np.roll(lam, i) - (z2 + u2)) ** 2).sum() for i in range(n)]
            )
        else:
            raise ValueError("alignment_strategy not valid")

        sr1s.append(sr1)
        sr2s.append(sr2)

# %%
# calculate a reference recon using the powell optimizer

ref_recon = fmin_powell(
    cost_function,
    lam,
    args=(diag_A, [s1, s2, s3], [d1, d2, d3], beta),
    xtol=1e-5,
    ftol=1e-5,
)
ref_cost = cost_function(ref_recon, diag_A, [s1, s2, s3], [d1, d2, d3], beta)

# %%

fig, ax = plt.subplots(4, 3, figsize=(16, 8), tight_layout=True)
ax[0, 0].plot(x, f, "k", label="ground truth")
ax[0, 0].plot(x, lam, "r", label=r"$\lambda$")
ax[0, 0].plot(x, ref_recon, "g--", label="opt.sol.(Pow)")
ax[0, 0].set_title(
    f"beta={beta:.1e}, rho={rho:.1e}, n={num_iter}, align={alignment_strategy}",
    fontsize="medium",
)

ax[1, 0].plot(x, d1, "k", label=r"$d_1$")
ax[1, 0].plot(x, r1, "b", label=r"$r_1$")
ax[1, 0].plot(x, z1, "r", label=r"$z_1$")

ax[2, 0].plot(x, d2, "k", label=r"$d_2$")
ax[2, 0].plot(x, r2, "b", label=r"$r_2$")
ax[2, 0].plot(x, z2, "r", label=r"$z_2$")
ax[3, 0].plot(x, d3, "k", label=r"$d_3$")
ax[3, 0].plot(x, r3, "b", label=r"$r_3$")
ax[3, 0].plot(x, z3, "r", label=r"$z_3$")

ax[1, 1].plot(x, d1, "k", label=r"$d_1$")
ax[1, 1].plot(x, r1, "b", label=r"$r_1$")
ax[1, 1].plot(x, z1 + rho * u1, "r", label=r"$z_1 + \rho u_1$")
ax[2, 1].plot(x, d2, "k", label=r"$d_2$")
ax[2, 1].plot(x, r2, "b", label=r"$r_2$")
ax[2, 1].plot(x, z2 + rho * u2, "r", label=r"$z_2 + \rho u_2$")
ax[3, 1].plot(x, d3, "k", label=r"$d_3$")
ax[3, 1].plot(x, r3, "b", label=r"$r_3$")
ax[3, 1].plot(x, z3 + rho * u3, "r", label=r"$z_3 + \rho u_3$")

ax[1, 2].plot(x, d1, "k", label=r"$d_1$")
ax[1, 2].plot(x, z1 + u1, "r", label=r"$z_1 + u_1$")
ax[2, 2].plot(x, d2, "k", label=r"$d_2$")
ax[2, 2].plot(x, z2 + u2, "r", label=r"$z_2 + u_2$")
ax[3, 2].plot(x, d3, "k", label=r"$d_3$")
ax[3, 2].plot(x, z3 + u3, "r", label=r"$z_3 + u_3$")

for axx in ax[1:, :].ravel():
    axx.legend()
    axx.grid(ls=":")
ax[0, 0].legend()
ax[0, 0].grid(ls=":")

ymin = d3.min()
ymax = d3.max()

for axx in ax[1:, :-1].ravel():
    axx.set_ylim(ymin, ymax)
ax[0, 0].set_ylim(ymin, ymax)

# plot the estimated shifts
ax[0, 1].plot(sr1s, label="shift 1")
ax[0, 1].plot(sr2s, label="shift 2")

imin = min(10, n)
ax[0, 2].plot(sr1s[:imin], label="shift 1")
ax[0, 2].plot(sr2s[:imin], label="shift 2")

for axx in ax[0, 1:]:
    axx.legend()
    axx.grid(ls=":")
    axx.axhline(s1, color=plt.cm.tab10(0), ls="--")
    axx.axhline(s2, color=plt.cm.tab10(1), ls="--")

fig.show()

# %%
# plot the cost function
fig2, ax2 = plt.subplots(tight_layout=True)
ax2.semilogx(np.arange(1, num_iter + 1), cost, "r", label="cost(lambda)")
ax2.axhline(ref_cost, color="g", ls="--", label="optimal cost (Powell)")
ax2.set_title("cost function")
ax2.grid(ls=":")
fig2.show()

print(
    f"rho: {rho:.1e}, num_iter: {num_iter}, cost: {cost[-1]:.4e}, ref_cost: {ref_cost:.4e}"
)
