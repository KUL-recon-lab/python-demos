""" minimal example of ADMM for motion correction 
    simple 1D "denoising" problem with 3 shifted signals
    solutions to the subproblems can be computed analytically

TODO: use simple diagonal operator A (multiplication with triangle profile)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# %%
# input parameters
np.random.seed(2)

n = 201
noise_level = 0.1
rho = 1e-1
num_iter = 100

# true shifts for the 3 gates
s1 = (n // 10) % n
s2 = (-n // 5) % n
s3 = 0

# errors for the shifts during recon
# 0 means use of true shifts during recon
s1_error = 0  # + n // 16  # set to -s1 for no shift modeling during recon
s2_error = 0  # + n // 16  # set to -s2 for no shift modeling during recon
s3_error = 0  #  # set to -s3 for no shift modeling during recon

# x coordinate
x = np.linspace(-n // 2, n // 2, n)
# setup a simple diagonal operator A which is a multiplication with a triangle profile
# this can be inverted / transposed easily
diag_A = 1.2 * (x.max() - np.abs(x)) / x.max() + 0.1

# sigma for Gaussian filter to emulate regularization
sig_lam = 0.0

motion_update_period = 100000000
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
# ADMM recons

# recon shifts
sr1 = s1 + s1_error
sr2 = s2 + s2_error
sr3 = s3 + s3_error

sr1s = [sr1]
sr2s = [sr2]
sr2s = [sr3]

# init variables
lam = (np.roll(d1, -sr1) + np.roll(d2, -sr2) + np.roll(d3, -sr3)) / 3
# lam = np.zeros(n)
# lam = f

u1 = np.zeros(n)
u2 = np.zeros(n)
u3 = np.zeros(n)

for i in range(num_iter):
    # 1st sub-problem
    e1 = np.roll(lam, sr1) - u1
    e2 = np.roll(lam, sr2) - u2
    e3 = np.roll(lam, sr3) - u3

    # analytic solution of sub problem 1
    z1 = (diag_A * d1 + rho * e1) / (diag_A**2 + rho)
    z2 = (diag_A * d2 + rho * e2) / (diag_A**2 + rho)
    z3 = (diag_A * d3 + rho * e3) / (diag_A**2 + rho)

    # sub-problem 2
    lam = (np.roll(z1 + u1, -sr1) + np.roll(z2 + u2, -sr2) + np.roll(z3 + u3, -sr3)) / 3

    #### HACK to emulate regularization on lambda -> not at all correct!
    # if sig_lam > 0:
    #    lam = gaussian_filter1d(lam, 5.0, mode="wrap")

    # update of u
    u1 = u1 + z1 - np.roll(lam, sr1)
    u2 = u2 + z2 - np.roll(lam, sr2)
    u3 = u3 + z3 - np.roll(lam, sr3)

    # update the shifts
    if (i + 1) % motion_update_period == 0:
        sr1 = np.argmin([((np.roll(z3, i) - z1) ** 2).sum() for i in range(n)])
        sr2 = np.argmin([((np.roll(z3, i) - z2) ** 2).sum() for i in range(n)])

        sr1s.append(sr1)
        sr2s.append(sr2)

# %%

fig, ax = plt.subplots(4, 3, figsize=(16, 8), tight_layout=True)
ax[0, 0].plot(x, f, "k", label="ground truth")
ax[0, 0].plot(x, lam, "r", label=r"$\lambda$")
ax[0, 0].set_title(f"rho={rho:.1e}, n={num_iter}", fontsize="medium")

ax[1, 0].plot(x, d1, "k", label=r"$d_1$")
ax[1, 0].plot(x, d1 / diag_A, "b", label=r"$A^{-1} d_1$")
ax[1, 0].plot(x, z1, "r", label=r"$z_1$")

ax[2, 0].plot(x, d2, "k", label=r"$d_2$")
ax[2, 0].plot(x, d2 / diag_A, "b", label=r"$A^{-1} d_2$")
ax[2, 0].plot(x, z2, "r", label=r"$z_2$")
ax[3, 0].plot(x, d3, "k", label=r"$d_3$")
ax[3, 0].plot(x, d3 / diag_A, "b", label=r"$A^{-1} d_3$")
ax[3, 0].plot(x, z3, "r", label=r"$z_3$")

ax[1, 1].plot(x, d1, "k", label=r"$d_1$")
ax[1, 1].plot(x, d1 / diag_A, "b", label=r"$A^{-1} d_1$")
ax[1, 1].plot(x, z1 + rho * u1, "r", label=r"$z_1 + \rho u_1$")
ax[2, 1].plot(x, d2, "k", label=r"$d_2$")
ax[2, 1].plot(x, d2 / diag_A, "b", label=r"$A^{-1} d_2$")
ax[2, 1].plot(x, z2 + rho * u2, "r", label=r"$z_2 + \rho u_2$")
ax[3, 1].plot(x, d3, "k", label=r"$d_3$")
ax[3, 1].plot(x, d3 / diag_A, "b", label=r"$A^{-1} d_3$")
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

# ymin = min([axx.get_ylim()[0] for axx in ax[:, :-1].ravel()])
# ymax = max([axx.get_ylim()[1] for axx in ax[:, :-1].ravel()])

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