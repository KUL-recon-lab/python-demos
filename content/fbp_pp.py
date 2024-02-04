from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import parallelproj

from utils import RadonDisk, RadonObjectSequence, RotationBasedProjector

# %%
# choose number of radial elements, number of views and angular coverage
num_rad = 201
phi_max = np.pi
num_phi = int(0.5 * num_rad * np.pi * (phi_max / np.pi)) + 1

num_phi = num_phi // 1

r = np.linspace(-3.1, 3.1, num_rad)
phi = np.linspace(0, phi_max, num_phi, endpoint=False)
R, PHI = np.meshgrid(r, phi, indexing="ij")
x = np.linspace(r.min(), r.max(), 1001)
X0, X1 = np.meshgrid(r, r, indexing="ij")
X0hr, X1hr = np.meshgrid(x, x, indexing="ij")

print(f"num rad:   {num_rad}")
print(f"phi max:   {180*phi_max/np.pi:.2f} deg")
print(f"delta phi: {180*(phi[1]-phi[0])/np.pi:.2f} deg")


# %%
# define an object with known radon transform
disk0 = RadonDisk(1.2)
disk0.amplitude = 1.0
disk0.s0 = 2.0

disk1 = RadonDisk(0.3)
disk1.amplitude = 0.5
disk1.x1_offset = 0.7

disk2 = RadonDisk(0.2)
disk2.amplitude = -1
disk2.x0_offset = -1.5

disk3 = RadonDisk(0.14)
disk3.amplitude = -0.5
disk3.x1_offset = -0.7

disk4 = RadonDisk(0.1)
disk4.amplitude = 1.0
disk4.x1_offset = -0.7

radon_object = RadonObjectSequence([disk0, disk1, disk2, disk3, disk4])

# %%
# analytic calculation of the randon transform
sino = radon_object.radon_transform(R, PHI)

# %%
# add Poisson noise
sens_sino = 197 * np.exp(-disk0.radon_transform(R, PHI))

noise_free_sino = sens_sino * sino
contam = np.full(noise_free_sino.shape, 0.1 * noise_free_sino.mean())

# emis_sino = np.random.poisson(noise_free_sino + contam).astype(float)
emis_sino = noise_free_sino.copy()

# pre-correct sinogram
pre_corrected_sino = (emis_sino - contam) / sens_sino

# %%
# filtered back projection

# see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4341983/
# for discrete implementation of ramp filter

n_filter = r.shape[0]

r_shift = np.arange(n_filter) - n_filter // 2
f = np.zeros(n_filter)
f[r_shift != 0] = -1 / (np.pi**2 * r_shift[r_shift != 0] ** 2)
f[(r_shift % 2) == 0] = 0
f[r_shift == 0] = 0.25

# %%
# ramp filter the sinogram in the radial direction
filtered_pre_corrected_sino = pre_corrected_sino.copy()

for i in range(num_phi):
    filtered_pre_corrected_sino[:, i] = np.convolve(
        filtered_pre_corrected_sino[:, i], f, mode="same"
    )

# %%
# define a parallel view projector

proj = parallelproj.ParallelViewProjector2D(
    (num_rad, num_rad),
    r,
    -phi,
    2 * r.max(),
    (r.min(), r.min()),
    (r[1] - r[0], r[1] - r[0]),
)

back_proj = proj.adjoint(sino)
filtered_back_proj = proj.adjoint(filtered_pre_corrected_sino)


# %%
# visualize images
fig, ax = plt.subplots(1, 4, figsize=(16, 4), tight_layout=True)
ax[0].imshow(
    radon_object.values(X0hr, X1hr).T,
    cmap="Greys",
    extent=[r.min(), r.max(), r.min(), r.max()],
    origin="lower",
)
ax[1].imshow(
    pre_corrected_sino.T,
    cmap="Greys",
    extent=[r.min(), r.max(), phi.min(), phi.max()],
    origin="lower",
)
ax[2].imshow(
    back_proj.T,
    cmap="Greys",
    extent=[r.min(), r.max(), r.min(), r.max()],
    origin="lower",
)
ax[3].imshow(
    filtered_back_proj.T,
    cmap="Greys",
    extent=[r.min(), r.max(), r.min(), r.max()],
    origin="lower",
)


ax[0].set_xlabel(r"$x_0$")
ax[0].set_ylabel(r"$x_1$")
ax[1].set_xlabel(r"$s$")
ax[1].set_ylabel(r"$\phi$")
ax[2].set_xlabel(r"$x_0$")
ax[2].set_ylabel(r"$x_1$")
ax[3].set_xlabel(r"$x_0$")
ax[3].set_ylabel(r"$x_1$")

ax[0].set_title(r"object ($x$)")
ax[1].set_title(r"Radon transform of object ($y = Rx$)")
ax[2].set_title(r"back projection ($R^T y$)")
ax[3].set_title(r"filtered back projection ($R^{-1} y$)")

fig.show()
