from __future__ import annotations

import array_api_compat.numpy as np
import matplotlib.pyplot as plt
import parallelproj

from array_api_compat import to_device
from utils import RadonDisk, RadonObjectSequence

# %%
import numpy.array_api as xp

# import array_api_compat.torch as xp

dev = "cpu"

# %%
# choose number of radial elements, number of views and angular coverage
num_rad = 201
phi_max = xp.pi
num_phi = int(0.5 * num_rad * xp.pi * (phi_max / xp.pi)) + 1

num_phi = num_phi // 1

r = xp.linspace(-3.1, 3.1, num_rad, device=dev)
phi = xp.linspace(0, phi_max, num_phi, endpoint=False, device=dev)
R, PHI = xp.meshgrid(r, phi, indexing="ij")
X0, X1 = xp.meshgrid(r, r, indexing="ij")
x = xp.linspace(float(xp.min(r)), float(xp.max(r)), 1001, device=dev)
X0hr, X1hr = xp.meshgrid(x, x, indexing="ij")

print(f"num rad:   {num_rad}")
print(f"phi max:   {180*phi_max/xp.pi:.2f} deg")
print(f"delta phi: {180*float(phi[1]-phi[0])/xp.pi:.2f} deg")


# %%
# define an object with known radon transform
disk0 = RadonDisk(xp, dev, 1.2)
disk0.amplitude = 1.0
disk0.s0 = 2.0

disk1 = RadonDisk(xp, dev, 0.3)
disk1.amplitude = 0.5
disk1.x1_offset = 0.7

disk2 = RadonDisk(xp, dev, 0.2)
disk2.amplitude = -1
disk2.x0_offset = -1.5

disk3 = RadonDisk(xp, dev, 0.14)
disk3.amplitude = -0.5
disk3.x1_offset = -0.7

disk4 = RadonDisk(xp, dev, 0.1)
disk4.amplitude = 1.0
disk4.x1_offset = -0.7

radon_object = RadonObjectSequence([disk0, disk1, disk2, disk3, disk4])

# %%
# analytic calculation of the randon transform
sino = radon_object.radon_transform(R, PHI)

# %%
# add Poisson noise
sens_sino = 19700 * xp.exp(-disk0.radon_transform(R, PHI))

noise_free_sino = sens_sino * sino
contam = xp.full(noise_free_sino.shape, 0.1 * xp.mean(noise_free_sino), device=dev)

emis_sino = xp.asarray(
    np.random.poisson(np.asarray(to_device(noise_free_sino + contam, "cpu"))).astype(
        float
    ),
    device=dev,
)
# emis_sino = noise_free_sino.copy()

# pre-correct sinogram
pre_corrected_sino = (emis_sino - contam) / sens_sino

# %%
# filtered back projection

# setup a discrete ramp filter
n_filter = r.shape[0]
r_shift = xp.arange(n_filter, device=dev, dtype=xp.float64) - n_filter // 2
f = xp.zeros(n_filter, device=dev)
f[r_shift != 0] = -1 / (xp.pi**2 * r_shift[r_shift != 0] ** 2)
f[(r_shift % 2) == 0] = 0
f[r_shift == 0] = 0.25

# %%
# ramp filter the sinogram in the radial direction
filtered_pre_corrected_sino = 1.0 * pre_corrected_sino

for i in range(num_phi):
    filtered_pre_corrected_sino[:, i] = xp.asarray(
        np.convolve(
            np.asarray(to_device(filtered_pre_corrected_sino[:, i], "cpu")),
            f,
            mode="same",
        ),
        device=dev,
    )

# %%
# define a parallel view projector

proj = parallelproj.ParallelViewProjector2D(
    (num_rad, num_rad),
    r,
    -phi,
    2 * float(xp.max(r)),
    (float(xp.min(r)), float(xp.min(r))),
    (float(r[1] - r[0]), float(r[1] - r[0])),
)

back_proj = proj.adjoint(pre_corrected_sino)
filtered_back_proj = proj.adjoint(filtered_pre_corrected_sino)


# %%
# visualize images
ext_img = [float(xp.min(r)), float(xp.max(r)), float(xp.min(r)), float(xp.max(r))]
ext_sino = [float(xp.min(r)), float(xp.max(r)), float(xp.min(phi)), float(xp.max(phi))]

fig, ax = plt.subplots(1, 4, figsize=(16, 4), tight_layout=True)
ax[0].imshow(
    np.asarray(to_device(radon_object.values(X0hr, X1hr).T, "cpu")),
    cmap="Greys",
    extent=ext_img,
    origin="lower",
)
ax[1].imshow(
    np.asarray(to_device(pre_corrected_sino.T, "cpu")),
    cmap="Greys",
    extent=ext_sino,
    origin="lower",
)
ax[2].imshow(
    np.asarray(to_device(back_proj.T, "cpu")),
    cmap="Greys",
    extent=ext_img,
    origin="lower",
)
ax[3].imshow(
    np.asarray(to_device(filtered_back_proj.T, "cpu")),
    cmap="Greys",
    extent=ext_img,
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
