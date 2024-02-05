from __future__ import annotations

import array_api_compat.numpy as np
import matplotlib.pyplot as plt
import parallelproj

from array_api_compat import to_device
from utils import RadonDisk, RadonObjectSequence

# %%
import numpy.array_api as xp

dev = "cpu"

add_noise = True
total_counts = 1e7
num_iter = 50
mu = 0.1

# %%
# choose number of radial elements, number of views and angular coverage
num_rad = 201
phi_max = xp.pi
num_phi = int(0.5 * num_rad * xp.pi * (phi_max / xp.pi)) + 1

num_phi = num_phi // 1

r = xp.linspace(-30, 30, num_rad, device=dev, dtype=xp.float32)
phi = xp.linspace(0, phi_max, num_phi, endpoint=False, device=dev, dtype=xp.float32)
R, PHI = xp.meshgrid(r, phi, indexing="ij")
X0, X1 = xp.meshgrid(r, r, indexing="ij")
x = xp.linspace(float(xp.min(r)), float(xp.max(r)), 1001, device=dev, dtype=xp.float32)
X0hr, X1hr = xp.meshgrid(x, x, indexing="ij")

print(f"num rad:   {num_rad}")
print(f"phi max:   {180*phi_max/xp.pi:.2f} deg")
print(f"delta phi: {180*float(phi[1]-phi[0])/xp.pi:.2f} deg")


# %%
# define an object with known radon transform
disk0 = RadonDisk(xp, dev, 8.0)
disk0.amplitude = 1.0
disk0.s0 = 3.0

disk1 = RadonDisk(xp, dev, 2.0)
disk1.amplitude = 0.5
disk1.x1_offset = 4.67

disk2 = RadonDisk(xp, dev, 1.4)
disk2.amplitude = -0.5
disk2.x0_offset = -10.0

disk3 = RadonDisk(xp, dev, 0.93)
disk3.amplitude = -0.5
disk3.x1_offset = -4.67

disk4 = RadonDisk(xp, dev, 0.67)
disk4.amplitude = 1.0
disk4.x1_offset = -4.67

radon_object = RadonObjectSequence([disk0, disk1, disk2, disk3, disk4])

# %%
# analytic calculation of the randon transform
sino = radon_object.radon_transform(R, PHI)

# %%
# add Poisson noise
sens_sino = xp.exp(-mu * disk0.radon_transform(R, PHI))

contam = xp.full(sino.shape, 0.1 * xp.mean(sens_sino * sino), device=dev)

emis_sino = sens_sino * sino + contam
count_fac = total_counts / float(xp.sum(emis_sino))

emis_sino *= count_fac
contam *= count_fac

if add_noise:
    emis_sino = xp.asarray(
        np.random.poisson(np.asarray(to_device(emis_sino, "cpu"))).astype(np.float32),
        device=dev,
    )

# pre-correct sinogram
pre_corrected_sino = (emis_sino - contam) / sens_sino

# %%
# filtered back projection

# setup a discrete ramp filter
n_filter = r.shape[0]
r_shift = xp.arange(n_filter, device=dev, dtype=xp.float64) - n_filter // 2
f = xp.zeros(n_filter, device=dev, dtype=xp.float64)
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
# define a projector and back project the pre-corrected and filtered and pre-corrected sinogram

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
# run MLEM

x_mlem = xp.ones((num_rad, num_rad), device=dev, dtype=xp.float32)
sens_img = proj.adjoint(sens_sino)

for i in range(num_iter):
    print(i, end="\r")
    exp = sens_sino * proj(x_mlem) + contam
    ratio = emis_sino / exp
    ratio_back = proj.adjoint(sens_sino * ratio)

    update_img = ratio_back / sens_img
    x_mlem *= update_img


# %%
# visualize images
ext_img = [float(xp.min(r)), float(xp.max(r)), float(xp.min(r)), float(xp.max(r))]
ext_sino = [float(xp.min(r)), float(xp.max(r)), float(xp.min(phi)), float(xp.max(phi))]

fig, ax = plt.subplots(1, 5, figsize=(15, 3), tight_layout=True)
ax[0].imshow(
    np.asarray(to_device(radon_object.values(X0hr, X1hr).T, "cpu")),
    cmap="Greys",
    extent=ext_img,
    origin="lower",
)
ax[1].imshow(
    np.asarray(to_device(emis_sino.T, "cpu")),
    cmap="Greys",
    aspect=20,
    extent=ext_sino,
    origin="lower",
)

ax[2].imshow(
    np.asarray(to_device(pre_corrected_sino.T, "cpu")),
    cmap="Greys",
    aspect=20,
    extent=ext_sino,
    origin="lower",
)
ax[3].imshow(
    np.asarray(to_device(filtered_back_proj.T, "cpu")),
    cmap="Greys",
    extent=ext_img,
    origin="lower",
)
ax[4].imshow(
    np.asarray(to_device(x_mlem.T, "cpu")),
    cmap="Greys",
    extent=ext_img,
    origin="lower",
)
ax[0].set_xlabel(r"$x_0$")
ax[0].set_ylabel(r"$x_1$")
ax[1].set_xlabel(r"$s$")
ax[1].set_ylabel(r"$\phi$")
ax[2].set_xlabel(r"$s$")
ax[2].set_ylabel(r"$\phi$")
ax[3].set_xlabel(r"$x_0$")
ax[3].set_ylabel(r"$x_1$")
ax[4].set_xlabel(r"$x_0$")
ax[4].set_ylabel(r"$x_1$")

ax[0].set_title("object", fontsize="medium")
ax[1].set_title("emission sino", fontsize="medium")
ax[2].set_title("pre-corrected sino", fontsize="medium")
ax[3].set_title("filtered back projection", fontsize="medium")
ax[4].set_title(f"MLEM {num_iter} it.", fontsize="medium")

fig.show()
