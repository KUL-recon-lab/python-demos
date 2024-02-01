from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
from utils import RadonDisk, RadonObjectSequence, RotationBasedProjector

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
# setup r and phi coordinates
r = np.linspace(-3.1, 3.1, 151)
num_phi = int(0.5 * r.shape[0] * np.pi) + 1
phi = np.linspace(0, 1 * np.pi, num_phi, endpoint=False)
PHI, R = np.meshgrid(phi, r, indexing="ij")
x = np.linspace(r.min(), r.max(), 351)
X0, X1 = np.meshgrid(r, r, indexing="ij")
X0hr, X1hr = np.meshgrid(x, x, indexing="ij")

# %%
# analytic calculation of the randon transform
sens_sino = 197 * np.exp(-disk0.radon_transform(R, PHI))
noise_free_sino = gaussian_filter(
    sens_sino * radon_object.radon_transform(R, PHI), (0, 1.2)
)
contam = np.full(noise_free_sino.shape, 0.1 * noise_free_sino.mean())

# %%
# add Poisson noise
emis_sino = np.random.poisson(noise_free_sino + contam).astype(float)

# pre-corrected sinogram
pre_corrected_sino = (emis_sino - contam) / sens_sino

print(f"counts: {(emis_sino.sum() / 1e6):.1f} million")

# %%
# filtered back projection
proj = RotationBasedProjector(phi, r)

# see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4341983/
# for discrete implementation of ramp filter
n_filter = r.shape[0]

r_shift = np.arange(n_filter) - n_filter // 2
f = np.zeros(n_filter)
f[r_shift != 0] = -1 / (np.pi**2 * r_shift[r_shift != 0] ** 2)
f[(r_shift % 2) == 0] = 0
f[r_shift == 0] = 0.25

proj.filter = f
filtered_back_projs = proj.backproject(pre_corrected_sino)
filtered_back_proj = filtered_back_projs.mean(axis=0)

# %%
# run MLEM

proj.filter = None

sig_res = 0
num_iter = 200

x_mlem = 0.05 * np.ones(X0.shape, dtype=float)
sens_img = proj.backproject(gaussian_filter(sens_sino, (0, sig_res))).sum(axis=0)


x_mlems = np.zeros((num_iter,) + x_mlem.shape)
exps = np.zeros((num_iter,) + emis_sino.shape)
ratios = np.zeros((num_iter,) + emis_sino.shape)
ratio_backs = np.zeros((num_iter,) + x_mlem.shape)
update_imgs = np.zeros((num_iter,) + x_mlem.shape)

for i in range(num_iter):
    print(i, end="\r")
    x_mlems[i, ...] = x_mlem
    exp = (
        gaussian_filter(sens_sino * proj.forwardproject(x_mlem), (0, sig_res)) + contam
    )
    ratio = emis_sino / exp
    ratio_back = proj.backproject(sens_sino * gaussian_filter(ratio, (0, sig_res))).sum(
        axis=0
    )

    update_img = ratio_back / sens_img
    x_mlem *= update_img

    exps[i, ...] = exp
    ratios[i, ...] = ratio
    ratio_backs[i, ...] = ratio_back
    update_imgs[i, ...] = update_img


# %%
def _update_animation(n):
    i = n // 5
    k = n % 5

    if k == 0:
        im10.set_data(x_mlems[i, ...].T)
    elif k == 1:
        im11.set_data(exps[i, ...])
    elif k == 2:
        im02.set_data(ratios[i, ...])
    elif k == 3:
        im12.set_data(ratio_backs[i, ...].T)
    elif k == 4:
        im03.set_data(update_imgs[i, ...].T)

    ax[1, 0].set_title(f"$x$ it:{i+1:03}")
    ax[1, 1].set_title(f"$Ax + s$ it:{i+1:03}")
    ax[0, 2].set_title(f"$y / (Ax + s)$ it:{i+1:03}")
    ax[1, 2].set_title(f"$A^T (y / (Ax + s))$ it:{i+1:03}")
    ax[0, 3].set_title(f"$A^T (y / (Ax + s)) / A^T 1$ it:{i+1:03}")


# %%

vmax = gaussian_filter(x_mlem, 1.2).max()

its = [5, 10, 20, 50, 100, 200]

fig, ax = plt.subplots(2, 6, figsize=(18, 6), tight_layout=True)

for i, it in enumerate(its):
    ax[0, i].imshow(
        x_mlems[it - 1, ...].T, cmap="Greys", origin="lower", vmin=0, vmax=vmax
    )
    ax[1, i].imshow(
        gaussian_filter(x_mlems[it - 1, ...].T, 1.2),
        cmap="Greys",
        origin="lower",
        vmin=0,
        vmax=vmax,
    )

    ax[0, i].set_title(f"MLEM {it:03} it.", fontsize="small")
    ax[1, i].set_title(f"smoothed MLEM {it:03} it.", fontsize="small")

for axx in ax.ravel():
    axx.set_axis_off()

fig.show()

# %%

# i = 0
#
# fig, ax = plt.subplots(2, 4, figsize=(16, 8), tight_layout=True)
#
# im01 = ax[0, 1].imshow(emis_sino, cmap="Greys", origin="lower")
# im10 = ax[1, 0].imshow(
#    x_mlems[i, ...].T, cmap="Greys", origin="lower", vmin=0, vmax=x_mlem.max()
# )
# im11 = ax[1, 1].imshow(
#    exps[i, ...], cmap="Greys", origin="lower", vmin=0, vmax=emis_sino.max()
# )
# im02 = ax[0, 2].imshow(ratios[i, ...], cmap="Greys", origin="lower", vmin=0.5, vmax=1.5)
# im12 = ax[1, 2].imshow(
#    ratio_backs[i, ...].T,
#    cmap="Greys",
#    origin="lower",
#    vmin=0,
#    vmax=sens_img.max(),
# )
# im03 = ax[0, 3].imshow(
#    update_imgs[i, ...].T, cmap="Greys", origin="lower", vmin=0.5, vmax=1.5
# )
# im13 = ax[1, 3].imshow(sens_img.T, cmap="Greys", origin="lower")
#
# for axx in ax.ravel():
#    axx.set_axis_off()
#
# ax[0, 1].set_title("emission sinogram y")
# ax[1, 0].set_title(f"$x$ it:{i+1:03}")
# ax[1, 1].set_title(f"$Ax + s$ it:{i+1:03}")
# ax[0, 2].set_title(f"$y / (Ax + s)$ it:{i+1:03}")
# ax[1, 2].set_title(f"$A^T (y / (Ax + s))$ it:{i+1:03}")
# ax[0, 3].set_title(f"$A^T (y / (Ax + s)) / A^T 1 it:{i+1:03}$")
# ax[1, 3].set_title(f"$A^T 1$")
#
## create animation
# ani = animation.FuncAnimation(
#    fig, _update_animation, num_iter * 5, interval=5, blit=False, repeat=False
# )
#
## save animation to gif
# ani.save("mlem_animation.mp4", writer=animation.FFMpegWriter(fps=10))
#
#
# fig.show()

# %%
## %%
## visualize images
# fig, ax = plt.subplots(2, 3, figsize=(12, 8), tight_layout=True)
# ax[0, 0].imshow(
#    radon_object.values(X0hr, X1hr).T,
#    cmap="Greys",
#    extent=[r.min(), r.max(), r.min(), r.max()],
#    origin="lower",
# )
# ax[1, 0].imshow(
#    emis_sino,
#    cmap="Greys",
#    extent=[r.min(), r.max(), phi.min(), phi.max()],
#    origin="lower",
# )
#
# ax[0, 1].imshow(
#    filtered_back_proj.T,
#    cmap="Greys",
#    extent=[r.min(), r.max(), r.min(), r.max()],
#    origin="lower",
# )
#
# ax[0, 2].imshow(
#    x_mlem.T,
#    cmap="Greys",
#    extent=[r.min(), r.max(), r.min(), r.max()],
#    origin="lower",
# )
#
# ax[1, 1].imshow(
#    gaussian_filter(filtered_back_proj, 1.2).T,
#    cmap="Greys",
#    extent=[r.min(), r.max(), r.min(), r.max()],
#    origin="lower",
# )
#
# ax[1, 2].imshow(
#    gaussian_filter(x_mlem, 1.2).T,
#    cmap="Greys",
#    extent=[r.min(), r.max(), r.min(), r.max()],
#    origin="lower",
# )
#
# for axx in ax.ravel():
#    axx.set_xlabel(r"$x_0$")
#    axx.set_ylabel(r"$x_1$")
#
# ax[1, 0].set_xlabel(r"$s$")
# ax[1, 0].set_ylabel(r"$\phi$")
#
# ax[0, 0].set_title(r"object ($x$)")
# ax[1, 0].set_title(r"emission sinogram $y$")
# ax[0, 1].set_title("filtered back projection (FBP)")
# ax[0, 2].set_title("MLEM 100 iterations")
# ax[1, 1].set_title("smoothed filtered projection")
# ax[1, 2].set_title("smoothed MLEM 100 iterations")
#
# fig.show()
