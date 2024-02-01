from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.transforms as mtransforms

from utils import RadonDisk, RadonObjectSequence, RotationBasedProjector

# %%
# choose number of radial elements, number of views and angular coverage
num_rad = 201
phi_max = np.pi
num_phi = int(0.5 * num_rad * np.pi * (phi_max / np.pi)) + 1

num_phi = num_phi // 1

r = np.linspace(-3.1, 3.1, num_rad)
phi = np.linspace(0, phi_max, num_phi, endpoint=False)
PHI, R = np.meshgrid(phi, r, indexing="ij")
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

# print(f"counts: {(emis_sino.sum() / 1e6):.1f} million")

## %%
## simulate "dead pixel" in detector
# pre_corrected_sino[:, ::16] = 0

# %%
# back projection
proj = RotationBasedProjector(phi, r)
back_projs = proj.backproject(pre_corrected_sino)
back_proj = back_projs.mean(axis=0)

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

proj.filter = f
filtered_back_projs = proj.backproject(pre_corrected_sino)
filtered_back_proj = filtered_back_projs.mean(axis=0)

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
    pre_corrected_sino,
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


# %%
def _update_animation(i):
    p1.set_ydata(pre_corrected_sino[i, :])
    p2.set_ydata(np.convolve(pre_corrected_sino[i, :], f, mode="same"))

    img3.set_data(back_projs[i, ...].T)
    img4.set_data(filtered_back_projs[i, ...].T)
    img5.set_data(back_projs[: (i + 1), ...].mean(axis=0).T)
    img6.set_data(filtered_back_projs[: (i + 1), ...].mean(axis=0).T)

    ax1.set_title(
        f"projection profile {(i+1):03} - $\phi$ = {180*phi[i]/np.pi:.1f}$^\circ$",
        fontsize="medium",
    )
    ax2.set_title(
        f"filtered projection profile {(i+1):03} - $\phi$ = {180*phi[i]/np.pi:.1f}",
        fontsize="medium",
    )
    ax3.set_title(f"back projection of profile {(i+1):03}", fontsize="medium")
    ax4.set_title(f"filtered back projection of profile {(i+1):03}", fontsize="medium")
    ax5.set_title(f"mean of first {(i+1):03} back projections", fontsize="medium")
    ax6.set_title(
        f"mean of first {(i+1):03} filtered back projections", fontsize="medium"
    )

    t = mtransforms.Affine2D().rotate_around(0, 0, phi[i])
    for ar in arr:
        ar.set_transform(t + ax0.transData)
    for k, s in enumerate(ss):
        d0 = s * np.cos(phi[i])
        d1 = s * np.sin(phi[i])
        ann[k].set_position(
            (1.2 * R * np.sin(phi[i]) + d0, -1.2 * R * np.cos(phi[i]) + d1)
        )
        ann[k].xy = (1.2 * R * np.sin(phi[i]) + d0, -1.2 * R * np.cos(phi[i]) + d1)

    return (p1, p2, img3, img4, img5, img6, arr, ann)


# %%
# animated random transform and sinogram

fig2 = plt.figure(tight_layout=True, figsize=(12, 8))
gs = gridspec.GridSpec(4, 6)

ax0 = fig2.add_subplot(gs[:2, :2])
ax1 = fig2.add_subplot(gs[2, :2])
ax2 = fig2.add_subplot(gs[3, :2])
ax3 = fig2.add_subplot(gs[:2, 2:4])
ax4 = fig2.add_subplot(gs[:2, 4:6])
ax5 = fig2.add_subplot(gs[2:4, 2:4])
ax6 = fig2.add_subplot(gs[2:4, 4:6])

ax0.imshow(
    radon_object.values(X0hr, X1hr).T,
    cmap="Greys",
    extent=[r.min(), r.max(), r.min(), r.max()],
    origin="lower",
)

i = 0
R = 1.5

arr = []
ann = []

ss = np.linspace(-1.5 * disk0.radius, 1.5 * disk0.radius, 9)

p1 = ax1.plot(r, pre_corrected_sino[i, :], color="k")[0]
ax1.set_ylim(pre_corrected_sino.min(), pre_corrected_sino.max())
ax1.grid(ls=":")

p2 = ax2.plot(r, np.convolve(pre_corrected_sino[i, :], f, mode="same"), color="k")[0]
ax2.set_ylim(filtered_back_projs.min(), filtered_back_projs.max())
ax2.grid(ls=":")


for s in ss:
    ax1.axvline(s, color="r", lw=0.5)
    ax2.axvline(s, color="r", lw=0.5)
    d0 = s * np.cos(phi[i])
    d1 = s * np.sin(phi[i])
    arr.append(
        ax0.arrow(
            R * np.sin(phi[i]) + d0,
            -R * np.cos(phi[i]) + d1,
            -2 * R * np.sin(phi[i]),
            2 * R * np.cos(phi[i]),
            color="r",
            width=0.001,
            head_width=0.1,
        )
    )
    ann.append(
        ax0.annotate(
            f"{s:.1f}",
            (1.2 * R * np.sin(phi[i]) + d0, -1.2 * R * np.cos(phi[i]) + d1),
            color="r",
            fontsize="small",
            ha="center",
            va="center",
            annotation_clip=True,
        )
    )

img3 = ax3.imshow(
    back_projs[i, ...].T,
    cmap="Greys",
    extent=[r.min(), r.max(), r.min(), r.max()],
    vmin=0,
    vmax=back_projs.max(),
    origin="lower",
)
img4 = ax4.imshow(
    filtered_back_projs[i, ...].T,
    cmap="Greys",
    extent=[r.min(), r.max(), r.min(), r.max()],
    vmin=filtered_back_projs.min(),
    vmax=filtered_back_projs.max(),
    origin="lower",
)
img5 = ax5.imshow(
    back_projs[: (i + 1), ...].mean(axis=0).T,
    cmap="Greys",
    extent=[r.min(), r.max(), r.min(), r.max()],
    origin="lower",
    vmin=back_proj.min(),
    vmax=1.05 * back_proj.max(),
)
img6 = ax6.imshow(
    filtered_back_projs[: (i + 1), ...].mean(axis=0).T,
    cmap="Greys",
    extent=[r.min(), r.max(), r.min(), r.max()],
    origin="lower",
    vmin=filtered_back_proj.min(),
    vmax=1.05 * filtered_back_proj.max(),
)

ax0.set_xlabel(r"$x_0$")
ax0.set_ylabel(r"$x_1$")
ax0.set_title("object", fontsize="medium")
ax1.set_xlabel(r"$s$")
ax2.set_xlabel(r"$s$")
ax3.set_xlabel(r"$x_0$")
ax3.set_ylabel(r"$x_1$")
ax4.set_xlabel(r"$x_0$")
ax4.set_ylabel(r"$x_1$")
ax5.set_xlabel(r"$x_0$")
ax5.set_ylabel(r"$x_1$")
ax6.set_xlabel(r"$x_0$")
ax6.set_ylabel(r"$x_1$")

ax1.set_title(
    f"projection profile {(i+1):03} - $\phi$ = {180*phi[i]/np.pi:.1f}$^\circ$",
    fontsize="medium",
)
ax2.set_title(
    f"filtered projection profile {(i+1):03} - $\phi$ = {180*phi[i]/np.pi:.1f}",
    fontsize="medium",
)
ax3.set_title(f"back projection of profile {(i+1):03}", fontsize="medium")
ax4.set_title(f"filtered back projection of profile {(i+1):03}", fontsize="medium")
ax5.set_title(f"mean of first {(i+1):03} back projections", fontsize="medium")
ax6.set_title(f"mean of first {(i+1):03} filtered back projections", fontsize="medium")

## create animation
# ani = animation.FuncAnimation(
#    fig2, _update_animation, num_phi, interval=5, blit=False, repeat=False
# )
#
## save animation to gif
# ani.save("fbp_animation.mp4", writer=animation.FFMpegWriter(fps=20))

# fig2.show()
