from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.transforms as mtransforms

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
sino = radon_object.radon_transform(R, PHI)


# %%
def _update_animation(i):
    p1.set_ydata(sino[i, :])
    ax1.set_title(
        f"projection profile {(i+1):03} - $\phi$ = {180*phi[i]/np.pi:.1f}$^\circ$",
        fontsize="medium",
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

    tmp_sino = sino.copy()
    tmp_sino[(i + 1) :] = 0
    img2.set_data(tmp_sino)

    return (p1, arr, ann)


# %%
# animated random transform and sinogram

fig2 = plt.figure(tight_layout=True, figsize=(12, 4))
gs = gridspec.GridSpec(1, 3)

ax0 = fig2.add_subplot(gs[:, 0])
ax1 = fig2.add_subplot(gs[:, 1])
ax2 = fig2.add_subplot(gs[:, 2])

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

p1 = ax1.plot(r, sino[i, :], color="k")[0]
ax1.set_ylim(sino.min(), sino.max())
ax1.grid(ls=":")

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

tmp_sino = sino.copy()
tmp_sino[(i + 1) :] = 0

img2 = ax2.imshow(
    tmp_sino,
    cmap="Greys",
    extent=[r.min(), r.max(), phi.min(), phi.max()],
    origin="lower",
)


ax0.set_xlabel(r"$x_0$")
ax0.set_ylabel(r"$x_1$")
ax0.set_title("object", fontsize="medium")
ax1.set_xlabel(r"$s$")
ax2.set_xlabel(r"$s$")
ax2.set_ylabel(r"$\phi$")

ax1.set_title(
    f"projection profile {(i+1):03} - $\phi$ = {180*phi[i]/np.pi:.1f}$^\circ$",
    fontsize="medium",
)
ax2.set_title(
    f"sinogram",
    fontsize="medium",
)
# create animation
ani = animation.FuncAnimation(
    fig2, _update_animation, num_phi, interval=5, blit=False, repeat=False
)

# save animation to gif
ani.save("sinogram.mp4", writer=animation.FFMpegWriter(fps=20))

fig2.show()
