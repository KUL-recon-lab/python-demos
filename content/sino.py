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

disk1 = RadonDisk(0.4)
disk1.amplitude = -0.5
disk1.x0_offset = 0.7

disk2 = RadonDisk(0.3)
disk2.amplitude = 0.5
disk2.x1_offset = 0.7

disk3 = RadonDisk(0.2)
disk3.amplitude = -1
disk3.x0_offset = -0.7

disk4 = RadonDisk(0.14)
disk4.amplitude = -0.5
disk4.x1_offset = -0.7

disk5 = RadonDisk(0.1)
disk5.amplitude = 1.0
disk5.x1_offset = -0.7

radon_object = RadonObjectSequence([disk0, disk1, disk2, disk3, disk4, disk5])

# %%
# setup r and phi coordinates
r = np.linspace(-2.5, 2.5, 151)
num_phi = int(0.5 * r.shape[0] * np.pi) + 1
phi = np.linspace(0, np.pi, num_phi, endpoint=False)
PHI, R = np.meshgrid(phi, r, indexing="ij")
x = np.linspace(r.min(), r.max(), 3 * r.shape[0])
X0, X1 = np.meshgrid(r, r, indexing="ij")
X0hr, X1hr = np.meshgrid(x, x, indexing="ij")

# %%
# analytic calculation of the randon transform
sino = radon_object.radon_transform(R, PHI)

# %%
# add Poisson noise
# sino = np.random.poisson(500*sino)

# %%
# back projection
proj = RotationBasedProjector(phi, r)
back_projs = proj.backproject(sino)
back_proj = back_projs.mean(axis=0)

# %%
# filtered back projection

# setup a ramp filter
k = np.fft.fftfreq(r.shape[0])
f = np.fft.fftshift(np.fft.ifft(np.abs(k))).real
f /= f.max()

proj.filter = f
filtered_back_projs = proj.backproject(sino)
filtered_back_proj = filtered_back_projs.mean(axis=0)

# %%
# visualize images
fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
ax[0].imshow(
    radon_object.values(X0hr, X1hr).T,
    cmap="Greys",
    extent=[r.min(), r.max(), r.min(), r.max()],
    origin="lower",
)
ax[1].imshow(
    sino, cmap="Greys", extent=[r.min(), r.max(), phi.min(), phi.max()], origin="lower"
)


ax[0].set_xlabel(r"$x_0$")
ax[0].set_ylabel(r"$x_1$")
ax[1].set_xlabel(r"$s$")
ax[1].set_ylabel(r"$\phi$")

ax[0].set_title(r"object ($x$)")
ax[1].set_title(r"Radon transform of object ($y = Rx$)")

plt.show()
