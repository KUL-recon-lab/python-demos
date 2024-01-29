from __future__ import annotations

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import scipy.ndimage as ndi
from collections.abc import Sequence

import abc

class RadonObject(abc.ABC):
    """abstract base class for objects with known radon transform"""

    def __init__(self) -> None:
        self._x0_offset: float = 0.
        self._x1_offset: float = 0.
        self._amplitude: float = 1.

    @abc.abstractmethod
    def _centered_radon_transform(self, r: np.ndarray, phi: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _centered_values(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        pass

    @property
    def x0_offset(self) -> float:
        return self._x0_offset
    
    @x0_offset.setter
    def x0_offset(self, value: float) -> None:
        self._x0_offset = value

    @property
    def x1_offset(self) -> float:
        return self._x1_offset
    
    @x1_offset.setter
    def x1_offset(self, value: float) -> None:
        self._x1_offset = value

    @property
    def amplitude(self) -> float:
        return self._amplitude
    
    @amplitude.setter
    def amplitude(self, value: float) -> None:
        self._amplitude = value

    def radon_transform(self, r, phi) -> float:
        return self._amplitude * self._centered_radon_transform(r - self._x0_offset*np.cos(phi) - self._x1_offset*np.sin(phi), phi) 
    
    def values(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return self._amplitude * self._centered_values(x0 - self._x0_offset, x1 - self._x1_offset)

class RadonObjectSequence(Sequence[RadonObject]):
    def __init__(self, objects: Sequence[RadonObject]) -> None:
        super().__init__()
        self._objects: Sequence[RadonObject] = objects

    def __len__(self) -> int:
        return len(self._objects)

    def __getitem__(self, i: int) -> RadonObject:
        return self._objects[i]

    def radon_transform(self, r, phi) -> float:
        return sum([x.radon_transform(r,phi) for x in self])

    def values(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return sum([x.values(x0,x1) for x in self])

class RadonDisk(RadonObject):
    """2D disk with known radon transform"""
    def __init__(self, radius: float) -> None:
        super().__init__()
        self._radius: float = radius
    
    def _centered_radon_transform(self, r: np.ndarray, phi: np.ndarray) -> np.ndarray:
        mask = np.where(np.abs(r) <= self._radius)
        rt = np.zeros_like(r)

        rt[mask] = 2 * np.sqrt(self._radius**2 - r[mask]**2)

        return rt

    def _centered_values(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return np.where(x0**2 + x1**2 <= self._radius**2, 1., 0.)
    
    @property
    def radius(self) -> float:
        return self._radius
    
    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = value


class RotationBasedProjector:
    def __init__(self, phis : np.ndarray, r: np.ndarray) -> None:
        self._phis : np.ndarray = phis
        self._r : np.ndarray = r
        self._filter : np.ndarray | None = None
    
    @property
    def phis(self) -> np.ndarray:
        return self._phis
    
    @phis.setter
    def phis(self, value: np.ndarray) -> None:
        self._phis = value

    @property
    def r(self) -> np.ndarray:
        return self._r
    
    @r.setter
    def r(self, value: np.ndarray) -> None:
        self._r = value

    @property
    def num_r(self) -> int:
        return self._r.shape[0]

    @property
    def num_phi(self) -> int:
        return self._phis.shape[0]

    @property
    def filter(self) -> np.ndarray | None:
        return self._filter

    @filter.setter
    def filter(self, value: np.ndarray | None) -> None:
        self._filter = value

    def backproject(self, sinogram: np.ndarray) -> np.ndarray:
        back_imgs = np.zeros((self.num_phi, self.num_r, self.num_r), dtype=float)

        for i, profile in enumerate(sinogram):
            if self.filter is not None:
                profile = np.convolve(profile, self.filter, mode='same')

            tmp_img = np.tile(profile, (self.num_r,1))
            back_imgs[i, ...] = ndi.rotate(tmp_img, (180./np.pi) * self._phis[i] - 90, reshape=False, order = 1, prefilter=False)

        return back_imgs

#-----------------------------------------------------------------------------------------------

# %%
# define an object with known radon transform
disk0 = RadonDisk(1.5)
disk0.amplitude = 1.

disk1 = RadonDisk(0.4)
disk1.amplitude = -0.5
disk1.x0_offset = 0.7

disk2 = RadonDisk(0.3)
disk2.amplitude = 0.5
disk2.x1_offset = 0.7

disk3 = RadonDisk(0.2)
disk3.amplitude = -1
disk3.x0_offset = -0.7

disk4 = RadonDisk(0.12)
disk4.amplitude = -0.5
disk4.x1_offset = -0.7

disk5 = RadonDisk(0.1)
disk5.amplitude = 1.
disk5.x1_offset = -0.7

radon_object = RadonObjectSequence([disk0, disk1, disk2, disk3, disk4, disk5])

# %%
# setup r and phi coordinates
r = np.linspace(-2, 2, 151)
num_phi = int(0.5*r.shape[0]*np.pi) + 1
phi = np.linspace(0, np.pi, num_phi, endpoint=False)
PHI, R = np.meshgrid(phi, r, indexing='ij')
x = np.linspace(r.min(), r.max(), 3*r.shape[0])
X0, X1 = np.meshgrid(r, r, indexing='ij')
X0hr, X1hr = np.meshgrid(x, x, indexing='ij')

# %%
# analytic calculation of the randon transform
sino = radon_object.radon_transform(R,PHI)

# %%
# add Poisson noise
#sino = np.random.poisson(500*sino)

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
fig, ax = plt.subplots(1,4, figsize=(16,4), tight_layout=True)
ax[0].imshow(radon_object.values(X0hr,X1hr), cmap='Greys', extent = [r.min(),r.max(),r.min(),r.max()], origin='lower')
ax[1].imshow(sino, cmap='Greys', extent = [r.min(),r.max(),phi.min(),phi.max()], origin='lower')
ax[2].imshow(back_proj, cmap='Greys', extent = [r.min(),r.max(),r.min(),r.max()], origin='lower')
ax[3].imshow(filtered_back_proj, cmap='Greys', extent = [r.min(),r.max(),r.min(),r.max()], origin='lower')


ax[0].set_xlabel(r'$x_0$')
ax[0].set_ylabel(r'$x_1$')
ax[1].set_xlabel(r'$s$')
ax[1].set_ylabel(r'$\phi$')
ax[2].set_xlabel(r'$x_0$')
ax[2].set_ylabel(r'$x_1$')
ax[3].set_xlabel(r'$x_0$')
ax[3].set_ylabel(r'$x_1$')

ax[0].set_title(r'object ($x$)')
ax[1].set_title(r'Radon transform of object ($y = Rx$)')
ax[2].set_title(r'back projection ($R^T y$)')
ax[3].set_title(r'filtered back projection ($R^{-1} y$)')

fig.show()

# %%
def _update_animation(i):
    p1.set_ydata(sino[i,:])
    p2.set_ydata(np.convolve(sino[i,:], f, mode = 'same'))

    img3.set_data(back_projs[i,...])
    img4.set_data(filtered_back_projs[i,...])
    img5.set_data(back_projs[:(i+1),...].mean(axis=0))
    img6.set_data(filtered_back_projs[:(i+1),...].mean(axis=0))

    ax1.set_title(f'projection profile {(i+1):03} - $\phi$ = {180*phi[i]/np.pi:.1f}$^\circ$', fontsize='medium')
    ax2.set_title(f'filtered projection profile {(i+1):03} - $\phi$ = {180*phi[i]/np.pi:.1f}', fontsize='medium')
    ax3.set_title(f'back projection of profile {(i+1):03}', fontsize='medium')
    ax4.set_title(f'filtered back projection of profile {(i+1):03}', fontsize='medium')
    ax5.set_title(f'mean of first {(i+1):03} back projections', fontsize='medium')
    ax6.set_title(f'mean of first {(i+1):03} filtered back projections', fontsize='medium')

    return (p1, p2, img3, img4, img5, img6)


# %%
# animated random transform and sinogram


fig2 = plt.figure(tight_layout=True, figsize = (12,8))
gs = gridspec.GridSpec(4,6)

ax0 = fig2.add_subplot(gs[:2, :2])
ax1 = fig2.add_subplot(gs[2, :2])
ax2 = fig2.add_subplot(gs[3, :2])
ax3 = fig2.add_subplot(gs[:2, 2:4])
ax4 = fig2.add_subplot(gs[:2, 4:6])
ax5 = fig2.add_subplot(gs[2:4, 2:4])
ax6 = fig2.add_subplot(gs[2:4, 4:6])

ax0.imshow(radon_object.values(X0hr,X1hr), cmap='Greys', extent = [r.min(),r.max(),r.min(),r.max()], origin='lower')

i = 0
R = 1.5

for s in np.linspace(-1,1,7):
    d0 = s*np.sin(phi[i])
    d1 = s*np.cos(phi[i])
    ax0.arrow(R*np.cos(phi[i]) + d0, -R*np.sin(phi[i]) + d1, -2*R*np.cos(phi[i]), 2*R*np.sin(phi[i]), color='r', width=0.001, head_width = 0.1)
    ax0.annotate(f'{s:.1f}', (1.2*R*np.cos(phi[i]) + d0, -1.2*R*np.sin(phi[i]) + d1), color = 'r', fontsize='small', ha='center', va='center')

p1 = ax1.plot(r, sino[i,:], color='k')[0]
ax1.set_ylim(sino.min(), sino.max())
ax1.grid(ls = ':')

p2 = ax2.plot(r, np.convolve(sino[i,:], f, mode = 'same'), color='k')[0]
ax2.set_ylim(filtered_back_projs.min(), filtered_back_projs.max())
ax2.grid(ls = ':')

img3 = ax3.imshow(back_projs[i,...], cmap='Greys', extent = [r.min(),r.max(),r.min(),r.max()], origin='lower')
img4 = ax4.imshow(filtered_back_projs[i,...], cmap='Greys', extent = [r.min(),r.max(),r.min(),r.max()], origin='lower')
img5 = ax5.imshow(back_projs[:(i+1),...].mean(axis=0), cmap='Greys', extent = [r.min(),r.max(),r.min(),r.max()], origin='lower',
                  vmin = back_proj.min(), vmax = back_proj.max())
img6 = ax6.imshow(filtered_back_projs[:(i+1),...].mean(axis=0), cmap='Greys', extent = [r.min(),r.max(),r.min(),r.max()], origin='lower',
                  vmin = filtered_back_proj.min(), vmax = filtered_back_proj.max())

ax0.set_xlabel(r'$x_0$')
ax0.set_ylabel(r'$x_1$')
ax0.set_title('object', fontsize='medium')
ax1.set_xlabel(r'$s$')
ax2.set_xlabel(r'$s$')
ax3.set_xlabel(r'$x_0$')
ax3.set_ylabel(r'$x_1$')
ax4.set_xlabel(r'$x_0$')
ax4.set_ylabel(r'$x_1$')
ax5.set_xlabel(r'$x_0$')
ax5.set_ylabel(r'$x_1$')
ax6.set_xlabel(r'$x_0$')
ax6.set_ylabel(r'$x_1$')

ax1.set_title(f'projection profile {(i+1):03} - $\phi$ = {180*phi[i]/np.pi:.1f}$^\circ$', fontsize='medium')
ax2.set_title(f'filtered projection profile {(i+1):03} - $\phi$ = {180*phi[i]/np.pi:.1f}', fontsize='medium')
ax3.set_title(f'back projection of profile {(i+1):03}', fontsize='medium')
ax4.set_title(f'filtered back projection of profile {(i+1):03}', fontsize='medium')
ax5.set_title(f'mean of first {(i+1):03} back projections', fontsize='medium')
ax6.set_title(f'mean of first {(i+1):03} filtered back projections', fontsize='medium')

ani = animation.FuncAnimation(fig2, _update_animation, num_phi, interval=5, blit=False, repeat = False)
fig2.show()

