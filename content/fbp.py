from __future__ import annotations

import numpy as np 
import matplotlib.pyplot as plt
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
        return self._amplitude * self._centered_radon_transform(r - self._x1_offset*np.cos(phi) - self._x0_offset*np.sin(phi), phi) 
    
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
        back_img = np.zeros((self.num_r, self.num_r), dtype=float)

        for i, profile in enumerate(sinogram):
            if self.filter is not None:
                profile = np.convolve(profile, self.filter, mode='same')

            tmp_img = np.tile(profile, (self.num_r,1))
            back_img += ndi.rotate(tmp_img, (-180./np.pi) * self._phis[i], reshape=False, order = 1, prefilter=False)

        return back_img

#-----------------------------------------------------------------------------------------------

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

# setup r and phi coordinates
r = np.linspace(-2, 2, 151)
num_phi = int(r.shape[0]*np.pi) + 1
phi = np.linspace(0, np.pi, num_phi, endpoint=False)
PHI, R = np.meshgrid(phi, r, indexing='ij')
x = np.linspace(r.min(), r.max(), 3*r.shape[0])
X0, X1 = np.meshgrid(x, x, indexing='ij')

# analytic calculation of the randon transform
sino = radon_object.radon_transform(R,PHI)

# back projections and filtered back projections
proj = RotationBasedProjector(phi, r)
back_proj = proj.backproject(sino)

# setup a ramp filter
k = np.fft.fftfreq(r.shape[0], r[1]-r[0])
f = np.fft.fftshift(np.fft.ifft(np.abs(k))).real

proj.filter = f
filtered_back_proj = proj.backproject(sino)

fig, ax = plt.subplots(1,4, figsize=(16,4))
ax[0].imshow(radon_object.values(X0,X1), cmap='Greys')
ax[1].imshow(sino, cmap='Greys', aspect = 1/np.pi)
ax[2].imshow(back_proj, cmap='Greys')
ax[3].imshow(filtered_back_proj, cmap='Greys')
ax[0].set_xticks(np.arange(x.shape[0])[::45], [f'{z:.1f}' for z in x[::45]])
ax[0].set_yticks(np.arange(x.shape[0])[::45], [f'{z:.1f}' for z in x[::45]])
ax[1].set_xticks(np.arange(r.shape[0])[::15], [f'{x:.1f}' for x in r[::15]])
ax[1].set_yticks(np.arange(phi.shape[0])[::45], [f'{180*x/np.pi:.1f}' for x in phi[::45]])
ax[2].set_xticks(np.arange(r.shape[0])[::15], [f'{x:.1f}' for x in r[::15]])
ax[2].set_yticks(np.arange(r.shape[0])[::15], [f'{x:.1f}' for x in r[::15]])
ax[3].set_xticks(np.arange(r.shape[0])[::15], [f'{x:.1f}' for x in r[::15]])
ax[3].set_yticks(np.arange(r.shape[0])[::15], [f'{x:.1f}' for x in r[::15]])

ax[0].set_xlabel(r'$x_1$')
ax[0].set_ylabel(r'$x_0$')
ax[1].set_xlabel(r'$r$')
ax[1].set_ylabel(r'$\phi$')
ax[2].set_xlabel(r'$x_1$')
ax[2].set_ylabel(r'$x_0$')
ax[3].set_xlabel(r'$x_1$')
ax[3].set_ylabel(r'$x_0$')

ax[0].set_title(r'object ($x$)')
ax[1].set_title(r'Radon transform of object ($y = Rx$)')
ax[2].set_title(r'back projection ($R^T y$)')
ax[3].set_title(r'filtered back projection ($R^{-1} y$)')


fig.tight_layout()
fig.show()