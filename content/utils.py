import numpy as np
import scipy.ndimage as ndi
from collections.abc import Sequence

import abc


class RadonObject(abc.ABC):
    """abstract base class for objects with known radon transform"""

    def __init__(self) -> None:
        self._x0_offset: float = 0.0
        self._x1_offset: float = 0.0
        self._s0: float = 1.0
        self._s1: float = 1.0
        self._amplitude: float = 1.0

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
    def s0(self) -> float:
        return self._s0

    @s0.setter
    def s0(self, value: float) -> None:
        self._s0 = value

    @property
    def s1(self) -> float:
        return self._s1

    @s1.setter
    def s1(self, value: float) -> None:
        self._s1 = value

    @property
    def amplitude(self) -> float:
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float) -> None:
        self._amplitude = value

    def radon_transform(self, s, phi) -> float:
        s_prime = s / np.sqrt(
            self._s0**2 * np.cos(phi) ** 2 + self._s1**2 * np.sin(phi) ** 2
        )
        phi_prime = np.arctan2(self._s0 * np.sin(phi), self._s1 * np.cos(phi))

        return self._amplitude * self._centered_radon_transform(
            s_prime
            - self._x0_offset * np.cos(phi_prime)
            - self._x1_offset * np.sin(phi_prime),
            phi_prime,
        )

    def values(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return self._amplitude * self._centered_values(
            x0 / self._s0 - self._x0_offset, x1 / self._s1 - self._x1_offset
        )


class RadonObjectSequence(Sequence[RadonObject]):
    def __init__(self, objects: Sequence[RadonObject]) -> None:
        super().__init__()
        self._objects: Sequence[RadonObject] = objects

    def __len__(self) -> int:
        return len(self._objects)

    def __getitem__(self, i: int) -> RadonObject:
        return self._objects[i]

    def radon_transform(self, r, phi) -> float:
        return sum([x.radon_transform(r, phi) for x in self])

    def values(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return sum([x.values(x0, x1) for x in self])


class RadonDisk(RadonObject):
    """2D disk with known radon transform"""

    def __init__(self, radius: float) -> None:
        super().__init__()
        self._radius: float = radius

    def _centered_radon_transform(self, r: np.ndarray, phi: np.ndarray) -> np.ndarray:
        mask = np.where(np.abs(r) <= self._radius)
        rt = np.zeros_like(r)

        rt[mask] = 2 * np.sqrt(self._radius**2 - r[mask] ** 2)

        return rt

    def _centered_values(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return np.where(x0**2 + x1**2 <= self._radius**2, 1.0, 0.0)

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = value


class RotationBasedProjector:
    def __init__(self, phis: np.ndarray, r: np.ndarray) -> None:
        self._phis: np.ndarray = phis
        self._r: np.ndarray = r
        self._filter: np.ndarray | None = None

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
                profile = np.convolve(profile, self.filter, mode="same")

            m = self.num_r // 2

            tmp_img = np.tile(profile, (2 * m + self.num_r, 1))
            back_imgs[i, ...] = ndi.rotate(
                tmp_img,
                (180.0 / np.pi) * self._phis[i] - 90,
                reshape=False,
                order=3,
                prefilter=True,
            )[m:-m, :]

        return back_imgs
