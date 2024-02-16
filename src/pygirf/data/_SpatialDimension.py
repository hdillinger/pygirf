"""SpatialDimension dataclass."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic
from typing import Protocol
from typing import TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar('T', int, float, npt.DTypeLike)


class XYZ(Protocol[T]):
    """Protocol for structures with attributes x, y and z of type T."""

    x: T
    y: T
    z: T


@dataclass(slots=True)
class SpatialDimension(Generic[T]):
    """Spatial dataclass of float/int/tensors (z, y, x)."""

    z: T
    y: T
    x: T

    @classmethod
    def from_xyz(cls, data: XYZ[T], conversion: Callable[[T], T] | None = None) -> SpatialDimension[T]:
        """Create a SpatialDimension from something with (.x .y .z) parameters.

        Parameters
        ----------
        data
            should implement .x .y .z. For example ismrmrd's matrixSizeType.
        conversion,  optional
            will be called for each value to convert it
        """
        if conversion is not None:
            return cls(conversion(data.z), conversion(data.y), conversion(data.x))
        return cls(data.z, data.y, data.x)

    @staticmethod
    def from_array_xyz(
        data: npt.ArrayLike,
        conversion: Callable[[np.dtype], npt.ArrayLike] | None = None,
    ) -> SpatialDimension[npt.ArrayLike]:
        """Create a SpatialDimension from an arraylike interface.

        Parameters
        ----------
        data
            shape (..., 3) in the order (x,y,z)
        conversion, optional
            will be called for each value to convert it, by default None
        """
        if not isinstance(data, (np.ndarray, npt.ArrayLike)):
            data = np.asarray(data)
        data = np.asarray(data)
        if np.size(data, -1) != 3:
            raise ValueError(f'Expected last dimension to be 3, got {np.size(data, -1)}')

        x = np.asarray(data[..., 0])
        y = np.asarray(data[..., 1])
        z = np.asarray(data[..., 2])

        if conversion is not None:
            x = conversion(x)
            y = conversion(y)
            z = conversion(z)
        return SpatialDimension(z, y, x)
