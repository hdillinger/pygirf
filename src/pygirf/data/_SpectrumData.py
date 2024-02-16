"""Gridded spetrum data class."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
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

from pint import Quantity
import numpy as np
import numpy.typing as npt
import dataclasses

@dataclasses.dataclass(init=False, slots=True, frozen=True)
class SpectrumData():
    """Spectrum data class including arrays for grid definition."""

    grid: npt.ArrayLike
    values: npt.ArrayLike

    def __init__(self, grid: npt.ArrayLike, values: npt.ArrayLike) -> None:
        """Create SpectrumData object from grid and value arrays.

        Parameters
        ----------
        grid
            Frequency grid point on which data points are defined.
        data
            Data values on frequency grid points.
        """

        object.__setattr__(self, 'grid', grid)
        object.__setattr__(self, 'values', values)
