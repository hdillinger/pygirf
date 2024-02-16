"""Tests the check_for_duplicates utility."""

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

import numpy as np
import numpy.typing as npt

from pygirf.utils._check_for_duplicates import check_for_duplicates

def test_check_for_duplicates(array: npt.ArrayLike, tol: float):
    """Test _check_for_duplicates."""
    data = np.array([0, 1, 2, 3, 2])
    
    assert not check_for_duplicates(data)