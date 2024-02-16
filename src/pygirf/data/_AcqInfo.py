"""Acquisition information dataclass."""

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

from __future__ import annotations

from dataclasses import dataclass

import ismrmrd
import numpy as np

from pygirf.data import SpatialDimension
import numpy.typing as npt

@dataclass(slots=True)
class AcqIdx:
    """Acquisition index for each readout."""

    k1: npt.DTypeLike
    k2: npt.DTypeLike
    average: npt.DTypeLike
    slice: npt.DTypeLike
    contrast: npt.DTypeLike
    phase: npt.DTypeLike
    repetition: npt.DTypeLike
    set: npt.DTypeLike
    segment: npt.DTypeLike
    user: npt.DTypeLike


@dataclass(slots=True)
class AcqInfo:
    """Acquisition information for each readout."""

    idx: AcqIdx
    acquisition_time_stamp: npt.DTypeLike
    active_channels: npt.DTypeLike
    available_channels: npt.DTypeLike
    center_sample: npt.DTypeLike
    channel_mask: npt.DTypeLike
    discard_post: npt.DTypeLike
    discard_pre: npt.DTypeLike
    encoding_space_ref: npt.DTypeLike
    flags: npt.DTypeLike
    measurement_uid: npt.DTypeLike

    number_of_samples: npt.DTypeLike
    """Number of readout sample points per readout (readouts may have different
    number of sample points)."""

    patient_table_position: SpatialDimension[np.dtype]
    phase_dir: SpatialDimension[np.dtype]
    physiology_time_stamp: npt.DTypeLike
    position: SpatialDimension[np.dtype]
    read_dir: SpatialDimension[np.dtype]
    sample_time_us: npt.DTypeLike
    scan_counter: npt.DTypeLike
    slice_dir: SpatialDimension[np.dtype]
    trajectory_dimensions: npt.DTypeLike  # =3. We only support 3D Trajectories: kz always exists.
    user_float: npt.DTypeLike
    user_int: npt.DTypeLike
    version: npt.DTypeLike

    @classmethod
    def from_ismrmrd_acquisitions(
        cls,
        acquisitions: list[ismrmrd.Acquisition],
    ) -> AcqInfo:
        """Read the header of a list of acquisition and store information.

        Parameters
        ----------
        acquisitions:
            list of ismrmrd acquisistions to read from. Needs at least one acquisition.
        """

        # Idea: create array of structs, then a struct of arrays,
        # convert it into tensors to store in our dataclass.
        # TODO: there might be a faster way to do this.

        if len(acquisitions) == 0:
            raise ValueError('Acquisition list must not be empty.')

        # Creating the dtype first and casting to bytes
        # is a workaround for a bug in cpython > 3.12 causing a warning
        # is np.array(AcquisitionHeader) is called directly.
        # also, this needs to check the dtyoe only once.
        acquisition_head_dtype = np.dtype(ismrmrd.AcquisitionHeader)
        headers = np.frombuffer(
            np.array([memoryview(a._head).cast('B') for a in acquisitions]), dtype=acquisition_head_dtype
        )

        idx = headers['idx']

        def spatialdimension(data):
            # all spatial dimensions are float32
            return SpatialDimension[np.array].from_array_xyz(np.array(data.astype(np.float32)))

        acq_idx = AcqIdx(
            k1=np.dtype(idx['kspace_encode_step_1']),
            k2=np.dtype(idx['kspace_encode_step_2']),
            average=np.dtype(idx['average']),
            slice=np.dtype(idx['slice']),
            contrast=np.dtype(idx['contrast']),
            phase=np.dtype(idx['phase']),
            repetition=np.dtype(idx['repetition']),
            set=np.dtype(idx['set']),
            segment=np.dtype(idx['segment']),
            user=np.dtype(idx['user']),
        )

        acq_info = cls(
            idx=acq_idx,
            acquisition_time_stamp=np.dtype(headers['acquisition_time_stamp']),
            active_channels=np.dtype(headers['active_channels']),
            available_channels=np.dtype(headers['available_channels']),
            center_sample=np.dtype(headers['center_sample']),
            channel_mask=np.dtype(headers['channel_mask']),
            discard_post=np.dtype(headers['discard_post']),
            discard_pre=np.dtype(headers['discard_pre']),
            encoding_space_ref=np.dtype(headers['encoding_space_ref']),
            flags=np.dtype(headers['flags']),
            measurement_uid=np.dtype(headers['measurement_uid']),
            number_of_samples=np.dtype(headers['number_of_samples']),
            patient_table_position=spatialdimension(headers['patient_table_position']),
            phase_dir=spatialdimension(headers['phase_dir']),
            physiology_time_stamp=np.dtype(headers['physiology_time_stamp']),
            position=spatialdimension(headers['position']),
            read_dir=spatialdimension(headers['read_dir']),
            sample_time_us=np.dtype(headers['sample_time_us']),
            scan_counter=np.dtype(headers['scan_counter']),
            slice_dir=spatialdimension(headers['slice_dir']),
            trajectory_dimensions=np.dtype(headers['trajectory_dimensions']).fill_(3),  # see above
            user_float=np.dtype(headers['user_float']),
            user_int=np.dtype(headers['user_int']),
            version=np.dtype(headers['version']),
        )
        return acq_info
