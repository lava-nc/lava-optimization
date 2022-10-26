# INTEL CORPORATION CONFIDENTIAL AND PROPRIETARY
#
# Copyright © 2022 Intel Corporation.
#
# This software and the related documents are Intel copyrighted
# materials, and your use of them is governed by the express
# license under which they were provided to you (License). Unless
# the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit  this software or the
# related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with
# no express or implied warranties, other than those that are
# expressly stated in the License.
# See: https://spdx.org/licenses/
# SPDX-License-Identifier: BSD-3-Clause
try:
    from lava.utils.loihi2_profiler import Loihi2Power, Loihi2ExecutionTime
except ImportError:
    class Loihi2Power:
        pass

    class Loihi2ExecutionTime:
        pass

import numpy as np


class SolverBenchmarker:
    """
    Utility class to benchmark power consumption and execution time of a
    solver on Loihi2.
    """

    def __init__(self):
        self._power_logger = None
        self._time_logger = None

    def get_power_measurement_cfg(self, num_steps: int):
        """
        Returns the pre- and post- fixtures to be passed to a ``Loihi2HwCfg``
        to enable power consumption monitoring.

        Parameters
        ----------
        :param num_steps: Number of timesteps the workload is supposed to run.
        """
        self._power_logger = Loihi2Power(num_steps=num_steps)
        pre_run_fxs = [
            self._power_logger.attach,
        ]
        post_run_fxs = [
            lambda _: self._power_logger.get_results(),
        ]
        return pre_run_fxs, post_run_fxs

    def get_time_measurement_cfg(self, num_steps: int):
        """
        Returns the pre- and post- fixtures to be passed to a ``Loihi2HwCfg``
        to enable execution time monitoring.

        :param num_steps: Number of timesteps the workload is supposed to run.
        """
        self._time_logger = Loihi2ExecutionTime(
            buffer_size=num_steps
        )
        pre_run_fxs = [
            self._time_logger.attach,
        ]
        post_run_fxs = [
            lambda _: self._time_logger.get_results(),
        ]
        return pre_run_fxs, post_run_fxs

    @property
    def measured_power(self):
        """
        Returns the measured power consumption, or an empty array if nothing was
        measured.
        """
        return (
            self._power_logger.total_power
            if self._power_logger is not None
            else np.array([])
        )

    @property
    def measured_time(self):
        """
        Returns the measured execution time, or an empty array if nothing was
        measured.
        """
        return (
            self._time_logger.time_per_step
            if self._time_logger is not None
            else np.array([])
        )
