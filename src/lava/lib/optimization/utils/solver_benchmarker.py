# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

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
        num_steps: int
            Number of timesteps the workload is supposed to run.
        """
        self._power_logger = Loihi2Power()
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

        num_steps: int
            Number of timesteps the workload is supposed to run.
        """
        self._time_logger = Loihi2ExecutionTime()
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
