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

from lava.utils import loihi2_profiler


class SolverBenchmarker():

    def __init__(self):
        self._power_logger = None
        self._time_logger = None

    def get_power_measurement_cfg(self, num_steps: int):
        self._power_logger = loihi2_profiler.Loihi2Power(num_steps=num_steps)
        pre_run_fxs = [
            self._power_logger.attach,
        ]
        post_run_fxs = [
            lambda _: self._power_logger.get_results(),
        ]
        return pre_run_fxs, post_run_fxs

    def get_time_measurement_cfg(self, num_steps: int):
        self._time_logger = loihi2_profiler.Loihi2ExecutionTime(
            buffer_size=num_steps)
        pre_run_fxs = [
            self._time_logger.attach,
        ]
        post_run_fxs = [
            lambda _: self._time_logger.get_results(),
        ]
        return pre_run_fxs, post_run_fxs

    @property
    def measured_power(self):
        return self._power_logger.total_power

    @property
    def measured_time(self):
        return self._time_logger.time_per_step
