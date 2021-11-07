# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

from lava.magma.core.process.process import AbstractProcess


class Readout(AbstractProcess):
    """Listener for solver network solution message reads solution when found.
    """
    pass


class HostMonitor(AbstractProcess):
    """Communicate with Readout to get solution and solving time."""
    pass
