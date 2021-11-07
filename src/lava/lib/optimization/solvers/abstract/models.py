# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from src.lava.lib.optimization.solvers.abstract.processes import Readout, \
    HostMonitor


@implements(proc=Readout, protocol=LoihiProtocol)
@requires(CPU)
class PyReadoutModel(PyLoihiProcessModel):

    def run_spk(self):
        pass


@implements(proc=HostMonitor, protocol=LoihiProtocol)
@requires(CPU)
class HostMonitorModel(PyLoihiProcessModel):
    def run(self):
        pass
