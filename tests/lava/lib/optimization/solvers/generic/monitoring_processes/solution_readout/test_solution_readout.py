# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np

from lava.lib.optimization.solvers.generic.monitoring_processes \
    .solution_readout.process import \
    SolutionReadout
from lava.lib.optimization.solvers.generic.read_gate.models import \
    get_read_gate_model_class
from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
from lava.magma.core.run_conditions import RunContinuous
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.spiker.models import SpikerModel
from lava.proc.spiker.process import Spiker


class TestSolutionReadout(unittest.TestCase):
    def setUp(self) -> None:
        # Create processes.
        spiker = Spiker(shape=(4,), period=3, payload=7)
        # Together, these processes spike a cost of (-1<<24)+16777212 = -4
        integrator_last_bytes = Spiker(shape=(1,), period=7, payload=16777212)
        integrator_first_byte = Spiker(shape=(1,), period=7, payload=-1)
        readgate = ReadGate(shape=(4,), target_cost=-3)
        self.readout = SolutionReadout(shape=(4,), target_cost=-3)

        # Connect processes.
        integrator_last_bytes.s_out.connect(readgate.cost_in_last_bytes_0)
        integrator_first_byte.s_out.connect(readgate.cost_in_first_byte_0)
        readgate.solution_reader.connect_var(spiker.payload)
        readgate.solution_out.connect(self.readout.read_solution)
        readgate.cost_out.connect(self.readout.cost_in)
        readgate.send_pause_request.connect(self.readout.timestep_in)

        # Execution configurations.
        ReadGatePyModel = get_read_gate_model_class(1)
        pdict = {ReadGate: ReadGatePyModel, Spiker: SpikerModel}
        self.run_cfg = Loihi2SimCfg(exception_proc_model_map=pdict)
        self.readout._log_config.level = 20

    def test_stops_when_desired_cost_reached(self):
        self.readout.run(RunContinuous(), run_cfg=self.run_cfg)
        self.readout.wait()
        solution = self.readout.solution.get()
        self.readout.stop()
        self.assertTrue(np.all(solution == np.zeros(4)))
