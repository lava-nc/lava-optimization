# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest
import numpy as np
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.core.run_configs import Loihi2SimCfg

from lava.lib.optimization.solvers.generic.read_gate.models import (
    ReadGatePyModel,
)
from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
from lava.lib.optimization.solvers.generic.monitoring_processes\
    .solution_readout.process import SolutionReadout
from lava.proc.spiker.models import SpikerModel
from lava.proc.spiker.process import Spiker


class TestSolutionReadout(unittest.TestCase):
    def setUp(self) -> None:
        # Create processes.
        spiker = Spiker(shape=(4,), period=3, payload=7)
        integrator = Spiker(shape=(1,), period=7, payload=-4)
        readgate = ReadGate(shape=(4,), target_cost=-3)
        self.readout = SolutionReadout(shape=(4,), target_cost=-3)

        # Connect processes.
        integrator.s_out.connect(readgate.cost_in)
        readgate.solution_reader.connect_var(spiker.payload)
        readgate.solution_out.connect(self.readout.read_solution)
        readgate.cost_out.connect(self.readout.cost_in)
        readgate.send_pause_request.connect(self.readout.timestep_in)

        # Execution configurations.
        pdict = {ReadGate: ReadGatePyModel, Spiker: SpikerModel}
        self.run_cfg = Loihi2SimCfg(exception_proc_model_map=pdict)
        self.readout._log_config.level = 20

    def test_stops_when_desired_cost_reached(self):
        self.readout.run(RunContinuous(), run_cfg=self.run_cfg)
        self.readout.wait()
        solution = self.readout.solution.get()
        self.readout.stop()
        self.assertTrue(np.all(solution == np.zeros(4)))
