# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.core.run_configs import Loihi2HwCfg

from lava.proc.read_gate.ncmodels import \
    ReadGateCModel
from lava.lib.optimization.solvers.generic.monitoring_processes.read_gate.process import \
    ReadGate
from lava.lib.optimization.solvers.generic.monitoring_processes import SolutionReadout
from lava.proc.spiker.ncmodels import \
    SpikerNcModel
from lava.proc.spiker.process import Spiker


@unittest.skip("Waiting for hanging issue to be sovled.")
class TestSolutionReadout(unittest.TestCase):
    def setUp(self) -> None:
        # Create processes.
        spiker = Spiker(shape=(4,), rate=3, payload=7)
        integrator = Spiker(shape=(1,), rate=7, payload=4)
        readgate = ReadGate(shape=(4,), target_cost=5)
        self.readout = SolutionReadout(shape=(4,), target_cost=5)

        # Connect processes.
        integrator.s_out.connect(readgate.cost_in)
        readgate.solution_reader.connect_var(spiker.counter)
        readgate.solution_out.connect(self.readout.read_solution)
        readgate.cost_out.connect(self.readout.cost_in)

        # Execution configurations.
        pdict = {ReadGate: ReadGateCModel, Spiker: SpikerNcModel}
        self.run_cfg = Loihi2HwCfg(exception_proc_model_map=pdict)
        self.readout._log_config.level = 20

    def test_runs_without_stalling(self):
        self.readout.run(RunSteps(num_steps=20), run_cfg=self.run_cfg)
        self.readout.stop()

    def test_stops_when_desired_cost_reached(self):
        self.readout.target_cost.init = 5
        self.readout._log_config.level = 20
        self.readout.run(RunContinuous(), run_cfg=self.run_cfg)
        self.readout.wait()
        self.readout.stop()

