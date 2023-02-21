# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest

import numpy as np
from lava.lib.optimization.solvers.generic.solution_reader.process import (
    SolutionReader,
)
from lava.lib.optimization.solvers.generic.read_gate.models import (
    ReadGatePyModel,
)
from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
from lava.magma.core.run_conditions import RunContinuous
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.spiker.models import SpikerModel
from lava.proc.spiker.process import Spiker


class TestSolutionReader(unittest.TestCase):
    def setUp(self) -> None:
        # Create processes.
        spiker = Spiker(shape=(4,), period=3, payload=7)
        integrator = Spiker(shape=(1,), period=7, payload=-4)

        self.solution_reader = SolutionReader(
            var_shape=(4,), target_cost=0, min_cost=2
        )

        # Connect processes.
        integrator.s_out.connect(self.solution_reader.read_gate_in_port_0)
        self.solution_reader.ref_port.connect_var(spiker.payload)

        # Execution configurations.
        pdict = {ReadGate: ReadGatePyModel, Spiker: SpikerModel}
        self.run_cfg = Loihi2SimCfg(exception_proc_model_map=pdict)
        self.solution_reader._log_config.level = 20

    def test_create_process(self):
        self.assertIsInstance(self.solution_reader, SolutionReader)

    def test_stops_when_desired_cost_reached(self):
        self.solution_reader.run(RunContinuous(), run_cfg=self.run_cfg)
        self.solution_reader.wait()
        solution = self.solution_reader.solution.get()
        self.solution_reader.stop()
        self.assertTrue(np.all(solution == np.zeros(4)))
