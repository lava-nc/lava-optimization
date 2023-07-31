# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import unittest

import numpy as np
from lava.lib.optimization.solvers.generic.solution_reader.process import (
    SolutionReader,
)
from lava.lib.optimization.solvers.generic.read_gate.models import \
    get_read_gate_model_class
from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
from lava.magma.core.run_conditions import RunContinuous
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.proc.spiker.models import SpikerModel
from lava.proc.spiker.process import Spiker


class TestSolutionReader(unittest.TestCase):
    def setUp(self) -> None:
        # Create processes.
        spiker = Spiker(shape=(4,), period=3, payload=7)
        # Together, these processes spike a cost of (-1<<24)+16777212 = -4
        integrator_last_bytes = Spiker(shape=(1,), period=7, payload=16777212)
        integrator_first_byte = Spiker(shape=(1,), period=7, payload=-1)

        self.solution_reader = SolutionReader(
            var_shape=(4,), target_cost=0, min_cost=2
        )

        # Connect processes.
        integrator_last_bytes.s_out.connect(
            self.solution_reader.read_gate_in_port_last_bytes_0)
        integrator_first_byte.s_out.connect(
            self.solution_reader.read_gate_in_port_first_byte_0)
        self.solution_reader.ref_port.connect_var(spiker.payload)

        # Execution configurations.
        ReadGatePyModel = get_read_gate_model_class(1)
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
