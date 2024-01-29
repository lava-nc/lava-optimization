# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
import unittest

import numpy as np
from lava.lib.optimization.problems.problems import OptimizationProblem, QUBO
from lava.lib.optimization.solvers.generic.read_gate.models import (
    get_read_gate_model_class,
)
from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
from lava.lib.optimization.solvers.generic.solution_finder.process import (
    SolutionFinder,
)
from lava.lib.optimization.solvers.generic.solution_reader.process import (
    SolutionReader,
)
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from numpy import typing as npt

ReadGatePyModel = get_read_gate_model_class(1)


class Mock:
    def __init__(self, val):
        self._val = val

    @property
    def init(self):
        return self._val


class OptimizationSolverProcess(AbstractProcess):
    def __init__(
        self,
        problem: OptimizationProblem,
        hyperparameters: ty.Dict[str, ty.Union[int, npt.ArrayLike]],
        name: ty.Optional[str] = None,
        log_config: ty.Optional[LogConfig] = None,
    ) -> None:
        super().__init__(
            hyperparameters=hyperparameters, name=name, log_config=log_config
        )
        self.problem = problem
        self.hyperparameters = hyperparameters
        self.cost_diagonal = problem.cost.coefficients[2].diagonal()


@implements(proc=OptimizationSolverProcess, protocol=LoihiProtocol)
@requires(CPU)
class OptimizationSolverModel(AbstractSubProcessModel):
    def __init__(self, proc):
        target_cost = 0
        cost_diagonal = proc.cost_diagonal
        problem = proc.problem
        constraints = proc.problem.constraints
        hyperparameters = proc.hyperparameters

        q_off_diag = problem.cost.coefficients[2]
        q_diag = problem.cost.coefficients[2].diagonal()
        init_value = proc.hyperparameters.get(
            "init_value", np.zeros((4,), dtype=int)
        )
        cost_coefficients = {1: Mock(q_diag), 2: Mock(q_off_diag)}

        self.solution_reader = SolutionReader(
            var_shape=(4,), target_cost=target_cost, min_cost=2**24
        )

        self.finder = SolutionFinder(
            cost_diagonal=cost_diagonal,
            cost_coefficients=cost_coefficients,
            constraints=constraints,
            hyperparameters=hyperparameters,
            discrete_var_shape=(4,),
            continuous_var_shape=None,
            backend="CPU",
            problem=problem,
        )

        # Connect processes
        self.finder.cost_out_first_byte.connect(
            self.solution_reader.read_gate_in_port_first_byte_0
        )
        self.finder.cost_out_last_bytes.connect(
            self.solution_reader.read_gate_in_port_last_bytes_0
        )
        self.solution_reader.ref_port.connect_var(
            self.finder.variables_assignment
        )


class TestOptSolverWrapper(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = QUBO(
            np.array(
                [[-5, 2, 4, 0], [2, -3, 1, 0], [4, 1, -8, 5], [0, 0, 5, -6]]
            )
        )
        self.solution = np.asarray([1, 0, 0, 1]).astype(int)

        self.solver = OptimizationSolverProcess(
            problem=self.problem,
            hyperparameters={
                "init_state": self.problem.cost.coefficients[2].diagonal()
            },
        )

        # Execution configurations.
        pdict = {ReadGate: ReadGatePyModel}
        self.run_cfg = Loihi2SimCfg(exception_proc_model_map=pdict)
        self.solver._log_config.level = 20

    def test_create_process(self):
        self.assertIsInstance(self.solver, OptimizationSolverProcess)

    @unittest.skip("CPU backend of QUBO solver is temporarily deactivated")
    def test_run_solver(self):
        self.solver.run(RunSteps(5), run_cfg=self.run_cfg)
        self.solver.stop()


class TestFinderReaderIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = QUBO(
            np.array(
                [[-5, 2, 4, 0], [2, -3, 1, 0], [4, 1, -8, 5], [0, 0, 5, -6]]
            )
        )
        self.solution = np.asarray([1, 0, 0, 1]).astype(int)
        from lava.magma.core.process.variable import Var

        cc = {
            1: Var(
                shape=self.problem.cost.coefficients[2].diagonal().shape,
                init=self.problem.cost.coefficients[2].diagonal(),
            ),
            2: Var(
                shape=self.problem.cost.coefficients[2].shape,
                init=self.problem.cost.coefficients[2],
            ),
        }

        # Create processes.
        self.solution_finder = SolutionFinder(
            cost_diagonal=self.problem.cost.coefficients[2].diagonal(),
            cost_coefficients=cc,
            constraints=None,
            hyperparameters={
                "init_state": self.problem.cost.coefficients[2].diagonal()
            },
            discrete_var_shape=(4,),
            continuous_var_shape=None,
            backend="CPU",
            problem=self.problem,
        )

        self.solution_reader = SolutionReader(
            var_shape=(4,), target_cost=0, min_cost=2**24
        )

        # Connect processes.
        self.solution_finder.cost_out_first_byte.connect(
            self.solution_reader.read_gate_in_port_first_byte_0
        )
        self.solution_finder.cost_out_last_bytes.connect(
            self.solution_reader.read_gate_in_port_last_bytes_0
        )
        self.solution_reader.ref_port.connect_var(
            self.solution_finder.variables_assignment
        )

        # Execution configurations.
        pdict = {ReadGate: ReadGatePyModel}
        self.run_cfg = Loihi2SimCfg(exception_proc_model_map=pdict)
        self.solution_finder._log_config.level = 20

    def test_create_process(self):
        self.assertIsInstance(self.solution_finder, SolutionFinder)
        self.assertIsInstance(self.solution_reader, SolutionReader)

    @unittest.skip("CPU backend of QUBO solver is temporarily deactivated")
    def test_run_finder(self):
        self.solution_finder.run(RunSteps(5), run_cfg=self.run_cfg)
        self.solution_finder.stop()


if __name__ == "__main__":
    unittest.main()
