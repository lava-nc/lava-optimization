# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later
# See: https://spdx.org/licenses/

import numpy as np
import random
import unittest

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.optimization.problems.bayesian.models import (
    SingleInputFunction
)
from lava.lib.optimization.solvers.bayesian.models import (
    BayesianOptimizer
)


class InputParamVecProcess(AbstractProcess):
    def __init__(self, num_params: int, spike: np.ndarray, **kwargs) -> None:
        """Process to set an input parameter vector to evaluate black-box
        function accuracy

        num_params : int
            the number of parameters to send to the test function
        spike : np.ndarray
            the parameter vector to send to the black-box process
        """
        super().__init__(**kwargs)

        self.x_out = OutPort(shape=(num_params, 1))
        self.data = Var(shape=(num_params, 1), init=spike)


class OutputPerfVecProcess(AbstractProcess):
    def __init__(self, num_params: int, num_objectives: int,
                 **kwargs) -> None:
        """Process to validate the resulting performance vector from the
        black-box function

        num_params : int
            the number of parameters within each performance vector
        num_objectives : int
            the number of objectives within each performance vector
        valid_spike : np.ndarray
            the expected performance vector
        """
        super().__init__(**kwargs)

        perf_vec_length: int = num_params + num_objectives
        self.y_in = InPort(shape=(perf_vec_length, 1))
        self.recv_data = Var(shape=(perf_vec_length, 1))


@implements(proc=InputParamVecProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyInputParamVecModel(PyLoihiProcessModel):
    x_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)
    data: np.ndarray = LavaPyType(np.ndarray, np.float64)

    def run_spk(self) -> None:
        """send the test data to the black-box process"""
        self.x_out.send(self.data)


@implements(proc=OutputPerfVecProcess, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputPerfVecProcess(PyLoihiProcessModel):
    y_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    recv_data: np.ndarray = LavaPyType(np.ndarray, np.float64)

    def run_spk(self) -> None:
        """receive the result vector from the black-box process"""
        self.recv_data = self.y_in.recv()


class TestModels(unittest.TestCase):
    """Tests all model behaviors associated with the Bayesian solver

    Refer to Bayesian models.py to learn more about behaviors.
    """

    def setUp(self) -> None:
        """set up general parameters for process tests"""

        # set default seed for consistent test runs
        random.seed(0)

        # create a variety of valid acquisition function configs
        valid_acq_funcs: list[str] = [
            "LCB", "EI", "PI", "gp_hedge", "EIps", "PIps"
        ]
        self.valid_acq_func_configs: list[dict] = [
            {"type": t} for t in valid_acq_funcs
        ]

        # create a variety of valid acquisition optimizer configs
        valid_acq_opts: list[str] = ["sampling", "lbfgs"]
        self.valid_acq_opt_configs: list[dict] = [
            {"type": t} for t in valid_acq_opts
        ]

        # create valid examples for all types of search space dimensions
        self.valid_continuous_dimension = np.array([
            "continuous", -20.0, 20.0, 0, "continuous_var0"
        ], dtype=object)
        self.valid_integer_dimension = np.array([
            "integer", -10, 10, 0, "discrete_var0"
        ], dtype=object)
        self.valid_categorical_dimension = np.array([
            "categorical", 0, 0, [x / 4 for x in range(10)], "cat_var0"
        ], dtype=object)
        self.valid_ss = np.array([
            self.valid_continuous_dimension,
            self.valid_integer_dimension,
            self.valid_categorical_dimension
        ], dtype=object)

        # create a variety of valid estimator configs
        valid_ests: list[str] = ["GP", "RF", "ET", "GBRT"]
        self.valid_est_configs: list[dict] = [
            {"type": t} for t in valid_ests
        ]

        # create a variety of valid initial point generator configs
        valid_ips: list[str] = [
            "random", "sobol", "halton", "hammersly", "lhs", "grid"
        ]
        self.valid_ip_configs: list[dict] = [
            {"type": t} for t in valid_ips
        ]

    @unittest.skip("Failing due to a change in numpy, to be investaget further")
    def test_model_bayesian_optimizer(self) -> None:
        """test behavior of the BayesianOptimizer process"""

        search_space: np.ndarray = np.array([
            ["categorical", np.nan, np.nan, [x / 4 for x in range(10)], "x1"],
        ], dtype=object)

        problem = SingleInputFunction()

        optimizer = BayesianOptimizer(
            acq_func_config=self.valid_acq_func_configs[0],
            acq_opt_config=self.valid_acq_opt_configs[0],
            search_space=search_space,
            est_config=self.valid_est_configs[0],
            ip_gen_config=self.valid_ip_configs[0],
            num_ips=1,
            num_objectives=1,
            seed=0
        )

        optimizer.next_point_out.connect(problem.x_in)
        problem.y_out.connect(optimizer.results_in)

        # validate the search space is exactly as it was argued to
        # the solver
        initial_ss: np.ndarray = optimizer.search_space
        self.assertEqual(initial_ss.shape, search_space.shape)
        self.assertEqual(optimizer.initialized.get(), False)
        self.assertEqual(optimizer.num_iterations.get(), -1)
        self.assertEqual(len(optimizer.results_log.get()[0]), 1)

        # run the initial step which serves to initialize the optimizer and
        # the associated data attributes
        optimizer.run(
            condition=RunSteps(num_steps=3),
            run_cfg=Loihi1SimCfg(select_sub_proc_model=True)
        )

        # verify that the optimizer and the associated data attributes have
        # been initialized
        self.assertEqual(optimizer.initialized.get(), True)
        self.assertEqual(optimizer.num_iterations.get(), 2)

        optimizer.stop()


if __name__ == "__main__":
    unittest.main()
