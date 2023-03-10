# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later
# See: https://spdx.org/licenses/

import numpy as np
from scipy.optimize import OptimizeResult
from skopt import Optimizer, Space
from skopt.space import Categorical, Integer, Real
from typing import Union

from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

from lava.lib.optimization.solvers.bayesian.processes import (
    BayesianOptimizer
)


@implements(proc=BayesianOptimizer, protocol=LoihiProtocol)
@requires(CPU)
@tag('floating_pt')
class PyBayesianOptimizerModel(PyLoihiProcessModel):
    """
    A Python-based implementation of the Bayesian Optimizer processes. For
    more information, please refer to bayesian/processes.py.
    """
    results_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.float64)
    next_point_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.float64)

    acq_func_config = LavaPyType(np.ndarray, np.ndarray)
    acq_opt_config = LavaPyType(np.ndarray, np.ndarray)
    search_space = LavaPyType(np.ndarray, np.ndarray)
    est_config = LavaPyType(np.ndarray, np.ndarray)
    ip_gen_config = LavaPyType(np.ndarray, np.ndarray)
    num_ips = LavaPyType(int, int)
    num_objectives = LavaPyType(int, int)
    seed = LavaPyType(int, int)

    initialized = LavaPyType(bool, bool)
    num_iterations = LavaPyType(int, int)
    results_log = LavaPyType(np.ndarray, np.ndarray)

    def run_spk(self) -> None:
        """tick the model forward by one time-step"""

        if self.initialized:
            # receive a result vector from the black-box function
            result_vec: np.ndarray = self.results_in.recv()

            opt_result: OptimizeResult = self.process_result_vector(
                result_vec
            )
            self.results_log[0].append(opt_result)
        else:
            # initialize the search space from the standard Bayesian
            # optimization search space schema; for more information,
            # please refer to the init_search_space method
            self.search_space = self.init_search_space()
            self.optimizer = Optimizer(
                dimensions=self.search_space,
                base_estimator=self.est_config[0],
                n_initial_points=self.num_ips,
                initial_point_generator=self.ip_gen_config[0],
                acq_func=self.acq_func_config[0],
                acq_optimizer=self.acq_opt_config[0],
                random_state=self.seed
            )
            self.results_log[0]: list[OptimizeResult] = []
            self.initialized: bool = True
            self.num_iterations: int = -1

        next_point: list = self.optimizer.ask()
        next_point: np.ndarray = np.ndarray(
            shape=(len(self.search_space), 1),
            buffer=np.array(next_point)
        )

        self.next_point_out.send(next_point)
        self.num_iterations += 1

    def __del__(self) -> None:
        """finalize the optimization processing upon runtime conclusion"""

        if hasattr(self, "results_log") and len(self.results_log) > 0:
            print(self.results_log[-1])

    def init_search_space(self) -> list:
        """initialize the search space from the standard schema

        This method is designed to convert the numpy ndarray-based search
        space description int scikit-optimize format compatible with all
        lower-level processes. Your search space should consist of three
        types of parameters:

        1) ("continuous", <min_value>, <max_value>, np.nan, <name>)
        2) ("integer", <min_value>, <max_value>, np.nan, <name>)
        3) ("categorical", np.nan, np.nan, <choices>, <name>)

        Returns
        -------
        search_space : list[Union[Real, Integer]]
            A collection of continuous and discrete dimensions that represent
            the entirety of the problem search space
        """
        search_space: list[Union[Real, Integer]] = []

        for i in range(self.search_space.shape[0]):
            p_type: str = self.search_space[i, 0]
            minimum: Union[int, float] = self.search_space[i, 1]
            maximum: Union[int, float] = self.search_space[i, 2]
            choices: list = self.search_space[i, 3]
            name: str = self.search_space[i, 4]

            factory_function: dict = {
                "continuous": (lambda: Real(minimum, maximum, name=name)),
                "integer": (lambda: Integer(minimum, maximum, name=name)),
                "categorical": (lambda: Categorical(choices, name=name))
            }

            if p_type not in factory_function.keys():
                raise ValueError(
                    f"parameter type [{p_type}] is not in valid "
                    + f"parameter types: {factory_function.keys()}"
                )

            dimension_lambda = factory_function[p_type]
            dimension = dimension_lambda()
            search_space.append(dimension)

        if not len(search_space) > 0:
            raise ValueError("search space is empty")

        return search_space

    def process_result_vector(self, vec: np.ndarray) -> None:
        """parse vec into params/objectives before informing optimizer

        Parameters
        ----------
        vec : np.ndarray
            A single array of data from the black-box process containing
            all parameters and objectives for a total length of num_params
            + num_objectives
        """
        vec: list = vec[:, 0].tolist()

        evaluated_point: list = vec[:-self.num_objectives]
        performance: list = vec[-self.num_objectives:]

        return self.optimizer.tell(evaluated_point, performance[0])
