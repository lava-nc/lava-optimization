# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: LGPL-2.1-or-later
# See: https://spdx.org/licenses/

import numpy as np
from schema import And, Schema, SchemaError

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

from lava.lib.optimization.solvers.bayesian.models import BayesianOptimizer


class BayesianSolver:
    """
    The BayesianSolver is a class-based interface to abstract the details
    of initializing the BayesianOptimizer process and connecting it with
    the user's specific black-box function
    """
    def __init__(self, acq_func_config: dict, acq_opt_config: dict,
                 ip_gen_config: dict, num_ips: int, seed: int,
                 est_config: dict = {"type": "GP"}, num_objectives: int = 1
                 ) -> None:
        """
        Constructor method for the BayesianSolver class.

        Parameters
        ----------
        acq_func_config : dict
            A dictionary to specify the function to minimize over the posterior
            distribution.
            #{
            #    "type": str
            #        specify the function to minimize over the posterior
            #        distribution:
            #            "LCB" = lower confidence bound
            #            "EI" = negative expected improvement
            #            "PI" = negative probability of improvement
            #            "gp_hedge" = probabilistically determine which of the
            #                aforementioned functions to use at every iteration
            #            "EIps" = negative expected improved with consideration
            #                of the total function runtime
            #            "PIps" = negative probability of improvement
            #                while taking into account the total function
            #                runtime
            #}
        acq_opt_config: dict
            A dictionary to specify the method to minimize the acquisition
            function.
            #{
            #    "type" : str
            #        specify the method to minimize the acquisition function:
            #            "sampling" = random selection from the acquisition
            #                function
            #            "lbfgs" = inverse Hessian matrix estimation
            #            "auto" = automatically configure based on the search
            #                space
            #}
        ip_gen_config : dict
            A dictionary to specify the method to explore the search space
            before the Gaussian regressor starts to converge.
            #{
            #    "type": str
            #        specify the method to explore the search space before the
            #        Gaussian regressor starts to converge:
            #            "random" = uniform distribution of random numbers
            #            "sobol" = Sobol sequence
            #            "halton" = Halton sequence
            #            "hammersly" = Hammersly sequence
            #            "lhs" = latin hypercube sequence
            #            "grid" = uniform grid sequence
            #}
        num_ips : int
            The number of points to explore with the initial point generator
            before using the regressor.
        seed : int
            An integer seed that sets the random state increases consistency
            in subsequent runs.
        est_config : dict, optional
            A dictionary to specify the type of surrogate regressor to learn
            the search space.
            #{
            #    "type": str
            #        specify the type of surrogate regressor to learn the search
            #        space:
            #            "GP" - gaussian process regressor
            #}
        num_objectives : int, optional
            Specify the number of objectives to optimize over; currently
            limited to single objective.
        """

        self.val_init_args(
            acq_func_config=acq_func_config,
            acq_opt_config=acq_opt_config,
            ip_gen_config=ip_gen_config,
            num_ips=num_ips,
            seed=seed,
            est_config=est_config,
            num_objectives=num_objectives
        )

        self.acquisition_function_config: dict = acq_func_config
        self.acquisition_optimizer_config: dict = acq_opt_config
        self.est_config: dict = est_config
        self.ip_gen_config: str = ip_gen_config
        self.num_initial_points: int = num_ips
        self.num_objectives: int = num_objectives
        self.seed: int = seed

    def solve(self, name: str, num_iter: int, problem: AbstractProcess,
              search_space: np.ndarray) -> None:
        """
        Conduct hyperparameter optimization for the argued problem.

        Parameters
        ----------
        name : str
            a unique identifier for the given experiment
        num_iter : int
            the number of Bayesian iterations to conduct
        problem : AbstractProcess
            the black-box function whose parameters are represented by the
            Bayesian optimizer's search space
        search_space : np.ndarray
            At every index, your search space should consist of three types
            of parameters:
            1) ("continuous", <min_value>, <max_value>, np.nan, <name>)
            2) ("integer", <min_value>, <max_value>, np.nan, <name>)
            3) ("categorical", np.nan, np.nan, <choices>, <name>)
        """

        self.val_solve_args(
            name=name,
            num_iter=num_iter,
            problem=problem,
            search_space=search_space,
            num_ips=self.num_initial_points,
            num_objectives=self.num_objectives
        )

        optimizer = BayesianOptimizer(
            acq_func_config=self.acquisition_function_config,
            acq_opt_config=self.acquisition_optimizer_config,
            search_space=search_space,
            est_config=self.est_config,
            ip_gen_config=self.ip_gen_config,
            num_ips=self.num_initial_points,
            num_objectives=self.num_objectives,
            seed=self.seed
        )

        optimizer.next_point_out.connect(problem.x_in)
        problem.y_out.connect(optimizer.results_in)

        optimizer.run(
            condition=RunSteps(num_steps=num_iter),
            run_cfg=Loihi1SimCfg()
        )

        optimizer.stop()

    @staticmethod
    def val_init_args(acq_func_config: dict, acq_opt_config: dict,
                      ip_gen_config: dict, num_ips: int, seed: int,
                      est_config: dict = {"type": "GP"},
                      num_objectives: int = 1) -> None:
        """
        Initialize the BayesianSolver interface.

        Parameters
        ----------
        acq_func_config : dict
            A dictionary to specify the function to minimize over the posterior
            distribution.
            # {
            #     "type": str
            #         specify the function to minimize over the posterior
            #         distribution:
            #             "LCB" = lower confidence bound
            #             "EI" = negative expected improvement
            #             "PI" = negative probability of improvement
            #             "gp_hedge" = probabilistically determine which of the
            #                 aforementioned functions to use at every iteration
            #             "EIps" = negative expected improved with consideration
            #                 of the total function runtime
            #             "PIps" = negative probability of improvement
            #                 while taking into account the total function
            #                 runtime
            # }
        acq_opt_config: dict
            A dictionary to specify the method to minimize the acquisition
            function.
            # {
            #     "type" : str
            #         specify the method to minimize the acquisition function:
            #             "sampling" = random selection from the acquisition
            #                 function
            #             "lbfgs" = inverse Hessian matrix estimation
            #             "auto" = automatically configure based on the search
            #                 space
            # }
        ip_gen_config : dict
            A dictionary to specify the method to explore the search space
            before the Gaussian regressor starts to converge.
            # {
            #     "type": str
            #         specify the method to explore the search space before the
            #         Gaussian regressor starts to converge:
            #             "random" = uniform distribution of random numbers
            #             "sobol" = Sobol sequence
            #             "halton" = Halton sequence
            #             "hammersly" = Hammersly sequence
            #             "lhs" = latin hypercube sequence
            #             "grid" = uniform grid sequence
            # }
        num_ips : int
            The number of points to explore with the initial point generator
            before using the regressor.
        seed : int
            An integer seed that sets the random state increases consistency
            in subsequent runs.
        est_config : dict
            A dictionary to specify the type of surrogate regressor to learn
            the search space.
            #{
            #     "type": str
            #         specify the type of surrogate regressor to learn the
            #         search space:
            #             "GP" - gaussian process regressor
            #}
        num_objectives : int
            Specify the number of objectives to optimize over; currently
            limited to single objective.
        """
        # validate input argument specifying the acquisition function
        # config
        valid_acq_funcs: list[str] = [
            "LCB", "EI", "PI", "gp_hedge", "EIps", "PIps"
        ]
        Schema({
            "type": And(
                lambda x: x in valid_acq_funcs,
                error=f"acq_func not in valid options: {valid_acq_funcs}"
            )
        }).validate(acq_func_config)

        # validate input argument specifying the acquisition optimizer config
        valid_acq_opts: list[str] = ["sampling", "lbfgs", "auto"]
        Schema({
            "type": And(
                lambda x: x in valid_acq_opts,
                error=f"acq_opt not in valid options: {valid_acq_opts}"
            )
        }).validate(acq_opt_config)

        # validate the type of specified surrogate regressor
        valid_estimators: list[str] = ["GP"]
        Schema({
            "type": And(
                lambda x: x in valid_estimators,
                error=f"estimator is not in valid options: {valid_estimators}"
            ),
        }).validate(est_config)

        # validate the argued initial point generator
        valid_ip_gens: list[str] = [
            "random", "sobol", "halton", "hammersly", "lhs", "grid"
        ]
        Schema({
            "type": And(
                lambda x: x in valid_ip_gens,
                error=f"ip generator is not in valid options: {valid_ip_gens}"
            )
        }).validate(ip_gen_config)

        # validate the argued number of initial points
        Schema(
            lambda x: type(x) == int and x > 0,
            error="the number of initial points should be an int > 0"
        ).validate(num_ips)

        # validate the argued number of objectives
        Schema(
            lambda x: type(x) == int and x == 1,
            error="num_objectives should be an int = 1"
        ).validate(num_objectives)

        # validate the argued seed
        Schema(
            lambda x: type(x) == int,
            error="random_state should be an integer"
        ).validate(seed)

    @staticmethod
    def val_solve_args(name: str, num_iter: int, problem: AbstractProcess,
                       search_space: np.ndarray, num_ips: int,
                       num_objectives: int) -> None:
        """validate the arguments associated with the solve method

        Parameters
        ----------
        name : str
            A unique identifier for the given experiment.
        num_iter : int
            The number of Bayesian iterations to conduct.
        problem : AbstractProcess
            The black-box function whose parameters are represented by the
            Bayesian optimizer's search space.
        search_space : np.ndarray
            At every index, your search space should consist of three types
            of parameters:
            1) ("continuous", <min_value>, <max_value>, np.nan, <name>)
            2) ("integer", <min_value>, <max_value>, np.nan, <name>)
            3) ("categorical", np.nan, np.nan, <choices>, <name>)
        num_ips : int
            The number of points to explore with the initial point generator
            before using the regressor.
        num_objectives : int
            Specify the number of objectives to optimize over; currently
            limited to single objective.
        """

        # validate the input argument specifying the name
        Schema(
            lambda x: type(x) == str,
            error="name should be a string object"
        ).validate(name)

        # validate the number of iterations
        Schema(
            lambda x: type(x) == int and x > 0,
            error="num_iter should be an int greater than 0"
        ).validate(num_iter)

        # validate the input argument specifying the problem
        outport_len: int = len(search_space) + num_objectives
        ss_len: int = len(search_space)
        Schema(
            And(
                And(
                    lambda x: issubclass(type(x), AbstractProcess),
                    error='problem should extend AbstractProcess class'
                ),
                And(
                    lambda x: x.in_ports.x_in.shape[0] == ss_len,
                    error='problem\'s ip_port shape should match search'
                          + ' space length'
                ),
                And(
                    lambda x: x.out_ports.y_out.shape[0] == outport_len,
                    error='problem\'s ip_port shape should match search'
                          + ' space length plus the number of objectives'
                )
            )
        ).validate(problem)

        # validate input argument specifying the search space
        Schema(
            lambda x: type(x) == np.ndarray and x.shape[0] > 0,
            error="search space should be a non-empty np.array"
        ).validate(search_space)
        for i in range(search_space.shape[0]):
            dim = search_space[i]
            # ensure the dimension has memory allocated for all possible
            # configuration parameters
            Schema(
                lambda x: x.size == 5,
                error="each search space dimension should have 5 elements"
            ).validate(dim)

            # validate the dimensions name
            Schema(
                lambda x: isinstance(x[4], str),
                error="each dimensions name should a string"
            ).validate(dim)

            # validate the other parameters based on the dimensions type
            if dim[0] in ["continuous", "integer"]:
                Schema(
                    lambda x: all([type(x[i]) == np.float64 for i in [1, 2]]),
                    error="the min and max values should be np.float64"
                ).validate(dim)
            elif dim[0] == "categorical":
                Schema(
                    And(
                        And(
                            lambda x: isinstance(x[3], (np.ndarray, list)),
                            error="choices should be a np.ndarray or list"
                        ),
                        And(
                            lambda x: len(x[3]) > 0,
                            error="choices are empty"
                        )
                    )
                ).validate(dim)
            else:
                raise SchemaError(f"search dimension {dim} is not valid")

        # warn the user if the regressor will never start to optimize
        # with the given number of iterations
        if num_iter <= num_ips:
            print(
                "WARNING: the number of iterations is less than the "
                + "number of initial points; the regressor will never start "
                + "to converge on learned information!!!"
            )
