from itertools import product
import numpy as np
import random
import sys
import unittest

from lava.lib.optimization.solvers.bayesian.processes import (
    BayesianOptimizer,
    BaseObjectiveFunction,
    SingleInputLinearFunction,
    SingleInputNonLinearFunction,
    DualContInputFunction,
)

class TestProcesses(unittest.TestCase):
    """Initialization Tests of all processes of the Bayesian solver
    
    All tests check if the vars are properly assigned in the process abd if
    the ports are shaped properly. Refer to Bayesian models.py to understand
    behaviors.
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
            "categorical", 0, 0, [x/4 for x in range(10)], "categorical_var0"
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

    def test_process_bayesian_optimizer(self) -> None:
        """test initialization of the BayesianOptimizer process"""
        
        all_perms: list[tuple] = product(
            self.valid_acq_func_configs,
            self.valid_acq_opt_configs,
            self.valid_est_configs,
            self.valid_ip_configs
        )

        for idx, perm in enumerate(all_perms):
            with self.subTest(line=f"BO config perm {idx}: {perm}"):
                enable_plotting = bool(random.randint(0, 1))
                num_ips: int = random.randint(1, sys.maxsize)
                num_objectives: int = 1
                log_dir: str = "."
                seed: int = random.randint(-sys.maxsize, sys.maxsize)


                # initialize the optimizer with the given config permutation
                opt = BayesianOptimizer(
                    acq_func_config=perm[0],
                    acq_opt_config=perm[1],
                    search_space=self.valid_ss,
                    enable_plotting=enable_plotting,
                    est_config=perm[2],
                    ip_gen_config=perm[3],
                    num_ips=num_ips,
                    num_objectives=num_objectives,
                    log_dir=log_dir,
                    seed=seed
                )

                # validate the shape of the input/output ports
                self.assertEqual(opt.results_in.shape, (4,1))
                self.assertEqual(opt.next_point_out.shape, (3, 1))

                # check internal var containing the acquisition optimizer's
                # configuration
                config: dict = perm[0]
                sorted_keys: list[str] = sorted(config.keys())
                self.assertEqual(
                    opt.vars.acq_func_config.shape, (len(sorted_keys),)
                )
                self.assertTrue(
                    np.array_equal(
                        opt.vars.acq_func_config.get(),
                        np.array([config[k] for k in sorted_keys])
                    )
                )

                # check internal var containing the acquisition optimizer's
                # configuration
                config: dict = perm[1]
                sorted_keys: list[str] = sorted(config.keys())
                self.assertEqual(
                    opt.vars.acq_opt_config.shape, (len(sorted_keys),)
                )
                self.assertTrue(
                    np.array_equal(
                        opt.vars.acq_opt_config.get(),
                        np.array([config[k] for k in sorted_keys])
                    )
                )

                # check internal var containing the acquisition optimizer's
                # configuration
                config: dict = perm[2]
                sorted_keys: list[str] = sorted(config.keys())
                self.assertEqual(
                    opt.vars.est_config.shape, (len(sorted_keys),)
                )
                self.assertTrue(
                    np.array_equal(
                        opt.vars.est_config.get(),
                        np.array([config[k] for k in sorted_keys])
                    )
                )

                # check internal var containing the acquisition optimizer's
                # configuration
                config: dict = perm[3]
                sorted_keys: list[str] = sorted(config.keys())
                self.assertEqual(
                    opt.vars.ip_gen_config.shape, (len(sorted_keys),)
                )
                self.assertTrue(
                    np.array_equal(
                        opt.vars.ip_gen_config.get(),
                        np.array([config[k] for k in sorted_keys])
                    )
                )

                # check internal var containing the search space
                self.assertEqual(
                    opt.vars.search_space.shape, self.valid_ss.shape
                )
                
                for dim_idx in range(opt.vars.search_space.get().shape[0]):
                    for param_idx in range(opt.vars.search_space.get().shape[1]):
                        if param_idx == 3:
                            continue

                        self.assertEqual(
                            opt.vars.search_space.get()[dim_idx][param_idx],
                            self.valid_ss[dim_idx][param_idx]
                        )

                # check internal var containing enable plotting
                self.assertEqual(opt.vars.enable_plotting.shape, (1,))
                self.assertEqual(
                    opt.vars.enable_plotting.get()[0], enable_plotting
                )

                # check internal var containing log directory
                self.assertEqual(opt.vars.log_dir.shape, (1,))
                self.assertEqual(opt.vars.log_dir.get()[0], log_dir)

                # check internal var containing the number of initial points
                self.assertEqual(opt.vars.num_ips.shape, (1,))
                self.assertEqual(opt.vars.num_ips.get(), num_ips)

                # check internal var containing the number of objectives
                self.assertEqual(opt.vars.num_objectives.shape, (1,))
                self.assertEqual(
                    opt.vars.num_objectives.get(), num_objectives
                )

                # check the internal var containing the seed
                self.assertEqual(opt.vars.seed.shape, (1,))
                self.assertEqual(opt.vars.seed.get(), seed)

                # check the internal var containing the frame-log
                self.assertEqual(opt.vars.frame_log.shape, (1,))
                self.assertIsInstance(
                    opt.vars.frame_log.get()[0],
                    dict
                )

                # check the internal var containing the number of iterations
                self.assertEqual(opt.vars.num_iterations.shape, (1,))
                self.assertEqual(opt.vars.num_iterations.get(), -1)

                # check the interval var containing the initialized boolean
                self.assertEqual(opt.vars.num_iterations.shape, (1,))
                self.assertEqual(opt.vars.initialized.get(), False)

    def test_process_base_objective_func(self) -> None:
        """test initialization of the BaseObjectiveFunction process"""
        
        num_params: int = 5
        num_objectives: int = 2
        
        process = BaseObjectiveFunction(
            num_params=num_params,
            num_objectives=num_objectives
        )

        # checking shape of input and output ports
        self.assertEqual(
            process.x_in.shape,
            (num_params, 1)
        )
        self.assertEqual(
            process.y_out.shape,
            (num_params + num_objectives, 1)
        )

        # checking shape of input variables
        self.assertEqual(
            process.vars.num_params.shape,
            (1,)
        )
        self.assertEqual(
            process.vars.num_objectives.shape,
            (1,)
        )

        # checking values of input_variables
        self.assertEqual(
            process.vars.num_params.get(),
            num_params
        )
        self.assertEqual(
            process.vars.num_objectives.get(),
            num_objectives
        )

    def test_process_single_input_linear_func(self) -> None:
        """test initialization of the SingleInputLinearFunction process"""

        num_params: int = 1
        num_objectives: int = 1
        
        process = SingleInputLinearFunction()

        # checking shape of input and output ports
        self.assertEqual(
            process.x_in.shape,
            (num_params, 1)
        )
        self.assertEqual(
            process.y_out.shape,
            (num_params + num_objectives, 1)
        )

        # checking shape of input variables
        self.assertEqual(
            process.vars.num_params.shape,
            (1,)
        )
        self.assertEqual(
            process.vars.num_objectives.shape,
            (1,)
        )

        # checking values of input_variables
        self.assertEqual(
            process.vars.num_params.get(),
            num_params
        )
        self.assertEqual(
            process.vars.num_objectives.get(),
            num_objectives
        )

    def test_process_single_input_nonlinear_func(self) -> None:
        """test initialization of the SingleInputNonLinearFunction process"""

        num_params: int = 1
        num_objectives: int = 1
        
        process = SingleInputNonLinearFunction()

        # checking shape of input and output ports
        self.assertEqual(
            process.x_in.shape,
            (num_params, 1)
        )
        self.assertEqual(
            process.y_out.shape,
            (num_params + num_objectives, 1)
        )

        # checking shape of input variables
        self.assertEqual(
            process.vars.num_params.shape,
            (1,)
        )
        self.assertEqual(
            process.vars.num_objectives.shape,
            (1,)
        )

        # checking values of input_variables
        self.assertEqual(
            process.vars.num_params.get(),
            num_params
        )
        self.assertEqual(
            process.vars.num_objectives.get(),
            num_objectives
        )

    def test_process_dual_cont_input_func(self) -> None:
        """test initialization of the DualContInputFunction process"""

        num_params: int = 2
        num_objectives: int = 1
        
        process = DualContInputFunction()

        # checking shape of input and output ports
        self.assertEqual(
            process.x_in.shape,
            (num_params, 1)
        )
        self.assertEqual(
            process.y_out.shape,
            (num_params + num_objectives, 1)
        )

        # checking shape of input variables
        self.assertEqual(
            process.vars.num_params.shape,
            (1,)
        )
        self.assertEqual(
            process.vars.num_objectives.shape,
            (1,)
        )

        # checking values of input_variables
        self.assertEqual(
            process.vars.num_params.get(),
            num_params
        )
        self.assertEqual(
            process.vars.num_objectives.get(),
            num_objectives
        )


if __name__ == "__main__":
    unittest.main()