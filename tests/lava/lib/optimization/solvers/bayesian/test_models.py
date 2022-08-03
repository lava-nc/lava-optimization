import numpy as np
import os
import random
import shutil
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

from lava.lib.optimization.solvers.bayesian.models import (
    BayesianOptimizer,
    DualContInputFunction,
    SingleInputLinearFunction,
    SingleInputNonLinearFunction
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

        self.x_out = OutPort(shape=(num_params,1))
        self.data = Var(shape=(num_params,1), init=spike)


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
        self.y_in = InPort(shape=(perf_vec_length,1))
        self.recv_data = Var(shape=(perf_vec_length,1))


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

    def test_model_bayesian_optimizer(self) -> None:
        """test behavior of the BayesianOptimizer process"""

        search_space: np.ndarray = np.array([
            ["categorical", np.nan, np.nan, [x/4 for x in range(10)], "x1"],
        ], dtype=object)

        log_dir = os.path.join(".", "tests", "temp")

        problem = SingleInputNonLinearFunction()

        optimizer = BayesianOptimizer(
            acq_func_config=self.valid_acq_func_configs[0],
            acq_opt_config=self.valid_acq_opt_configs[0],
            enable_plotting=True,
            search_space=search_space,
            est_config=self.valid_est_configs[0],
            ip_gen_config=self.valid_ip_configs[0],
            log_dir=log_dir,
            num_ips=1,
            num_objectives=1,
            seed=0
        )

        optimizer.next_point_out.connect(problem.x_in)
        problem.y_out.connect(optimizer.results_in)

        # initialize the search space being exactly as it was argued to
        # the solver
        initial_ss: np.ndarray = optimizer.search_space
        self.assertEqual(initial_ss.shape, search_space.shape)
        self.assertEqual(optimizer.initialized.get(), False)
        self.assertEqual(optimizer.num_iterations.get(), -1)
        self.assertEqual(len(optimizer.frame_log.get()[0]), 0)

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

        del problem
        del optimizer

        # verify that all of the valid plots and videos have been created
        valid_files: list[str] = [
            "convergence.mp4", "evaluations.mp4",
            "gaussian_process.mp4", "objective.mp4"
        ]

        for i in range(2):
            valid_files.append(f"convergence_iter{i}.png")
            valid_files.append(f"eval_iter{i}.png")
            valid_files.append(f"gp_iter{i}.png")
            valid_files.append(f"obj_iter{i}.png")

        found_files: list[str] = os.listdir(log_dir)

        self.assertEqual(len(found_files), len(valid_files))
        for f in found_files:
            self.assertIn(f, valid_files)

        shutil.rmtree(log_dir)
 

    def test_model_dual_cont_input_func(self) -> None:
        """test behavior of the DualContInputFunction process"""

        input_spike = np.ndarray((2,1), buffer=np.array([0.1, 0.1]))
        valid_spike = np.array([0.1, 0.1, 1.00540399861])

        input_probe = InputParamVecProcess(num_params=2, spike=input_spike)
        bb_process = DualContInputFunction()
        output_probe = OutputPerfVecProcess(num_params=2,num_objectives=1)

        input_probe.x_out.connect(bb_process.x_in)
        bb_process.y_out.connect(output_probe.y_in)

        output_probe.run(
            condition=RunSteps(num_steps=1),
            run_cfg=Loihi1SimCfg()
        )

        result: np.ndarray = output_probe.recv_data.get()
        self.assertEqual(result[0][0], valid_spike[0])
        self.assertEqual(result[1][0], valid_spike[1])
        self.assertAlmostEqual(result[2][0], valid_spike[2])

        output_probe.stop()

    def test_model_single_input_linear_func(self) -> None:
        """test behavior of the SingleInputLinearFunction process"""
        
        input_spike = np.array([5])
        valid_spike = np.array([5, 49])

        input_probe = InputParamVecProcess(num_params=1, spike=input_spike)
        bb_process = SingleInputLinearFunction()
        output_probe = OutputPerfVecProcess(num_params=1, num_objectives=1)

        input_probe.x_out.connect(bb_process.x_in)
        bb_process.y_out.connect(output_probe.y_in)

        output_probe.run(
            condition=RunSteps(num_steps=1),
            run_cfg=Loihi1SimCfg()
        )

        result: np.ndarray = output_probe.recv_data.get()
        self.assertEqual(result[0][0], valid_spike[0])
        self.assertEqual(result[1][0], valid_spike[1])

        output_probe.stop()

    def test_model_single_input_nonlinear_func(self) -> None:
        """test behavior of the SingleInputNonLinearFunction process"""

        input_spike = np.array([5])
        valid_spike = np.array([5, 0.727989444555])

        input_probe = InputParamVecProcess(num_params=1, spike=input_spike)
        bb_process = SingleInputNonLinearFunction()
        output_probe = OutputPerfVecProcess(num_params=1, num_objectives=1)

        input_probe.x_out.connect(bb_process.x_in)
        bb_process.y_out.connect(output_probe.y_in)

        output_probe.run(
            condition=RunSteps(num_steps=1),
            run_cfg=Loihi1SimCfg()
        )

        result: np.ndarray = output_probe.recv_data.get()
        self.assertEqual(result[0][0], valid_spike[0])
        self.assertAlmostEqual(result[1][0], valid_spike[1])

        output_probe.stop()


if __name__ == "__main__":
    unittest.main()
