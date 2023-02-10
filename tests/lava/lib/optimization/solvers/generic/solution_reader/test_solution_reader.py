# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty
import unittest

import numpy as np
from lava.lib.optimization.problems.problems import OptimizationProblem, QUBO
from lava.lib.optimization.solvers.generic.builder import (
    SolutionReader,
    SolutionFinder
    )
from lava.lib.optimization.solvers.generic.read_gate.models import \
    ReadGatePyModel
from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.process import AbstractProcess, LogConfig
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps, RunContinuous
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.spiker.models import SpikerModel
from lava.proc.spiker.process import Spiker
from numpy import typing as npt


class Mock:
	def __init__(self, val):
		self._val = val

	@property
	def init(self):
		return self._val


class OptimizationSolverProcess(AbstractProcess):
	def __init__(self,
				 problem: OptimizationProblem,
				 hyperparameters: ty.Dict[str, ty.Union[int, npt.ArrayLike]],
				 name: ty.Optional[str] = None,
				 log_config: ty.Optional[LogConfig] = None) -> None:
		super().__init__(hyperparameters=hyperparameters,
						 name=name,
						 log_config=log_config)
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
		init_value = proc.hyperparameters.get("init_value",
											  np.zeros((4,), dtype=int))
		cost_coefficients = {
				1: Mock(q_diag),
				2: Mock(q_off_diag)
				}

		self.solution_reader = SolutionReader(var_shape=(4,),
											  target_cost=target_cost,
											  min_cost=2 ** 24)

		self.finder = SolutionFinder(cost_diagonal=cost_diagonal,
									 cost_coefficients=cost_coefficients,
									 constraints=constraints,
									 hyperparameters=hyperparameters,
									 discrete_var_shape=(4,),
									 continuous_var_shape=None, )

		# Connect processes
		self.finder.cost_out.connect(self.solution_reader.read_gate_in_port_0)
		self.solution_reader.ref_port.connect_var(
			self.finder.variables_assignment)


class OptimizationSolverProcess2:
	def __init__(self,
				 problem: OptimizationProblem,
				 hyperparameters: ty.Dict[str, ty.Union[int, npt.ArrayLike]],
				 name: ty.Optional[str] = None,
				 log_config: ty.Optional[LogConfig] = None) -> None:
		target_cost = 0
		q_off_diag = problem.cost.coefficients[2]
		q_diag = problem.cost.coefficients[2].diagonal()

		init_value = hyperparameters.get("init_value",
										 np.zeros((4,), dtype=int))
		cost_coefficients = {
				1: Mock(q_diag),
				2: Mock(q_off_diag)
				}

		self.solution_reader = SolutionReader(var_shape=(4,),
											  target_cost=target_cost,
											  min_cost=2 ** 24)

		self.finder = SolutionFinder(var_shape=(4,),
									 cost_diagonal=q_diag,
									 cost_coefficients=cost_coefficients,
									 constraints=None,
									 hyperparameters=hyperparameters,
									 discrete_var_shape=(4,),
									 continuous_var_shape=None, )

		# Connect processes
		self.finder.cost_out.connect(self.solution_reader.read_gate_in_port)
		self.solution_reader.ref_port.connect_var(
			self.finder.variables_assignment)


class TestSolutionReader(unittest.TestCase):
	def setUp(self) -> None:
		# Create processes.
		spiker = Spiker(shape=(4,), period=3, payload=7)
		integrator = Spiker(shape=(1,), period=7, payload=-4)

		self.solution_reader = SolutionReader(var_shape=(4,),
											  target_cost=0,
											  min_cost=2)

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


class TestSolutionFinder(unittest.TestCase):
	def setUp(self) -> None:
		self.problem = QUBO(
				np.array([[-5, 2, 4, 0],
						  [2, -3, 1, 0],
						  [4, 1, -8, 5],
						  [0, 0, 5, -6]]))
		self.solution = np.asarray([1, 0, 0, 1]).astype(int)
		from lava.magma.core.process.variable import Var

		cc = {
				1: Var(shape=self.problem.cost.coefficients[2].diagonal(

                        ).shape,
					   init=self.problem.cost.coefficients[2].diagonal()),
				2: Var(shape=self.problem.cost.coefficients[2].shape,
					   init=self.problem.cost.coefficients[2])
				}

		# Create processes.
		self.solution_finder = SolutionFinder(
				cost_diagonal=self.problem.cost.coefficients[
					2].diagonal(),
				cost_coefficients=cc,
				constraints=None,
				hyperparameters={},
				discrete_var_shape=(4,),
				continuous_var_shape=None)

		# Execution configurations.
		pdict = {ReadGate: ReadGatePyModel}
		self.run_cfg = Loihi2SimCfg(exception_proc_model_map=pdict)
		self.solution_finder._log_config.level = 20

	def test_create_process(self):
		self.assertIsInstance(self.solution_finder, SolutionFinder)

	def test_run(self):
		self.solution_finder.run(RunSteps(5), run_cfg=self.run_cfg)
		self.solution_finder.stop()

	#
	# def test_cloning(self):
	#     clone = self.solution_finder.clone({"p1":3})
	#     self.assertEqual(clone.hyperparameters["p1"],3)
	#     clone = self.solution_finder.clone({"p1": 5})
	#     self.assertEqual(clone.hyperparameters["p1"], 5)
	#     clones=[]
	#     for i in range(9):
	#         clone=self.solution_finder.clone({"a":i})
	#         clones.append(clone)
	#
	#     for i in range(9):
	#         self.assertEqual(clones[i].hyperparameters["a"],i)
	#     self.solution_finder.run(RunSteps(1), run_cfg=self.run_cfg)


class TestFinderReaderIntegration(unittest.TestCase):
	def setUp(self) -> None:
		self.problem = QUBO(
				np.array([[-5, 2, 4, 0],
						  [2, -3, 1, 0],
						  [4, 1, -8, 5],
						  [0, 0, 5, -6]]))
		self.solution = np.asarray([1, 0, 0, 1]).astype(int)
		from lava.magma.core.process.variable import Var

		cc = {
				1: Var(shape=self.problem.cost.coefficients[2].diagonal(

                        ).shape,
					   init=self.problem.cost.coefficients[2].diagonal()),
				2: Var(shape=self.problem.cost.coefficients[2].shape,
					   init=self.problem.cost.coefficients[2])
				}

		# Create processes.
		self.solution_finder = SolutionFinder(
				cost_diagonal=self.problem.cost.coefficients[
					2].diagonal(),
				cost_coefficients=cc,
				constraints=None,
				hyperparameters={
						"init_state": self.problem.cost.coefficients[
							2].diagonal()
						},
				discrete_var_shape=(4,),
				continuous_var_shape=None)

		self.solution_reader = SolutionReader(var_shape=(4,),
											  target_cost=0,
											  min_cost=2 ** 24)

		# Connect processes.
		self.solution_finder.cost_out.connect(
				self.solution_reader.read_gate_in_port_0)
		self.solution_reader.ref_port.connect_var(
				self.solution_finder.variables_assignment)

		# Execution configurations.
		pdict = {ReadGate: ReadGatePyModel}
		self.run_cfg = Loihi2SimCfg(exception_proc_model_map=pdict)
		self.solution_finder._log_config.level = 20

	def test_create_process(self):
		self.assertIsInstance(self.solution_finder, SolutionFinder)
		self.assertIsInstance(self.solution_reader, SolutionReader)

	def test_run_finder(self):
		self.solution_finder.run(RunSteps(5), run_cfg=self.run_cfg)
		self.solution_finder.stop()

	def test_run_reader(self):
		self.solution_reader.run(RunSteps(5), run_cfg=self.run_cfg)
		self.solution_reader.stop()


class TestOptSolverWrapper(unittest.TestCase):
	def setUp(self) -> None:
		self.problem = QUBO(
				np.array([[-5, 2, 4, 0],
						  [2, -3, 1, 0],
						  [4, 1, -8, 5],
						  [0, 0, 5, -6]]))
		self.solution = np.asarray([1, 0, 0, 1]).astype(int)

		self.solver = OptimizationSolverProcess(problem=self.problem,
												hyperparameters={
														"init_state":
															self.problem.cost.coefficients[
																2].diagonal()
														})

		# Execution configurations.
		pdict = {ReadGate: ReadGatePyModel}
		self.run_cfg = Loihi2SimCfg(exception_proc_model_map=pdict)
		self.solver._log_config.level = 20

	def test_create_process(self):
		self.assertIsInstance(self.solver, OptimizationSolverProcess)

	def test_run_solver(self):
		self.solver.run(RunSteps(5), run_cfg=self.run_cfg)
		self.solver.stop()

	def test_run_solver(self):
		self.solver.run(RunSteps(5), run_cfg=self.run_cfg)
		self.solver.stop()
