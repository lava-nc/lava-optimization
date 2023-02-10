# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import numpy as np
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyOutPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.lib.optimization.solvers.generic.read_gate.process import ReadGate
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.lib.optimization.solvers.generic import sconfig


def readgate_post_guard(self):
	"""Decide whether to run post management phase."""
	return True if self.min_cost else False


def readgate_run_spk(self):
	"""Execute spiking phase, integrate input, update dynamics and
	send messages out."""
	in_ports = [port for port in self.py_ports if
				issubclass(type(port), PyInPort)]
	costs = []
	for port in in_ports:
		costs.append(port.recv()[0])
	cost = min(costs)
	id = costs.index(cost)

	if self.solution is not None:
		timestep = - np.array([self.time_step])
		if self.min_cost <= self.target_cost:
			self._req_pause = True
		self.cost_out.send(np.array([self.min_cost, self.min_cost_id]))
		self.send_pause_request.send(timestep)
		self.solution_out.send(self.solution)
		self.solution = None
		self.min_cost = None
		self.min_cost_id = None
	else:
		self.cost_out.send(np.array([0, 0]))
	if cost:
		self.min_cost = cost
		self.min_cost_id = id


def readgate_run_post_mgmt(self):
	"""Execute post management phase."""
	self.solution = self.solution_reader.read()


def get_readgate_members(num_in_ports):
	in_ports = {f"cost_in_{id}": LavaPyType(PyInPort.VEC_DENSE, np.int32,
											precision=32) for id in
				range(num_in_ports)}
	readgate_members = {
			"target_cost"       : LavaPyType(int, np.int32, 32),
			"best_solution"     : LavaPyType(int, np.int32, 32),
			"cost_out"          : LavaPyType(
					PyOutPort.VEC_DENSE,
					np.int32,
					precision=32),
			"solution_out"      : LavaPyType(
					PyOutPort.VEC_DENSE,
					np.int32,
					precision=32),
			"send_pause_request": LavaPyType(
					PyOutPort.VEC_DENSE,
					np.int32,
					precision=32),
			"solution_reader"   : LavaPyType(
					PyRefPort.VEC_DENSE,
					np.int32,
					precision=32),
			"min_cost"          : None,
			"min_cost_id"       : None,
			"solution"          : None,

			"post_guard"        : readgate_post_guard,
			"run_spk"           : readgate_run_spk,
			"run_post_mgmt"     : readgate_run_post_mgmt
			}
	readgate_members.update(in_ports)
	return readgate_members


"""CPU model for the ReadGate process.

The model verifies if better payload (cost) has been notified by the
downstream processes, if so, it reads those processes state and sends out to
the upstream process the new payload (cost) and the network state.
"""
ReadGatePyModelBase = type("ReadGatePyModel",
						   (PyLoihiProcessModel,),
						   get_readgate_members(sconfig.num_in_ports)
						   )
ReadGatePyModelImpl = implements(ReadGate, protocol=LoihiProtocol)(
		ReadGatePyModelBase)
ReadGatePyModel = requires(CPU)(ReadGatePyModelImpl)
