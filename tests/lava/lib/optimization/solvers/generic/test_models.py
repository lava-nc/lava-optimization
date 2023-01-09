# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import unittest

import numpy as np
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.resources import CPU
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.proc.monitor.process import Monitor
from lava.lib.optimization.solvers.generic.sub_process_models import (
    StochasticIntegrateAndFireModelSCIF,
    BoltzmannAbstractModel
)
from lava.lib.optimization.solvers.generic.hierarchical_processes import (
    StochasticIntegrateAndFire, BoltzmannAbstract
)


class VecSendProcess(AbstractProcess):
    """
    Process of a user-defined shape that sends an arbitrary vector

    Parameters
    ----------
    shape: tuple, shape of the process
    vec_to_send: np.ndarray, vector of spike values to send
    send_at_times: np.ndarray, vector bools. Send the `vec_to_send` at times
    when there is a True
    """

    def __init__(self, **kwargs):
        super().__init__()
        shape = kwargs.pop("shape", (1,))
        vec_to_send = kwargs.pop("vec_to_send")
        send_at_times = kwargs.pop("send_at_times")
        num_steps = kwargs.pop("num_steps", 1)
        self.shape = shape
        self.num_steps = num_steps
        self.vec_to_send = Var(shape=shape, init=vec_to_send)
        self.send_at_times = Var(shape=(num_steps,), init=send_at_times)
        self.s_out = OutPort(shape=shape)


@implements(proc=VecSendProcess, protocol=LoihiProtocol)
@requires(CPU)
# need the following tag to discover the ProcessModel using LifRunConfig
@tag("floating_pt")
class PyVecSendModelFloat(PyLoihiProcessModel):
    s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    vec_to_send: np.ndarray = LavaPyType(np.ndarray, float)
    send_at_times: np.ndarray = LavaPyType(np.ndarray, bool, precision=1)

    def run_spk(self):
        """
        Send `spikes_to_send` if current time-step requires it
        """
        if self.send_at_times[self.time_step - 1]:
            self.s_out.send(self.vec_to_send)
        else:
            self.s_out.send(np.zeros_like(self.vec_to_send))


def set_up(self, var="state", input_time=None, input_val=2 ** 7, **kwargs):
    self.bit = StochasticIntegrateAndFire(**kwargs)
    if input_time:
        shape = kwargs["shape"]
        times1 = np.zeros((self.steps,))
        times1[input_time] = 1
        times2 = np.zeros((self.steps,))
        times2[input_time + 1] = 1

        sps1 = VecSendProcess(
            shape=shape,
            num_steps=self.steps,
            vec_to_send=[-input_val],
            send_at_times=times1,
        )
        sps1.s_out.connect(self.bit.added_input)
        sps2 = VecSendProcess(
            shape=shape,
            num_steps=self.steps,
            vec_to_send=[input_val],
            send_at_times=times2,
        )
        sps2.s_out.connect(self.bit.added_input)
    self.monitor = Monitor()
    self.monitor.probe(getattr(self.bit, var), num_steps=self.steps)
    self.bit.run(
        condition=RunSteps(self.steps),
        run_cfg=Loihi2SimCfg(exception_proc_model_map={
            StochasticIntegrateAndFire: StochasticIntegrateAndFireModelSCIF}),
    )

    data = self.monitor.get_data()
    self.bit.stop()
    return data


class TestStochasticIntegrateAndFire(unittest.TestCase):
    def setUp(self) -> None:
        self.kwargs = dict(
            shape=(1,),
            steps_to_fire=5,
            step_size=100,
            init_value=0,
            noise_amplitude=0,
            refractory_period=1,
        )
        self.steps = 22

    def tearDown(self) -> None:
        self.bit.stop()

    def test_noiseless_state_progression(self):
        self.data = set_up(self, var="state", **self.kwargs)
        state_progression = self.data[self.bit.name]["state"]
        expected = np.tile(np.arange(0, 401, 100), self.steps // 4)[
            1:self.steps + 1
        ][None].T
        self.assertTrue((state_progression == expected).all())

    def test_noiseless_firing(self):
        self.data = set_up(self, var="messages", **self.kwargs)
        spike_vector = self.data[self.bit.name]["messages"]
        expected = np.tile(np.arange(0, 401, 100), self.steps // 4)[
            1 : self.steps + 1
        ][None].T
        expected = np.where(expected == 0, 1, 0)
        self.assertTrue((spike_vector == expected).all())

    def test_noisy_state_progression(self):
        ns_am = 1
        ns_st = 4
        np.random.seed(1)
        rand_nums = \
            np.random.randint(0, (2 ** 16) - 1, size=(self.steps,))
        # Assign random numbers only to neurons, for which noise is enabled
        prand = np.right_shift((rand_nums * ns_am).astype(int), 16 - ns_st)
        expected = np.cumsum(prand)
        expected += np.arange(100, (self.steps + 1) * 100, 100)

        self.kwargs.update(
            dict(noise_amplitude=ns_am, name="Process_2", steps_to_fire=1000,
                 noise_precision=ns_st)
        )
        np.random.seed(1)
        self.data = set_up(self, "state", **self.kwargs)
        state = self.data[self.bit.name]["state"]
        self.assertTrue((state.T == expected).all())

    def test_noisy_firing(self):
        expected = np.array([[1., 0., 1., 0., 1., 1., 1., 0., 1., 0.,
                              1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 0., 1.]])
        self.kwargs.update(dict(noise_amplitude=10, name="Process_2",
                                noise_precision=6))
        np.random.seed(1)
        self.data = set_up(self, var="messages", **self.kwargs)
        spike_vector = self.data[self.bit.name]["messages"]
        self.assertTrue((spike_vector.T == expected).all())

    def test_noiseless_state_progression_with_input(self):
        self.data = set_up(self, var="state", input_time=7, **self.kwargs)
        state_progression = self.data[self.bit.name]["state"]
        expected = np.array(
            [
                [
                    100.0,
                    200.0,
                    300.0,
                    400.0,
                    0.0,
                    100.0,
                    200.0,
                    172.0,
                    400.0,
                    0.0,
                    100.0,
                    200.0,
                    300.0,
                    400.0,
                    0.0,
                    100.0,
                    200.0,
                    300.0,
                    400.0,
                    0.0,
                    100.0,
                    200.0
                ]
            ]
        )
        self.assertTrue((state_progression.T == expected).all())

    @unittest.skip("Not necessary for QUBO")
    def test_noiseless_satisfiability_with_input(self):
        self.data = set_up(
            self,
            var="satisfiability",
            input_time=11,
            input_val=10,
            **self.kwargs
        )
        state_progression = self.data[self.bit.name]["satisfiability"]
        expected = np.array(
            [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
        )

        self.assertTrue((state_progression.T == expected).all())

        if __name__ == "__main__":
            unittest.main()
