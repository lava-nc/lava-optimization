# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import typing as ty
from numpy import typing as npty

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var


class SimulatedAnnealingLocal(AbstractProcess):
    """
    Non-equilibrium Boltzmann (NEBM) neuron model to solve QUBO problems.
    This model uses purely information available at the level of individual
    neurons to decide whether to switch or not, in contrast to the inheriting
    Process NEBMSimulatedAnnealing.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        cost_diagonal: npty.ArrayLike,
        max_temperature: npty.ArrayLike,
        refract_scaling: ty.Union[npty.ArrayLike, None],
        refract_seed: int,
        init_value: npty.ArrayLike,
        init_state: npty.ArrayLike,
    ):
        """
        SA Process.

        Parameters
        ----------
        shape: Tuple
            Number of neurons. Default is (1,).

        refract_scaling : ArrayLike
            After a neuron has switched its binary variable, it remains in a
            refractory state that prevents any variable switching for a
            number of time steps. This number of time steps is determined by
                rand(0, 255) >> refract_scaling
            Refract_scaling thus denotes the order of magnitude of timesteps a
            neuron remains in a state after a transition.
        refract_seed : int
            Random seed to initialize the refractory periods. Allows
            repeatability.
        init_value : ArrayLike
            The spiking history with which the network is initialized
        init_state : ArrayLike
            The state of neurons with which the network is initialized
        neuron_model : str
            The neuron model to be used. The latest list of allowed values
            can be found in NEBMSimulatedAnnealing.enabled_neuron_models.
        """

        super().__init__(
            shape=shape,
            cost_diagonal=cost_diagonal,
            refract_scaling=refract_scaling,
        )

        self.a_in = InPort(shape=shape)
        self.delta_temperature_in = InPort(shape=shape)
        self.control_cost_integrator = InPort(shape=shape)
        self.s_sig_out = OutPort(shape=shape)
        self.s_wta_out = OutPort(shape=shape)
        self.best_state_out = OutPort(shape=shape)

        self.spk_hist = Var(
            shape=shape, init=(np.zeros(shape=shape) + init_value).astype(int)
        )

        self.temperature = Var(shape=shape, init=int(max_temperature))

        np.random.seed(refract_seed)
        self.refract_counter = Var(
            shape=shape,
            init=0 + np.right_shift(
                np.random.randint(0, 2**8, size=shape), (refract_scaling or 0)
            ),
        )
        # Storage for the best state. Will get updated whenever a better
        # state was found
        # Default is all zeros
        self.best_state = Var(shape=shape,
                              init=np.zeros(shape=shape, dtype=int))
        # Initial state determined in DiscreteVariables
        self.state = Var(
            shape=shape,
            init=init_state.astype(int)
            if init_state is not None
            else np.zeros(shape=shape, dtype=int),
        )

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        return self.proc_params["shape"]


class SimulatedAnnealing(SimulatedAnnealingLocal):
    """
    Non-equilibrium Boltzmann (NEBM) neuron model to solve QUBO problems.
    This model combines the switching intentions of all NEBM neurons to
    decide whether to switch or not, to avoid conflicting variable switches.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        cost_diagonal: npty.ArrayLike,
        max_temperature: npty.ArrayLike,
        init_value: npty.ArrayLike,
        init_state: npty.ArrayLike,
    ):
        """
        SA Process.

        Parameters
        ----------
        shape: Tuple
            Number of neurons. Default is (1,).

        refract_scaling : ArrayLike
            After a neuron has switched its binary variable, it remains in a
            refractory state that prevents any variable switching for a
            number of time steps. This number of time steps is determined by
                rand(0, 255) >> refract_scaling
            Refract_scaling thus denotes the order of magnitude of timesteps a
            neuron remains in a state after a transition.
        init_value : ArrayLike
            The spiking history with which the network is initialized
        init_state : ArrayLike
            The state of neurons with which the network is initialized
        neuron_model : str
            The neuron model to be used. The latest list of allowed values
            can be found in NEBMSimulatedAnnealing.enabled_neuron_models.
        """

        super().__init__(
            shape=shape,
            cost_diagonal=cost_diagonal,
            max_temperature=max_temperature,
            refract_scaling=None,
            refract_seed=0,
            init_value=init_value,
            init_state=init_state,
        )

        # number of NEBM neurons that suggest switching in a time step
        self.n_switches_in = InPort(shape=shape)
        # port to notify other NEBM neurons of switching intentions
        self.suggest_switch_out = OutPort(shape=shape)
