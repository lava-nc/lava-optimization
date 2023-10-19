import numpy as np
import typing as ty
from numpy import typing as npty

from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var


class NEBM(AbstractProcess):
    """
    Non-equilibrium Boltzmann (NEBM) neuron model to solve QUBO problems.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        temperature: ty.Optional[ty.Union[int, npty.NDArray]] = 1,
        refract: ty.Optional[ty.Union[int, npty.NDArray]] = 1,
        refract_counter: ty.Optional[ty.Union[int, npty.NDArray]] = 0,
        init_value=0,
        init_state=0,
        neuron_model: str = 'nebm',
    ):
        """
        NEBM Process.

        Parameters
        ----------
        shape: Tuple
            Number of neurons. Default is (1,).
        temperature: ArrayLike
            Temperature of the system, defining the level of noise.
        refract : ArrayLike
            Minimum number of timesteps a neuron remains in a state after a
            transition.
        init_value : ArrayLike
            The spiking history with which the network is initialized
        init_state : ArrayLike
            The state of neurons with which the network is initialized
        """
        super().__init__(shape=shape)

        self.a_in = InPort(shape=shape)
        self.s_sig_out = OutPort(shape=shape)
        self.s_wta_out = OutPort(shape=shape)

        self.spk_hist = Var(
            shape=shape, init=(np.zeros(shape=shape) + init_value).astype(int)
        )

        self.temperature = Var(shape=shape, init=int(temperature))
        self.refract = Var(shape=shape, init=refract)
        self.refract_counter = Var(shape=shape, init=refract_counter)

        # Initial state determined in DiscreteVariables
        self.state = Var(shape=shape, init=init_state.astype(int))

        @property
        def shape(self) -> ty.Tuple[int, ...]:
            return self.proc_params["shape"]


class NEBMSimulatedAnnealing(AbstractProcess):
    """
    Non-equilibrium Boltzmann (NEBM) neuron model to solve QUBO problems.
    """

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        max_temperature: int,
        min_temperature: int,
        delta_temperature: int,
        steps_per_temperature: int,
        refract_scaling: int,
        refract: ty.Optional[ty.Union[int, npty.NDArray]] = 0,
        exp_temperature=None,
        init_value=0,
        init_state=None,
        neuron_model: str,
        annealing_schedule: str = 'linear',
    ):
        """
        SA Process.

        Parameters
        ----------
        shape: Tuple
            Number of neurons. Default is (1,).


        temperature: ArrayLike
            Temperature of the system, defining the level of noise.
        refract : ArrayLike
            Minimum number of timesteps a neuron remains in a state after a
            transition.
        init_value : ArrayLike
            The spiking history with which the network is initialized
        init_state : ArrayLike
            The state of neurons with which the network is initialized
        """

        super().__init__(
            shape=shape,
            min_temperature=min_temperature,
            delta_temperature=delta_temperature,
            steps_per_temperature=steps_per_temperature,
            refract=refract,
            refract_scaling=refract_scaling,
            exp_temperature=exp_temperature,
            neuron_model=neuron_model,
            annealing_schedule=annealing_schedule,
        )

        self.a_in = InPort(shape=shape)
        self.s_sig_out = OutPort(shape=shape)
        self.s_wta_out = OutPort(shape=shape)

        self.spk_hist = Var(
            shape=shape, init=(np.zeros(shape=shape) + init_value).astype(int)
        )

        self.temperature = Var(shape=shape, init=int(max_temperature))

        self.refract_counter = Var(
            shape=shape,
            init=refract
            + np.right_shift(
                np.random.randint(0, 2**8, size=shape), (refract_scaling or 0)
            ),
        )

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
