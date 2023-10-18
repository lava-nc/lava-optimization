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


class NEBMSimulatedAnnealingLocal(AbstractProcess):
    """
    Non-equilibrium Boltzmann (NEBM) neuron model to solve QUBO problems.
    This model uses purely information available at the level of individual
    neurons to decide whether to switch or not, in contrast to the inheriting
    Process NEBMSimulatedAnnealing.
    """

    enabled_neuron_models = ['nebm-sa-refract']
    deprecated_neuron_models = ['nebm-sa-refract-approx',
                                'nebm-sa-balanced',
                                'nebm-sa-refract-approx-unbalanced']

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        cost_diagonal: npty.ArrayLike,
        max_temperature: int,
        refract_scaling: ty.Optional[int],
        init_value=0,
        init_state=None,
        neuron_model: str,
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

        self._validate_input(neuron_model)
        super().__init__(
            shape=shape,
            cost_diagonal=cost_diagonal,
            refract_scaling=refract_scaling,
            neuron_model=neuron_model,
        )

        self.a_in = InPort(shape=shape)
        self.delta_temperature_in = InPort(shape=shape)
        self.s_sig_out = OutPort(shape=shape)
        self.s_wta_out = OutPort(shape=shape)

        self.spk_hist = Var(
            shape=shape, init=(np.zeros(shape=shape) + init_value).astype(int)
        )

        self.temperature = Var(shape=shape, init=int(max_temperature))

        self.refract_counter = Var(
            shape=shape,
            init=0 + np.right_shift(
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

    def _validate_input(self, neuron_model: str) -> None:
        """Validates input. At the moment, it only checks that the user has
        chosen a supported neuron model is """

        if neuron_model in self.enabled_neuron_models:
            return
        elif neuron_model in self.deprecated_neuron_models:
            raise NotImplementedError(
                f"The model {neuron_model} has been deprecated. Instead, we "
                f"recommend switching to the new neuron model 'nebm-sa-refract'"
                f" for a better solver performance.")
        else:
            raise ValueError(
                f"The model {neuron_model} does not exist. Please specify a "
                f"correct neuron model as hyperparameter.")

    @property
    def shape(self) -> ty.Tuple[int, ...]:
        return self.proc_params["shape"]


class NEBMSimulatedAnnealing(NEBMSimulatedAnnealingLocal):
    """
    Non-equilibrium Boltzmann (NEBM) neuron model to solve QUBO problems.
    This model combines the switching intentions of all NEBM neurons to
    decide whether to switch or not, to avoid conflicting variable switches.
    """

    enabled_neuron_models = ['nebm-sa', 'nebm-sa-refract']

    def __init__(
        self,
        *,
        shape: ty.Tuple[int, ...],
        cost_diagonal: npty.ArrayLike,
        max_temperature: int,
        init_value=0,
        init_state=None,
        neuron_model: str,
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
            refract_scaling=None,
            max_temperature=max_temperature,
            init_value=init_value,
            init_state=init_state,
            neuron_model=neuron_model,
        )

        # number of NEBM neurons that suggest switching in a time step
        self.n_switches_in = InPort(shape=shape)
        # port to notify other NEBM neurons of switching intentions
        self.suggest_switch_out = OutPort(shape=shape)
