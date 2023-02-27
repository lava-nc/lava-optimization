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

    def __init__(self,
                 *,
                 shape: ty.Tuple[int, ...],
                 temperature: ty.Optional[ty.Union[int, npty.NDArray]] = 1,
                 refract: ty.Optional[ty.Union[int, npty.NDArray]] = 0,
                 init_value=0,
                 init_state=0):
        """
         NEBM Process.

         Parameters
         ----------
         shape: Tuple
             Number of neurons. Default is (1,).
         temperature: ArrayLike
             Temperature of the system, defining the level of noise.
         refract : ArrayLike
             Refractory period for each neuron. This is the time for which a
             neuron 'stays ON'.
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

        # Initial state determined in DiscreteVariables
        self.state = Var(shape=shape, init=init_state.astype(int))

        @property
        def shape(self) -> ty.Tuple[int, ...]:
            return self.proc_params['shape']


class NEBMSA(AbstractProcess):
    """
    Non-equilibrium Boltzmann (NEBM) neuron model to solve QUBO problems.
    """

    def __init__(self,
                 *,
                 shape: ty.Tuple[int, ...],
                 max_temperature: int,
                 min_temperature: int,
                 delta_temperature: int,
                 steps_per_temp: int,
                 refract: ty.Optional[ty.Union[int, npty.NDArray]] = 0,
                 init_value=0,
                 init_state=None
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
             Refractory period for each neuron. This is the time for which a
             neuron 'stays ON'.
         init_value : ArrayLike
             The spiking history with which the network is initialized
         init_state : ArrayLike
             The state of neurons with which the network is initialized
         """
        super().__init__(shape=shape,
                         max_temperature=max_temperature,
                         min_temperature=min_temperature,
                         delta_temperature=delta_temperature,
                         steps_per_temp=steps_per_temp)

        self.a_in = InPort(shape=shape)
        self.s_sig_out = OutPort(shape=shape)
        self.s_wta_out = OutPort(shape=shape)

        self.spk_hist = Var(
            shape=shape, init=(np.zeros(shape=shape) + init_value).astype(int)
        )

        self.temperature = Var(shape=shape, init=int(max_temperature))

        self.refract = Var(shape=shape, init=refract)

        # Initial state determined in DiscreteVariables
        self.state = Var(shape=shape, init=(init_state or np.array([0])).astype(
            int))

        @property
        def shape(self) -> ty.Tuple[int, ...]:
            return self.proc_params['shape']
