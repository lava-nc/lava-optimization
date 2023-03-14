# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/
import typing as ty

from lava.magma.core.resources import CPU, Loihi2NeuroCore, NeuroCore

BACKENDS = [CPU, Loihi2NeuroCore, NeuroCore, "Loihi2", "CPU"]
BACKENDS_TYPE = ty.Union[CPU, Loihi2NeuroCore, NeuroCore, str]
HP_TYPE = ty.Union[ty.Dict, ty.List[ty.Dict]]
CPUS = [CPU, "CPU"]
NEUROCORES = [Loihi2NeuroCore, NeuroCore, "Loihi2"]
BACKEND_MSG = f""" was requested as backend. However,
the solver currently supports only Loihi 2 and CPU backends.
These can be specified by calling solve with any of the following:
backend = "CPU"
backend = "Loihi2"
backend = CPU
backend = Loihi2NeuroCore
backend = NeuroCoreS
The explicit resource classes can be imported from
lava.magma.core.resources"""
