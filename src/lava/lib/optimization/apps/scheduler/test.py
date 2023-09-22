import pprint

import numpy as np
from matplotlib import pyplot as plt

from lava.lib.optimization.apps.scheduler.problems import (
    SchedulingProblem, SatelliteScheduleProblem)
from lava.lib.optimization.apps.scheduler.solver import (Scheduler,
                                                         SatelliteScheduler)


if __name__ == "__main__":
    if True:
        ssp = SatelliteScheduleProblem(num_satellites=3,
                                       num_requests=100,
                                       view_height=0.75)
        ssp.generate(42)
        ssp.plot_problem()

        sat_scheduler = SatelliteScheduler(ssp,
                                           qubo_weights=(1, 8),
                                           probe_loihi_exec_time=True)

        # sat_scheduler.solve_with_netx()
        # print(f'Scheduled {sat_scheduler.netx_solution.shape[0]} Requests.')

        sat_scheduler.qubo_hyperparams = ({"temperature": 1},
                                          True)
        # {"neuron_model": "nebm-sa-refract",
        #  "temperature": 1,
        #  "refract": 64,
        #  "refract_scaling": 1,
        #  "max_temperature": 5,
        #  "min_temperature": 1,
        #  "steps_per_temperature": 200}
        sat_scheduler.lava_backend = "Loihi2"
        sat_scheduler.solve_with_lava_qubo(timeout=1000)
        print(f'Scheduled {sat_scheduler.lava_solution.shape[0]} Requests.')
        plt.semilogy(sat_scheduler.lava_solver_report.profiler.execution_time)
        sat_scheduler.plot_solutions()
    else:
        sp = SchedulingProblem(num_agents=3, num_tasks=3)
        sp.generate(42)

        scheduler = Scheduler(sp, qubo_weights=(4, 20))

        scheduler.solve_with_netx()
        print(f'Scheduled {scheduler.netx_solution.shape[0]} Requests.')
        print(f"{scheduler.netx_solution=}")

        scheduler.qubo_hyperparams = ({"temperature": 1}, True)
        scheduler.lava_backend = "Loihi2"
        scheduler.solve_with_lava_qubo(timeout=1000)
        print(f'Scheduled {scheduler.lava_solution.shape[0]} Requests.')
        print(f"{scheduler.lava_solution=}")
