import pprint

import numpy as np
from matplotlib import pyplot as plt

from lava.lib.optimization.apps.scheduler.problems import (
    SchedulingProblem, SatelliteScheduleProblem)
from lava.lib.optimization.apps.scheduler.solver import (Scheduler,
                                                         SatelliteScheduler)


if __name__ == "__main__":
    total_exec_time = np.zeros((65,))
    for j, num_req in enumerate(range(100, 750, 10)):
        ssp = SatelliteScheduleProblem(num_satellites=12,
                                       num_requests=num_req)
        ssp.generate(42)
        # ssp.plot_problem()
        sat_scheduler = SatelliteScheduler(ssp,
                                           qubo_weights=(4, 20),
                                           probe_loihi_exec_time=True)

        # sat_scheduler.solve_with_netx()
        # print(f'Scheduled {sat_scheduler.netx_solution.shape[0]} Requests.')

        sat_scheduler.qubo_hyperparams = ({"neuron_model": "nebm-sa-refract",
                                           "temperature": 1,
                                           "refract": 64,
                                           "refract_scaling": 1,
                                           "max_temperature": 5,
                                           "min_temperature": 1,
                                           "steps_per_temperature": 200},
                                          True)
        sat_scheduler.lava_backend = "Loihi2"
        sat_scheduler.solve_with_lava_qubo(timeout=1000)
        print(f'Scheduled {sat_scheduler.lava_solution.shape[0]} Requests.')
        # plt.semilogy(sat_scheduler.lava_solver_report.profiler.execution_time)
        # sat_scheduler.plot_solutions()
        exec_time_series = (sat_scheduler.lava_solver_report.profiler
                            .execution_time)
        total_exec_time[j] = np.sum(exec_time_series) * 1000
    fig = plt.figure(1)
    plt.xlabel("Number of requests", fontdict={"fontsize": 16})
    plt.ylabel("Execution time to solution (ms)", fontdict={"fontsize": 16})
    plt.scatter(np.arange(100, 750, 10), total_exec_time, 15, 'b', 'o')
    plt.tick_params(labelsize=14)
    plt.show()
