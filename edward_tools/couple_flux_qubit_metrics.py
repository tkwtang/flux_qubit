import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from numba import njit
import datetime
import json
from edward_tools.visualization import separate_by_state_2

def fidelityEvaluation(state_1, state_2, mapping_state_1_to_state_2_dict = None):
    # input: a list to define what are successful transitions
    # e.g. {"01": [11], "00": [10], "10": [10], "11": [11]}
    # check if the initial position and final positions of the particles are both true according to the
    # above mapping_state_1_to_state_2_dict
    state_1_index = separate_by_state_2(state_1)
    state_2_index = separate_by_state_2(state_2)
    fidelityInformation = []
    fidelitySummaryText = ""

    for key, destination_list in mapping_state_1_to_state_2_dict.items():
        initial_count = np.sum(state_1_index[key])
        goodNumber = 0
        fidelityItem = {"initial": {"location": key, "count":  initial_count}, "final": []}
        result = []
        for location in destination_list: # location is the final position that we want the particle to land on
            final_count = np.sum(np.all(np.vstack([state_1_index[key], state_2_index[location]]), axis=0))

            fidelityItem["final"].append({"location": location, "count": final_count})
            # print(f"initial: {key} ({initial_count}), final: {location} ({final_count})")
        fidelityInformation.append(fidelityItem)
    return fidelityInformation

def get_work_distribution(simRunner):
    all_states = simRunner.sim.output.all_state["states"]
    nsteps = all_states.shape[0]
    ntrials =  all_states.shape[1]
    time_index_array = simRunner.sim.output.all_state["step_indices"]
    time_step = time_index_array.step
    work_time_series = np.empty([ntrials, nsteps])

    for n in time_index_array:
        i = int(n / time_step)
        coordinates = all_states[:, i, ...]
        U_i = simRunner.system.get_potential(coordinates, n * simRunner.sim.dt)
        U_f = simRunner.system.get_potential(coordinates, (n + 1) * simRunner.sim.dt)
        work_time_series[i] = U_f - U_i

    work_distribution = np.sum(work_time_series, axis = 0)
    return work_distribution

def work_statistic_graph(work_mean, work_std, protocol_list, skip_step = 2000):
    step_array = np.arange(0, work_mean.shape[0])
    plt.figure(figsize=(10, 7))
    plt.errorbar(step_array[::skip_step], work_mean[::skip_step], yerr = work_std[::skip_step])
    substep_array = np.cumsum([substep["duration"]/cfqr.sim.dt for substep in protocol_list])

    for _t in substep_array[:-1]:
        plt.vlines(x=_t, ymin = 0, ymax = 70, ls="--", colors = "purple")
