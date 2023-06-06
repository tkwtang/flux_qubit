import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from numba import njit
import datetime
import json, hashlib
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
        fidelityItem = {"initial": {"location": key, "count":  int(initial_count)}, "final": []}
        result = []
        for location in destination_list: # location is the final position that we want the particle to land on
            final_count = np.sum(np.all(np.vstack([state_1_index[key], state_2_index[location]]), axis=0))

            fidelityItem["final"].append({"location": location, "count": int(final_count)})
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
        plt.vlines(x=_t, ymin = np.min(work_mean), ymax = np.max(work_mean), ls="--", colors = "purple")


def truncateFinalW(final_W):
    sorted_final_W = np.sort(final_W)
    naive_jarzynski = np.mean(np.exp(-np.sort(final_W)))
    truncated_jarzynski = 0.0
    truncated_part = [0.0]

    if naive_jarzynski > 1:
        for x in range(0, len(final_W)):
            truncated_jarzynski = np.mean(np.exp(-np.sort(final_W)[x:]))
            truncated_jarzynski_next = np.mean(np.exp(-np.sort(final_W)[x+1:]))
            truncated_part = sorted_final_W[:x+1]
            remaining_part = sorted_final_W[x+1:]
            if truncated_jarzynski_next < 1 and truncated_jarzynski > 1:
                break
    else:
        for x in range(0, len(final_W)):
            truncated_jarzynski = np.mean(np.exp(-np.sort(final_W)[:-x]))
            truncated_jarzynski_next = np.mean(np.exp(-np.sort(final_W)[:-x-1]))
            truncated_part = sorted_final_W[-x-1:]
            remaining_part = sorted_final_W[:-x-1]
            if truncated_jarzynski_next > 1 and truncated_jarzynski < 1:
                break

    naive_jarzynski = np.mean(np.exp(-np.sort(final_W)))

    print("naive_jarzynski = ", naive_jarzynski)
    print("truncated_jarzynski = ", truncated_jarzynski)
    print("truncated_jarzynski_next = ", truncated_jarzynski_next)
    print("truncated_percentage = ", len(truncated_part) / (len(truncated_part) + len(remaining_part)))


    fig, ax = plt.subplots(1, 2, figsize=(18,4))

    selfDefinedBins = np.linspace(np.min(final_W), np.max(final_W), 100)
    ax[0].hist(remaining_part, bins = selfDefinedBins)
    ax[0].hist(truncated_part, bins = selfDefinedBins, color = "red")
    ax[0].title.set_text('no scale')

    ax[1].set_yscale("log")
    ax[1].hist(remaining_part, bins = selfDefinedBins)
    ax[1].hist(truncated_part, bins = selfDefinedBins, color = "red")
    ax[1].title.set_text('log scale')

    plt.show()

    return truncated_jarzynski, truncated_jarzynski_next, truncated_part, remaining_part


# import itertools
# import operator

# def getProtocolSubstepName(protocol_list, t):
#     time_array = [item["duration"] for item in protocol_list]
#     name_array = [item["name"] for item in protocol_list]
#     cumulative_time_array = list(itertools.accumulate(time_array, operator.add))

#     targetIndex = 0

#     for i, x in enumerate(cumulative_time_array):
#         if i == len(cumulative_time_array) - 1:
#             targetIndex = i
#             break
#         elif i == 0 and t < cumulative_time_array[i]:
#             print("case 2")
#             targetIndex = i
#             break
#         else:
#             if t >= cumulative_time_array[i] and t <= cumulative_time_array[i+1]:
#                 targetIndex = i + 1
#                 break

#     print(time_array, cumulative_time_array, name_array[targetIndex])
    
