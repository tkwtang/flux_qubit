import numpy as np
import os
from kyle_tools import helpingTools
import sys

source_path = os.path.expanduser('~/Project/source/')
sys.path.append(source_path)

def get_statistics_for_all_categories(simulation, time, trial_request=np.s_[:]):
    """
    This function returns the mean and var for x and y coordinates at different steps

    return: an array of (x_mean, x_var, y_mean, y_var) for each categories (e.g. in 2d, it is 00, 01, 10, 11)
    """
    # state = simulation.current_state[trial_request]
    time = simulation.current_time
    state = simulation.current_state[trial_request]
    trials_number = trial_request # if trial_request exists, use it instead of the whole trials number

    # it is an array of [x_mean, x_var, y_mean, y_var]
    result_array = np.empty([4, 4])

    if time == 0:
        # to get the index for the categories of the initial positions
        simulation.position_categories_index_dict = helpingTools.position_classification(state)
        # print(simulation.position_categories_index_dict)

    index = 0
    for label, val in simulation.position_categories_index_dict.items():
        # label = 00, 01, 10, 11
        # plt.scatter(*state[val].T)
        x_coordinates = state[val][:, 0]
        y_coordinates = state[val][:, 1]

        statistics_result = np.array([np.mean(x_coordinates), np.var(x_coordinates), np.mean(y_coordinates), np.var(y_coordinates)])
        result_array[index:] = statistics_result
        index += 1

    return result_array
