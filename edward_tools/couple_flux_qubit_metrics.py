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
    for key, destination_list in mapping_state_1_to_state_2_dict.items():
        initial_count = np.sum(state_1_index[key])
        for location in destination_list: # location is the final position that we want the particle to land on
            final_count = np.sum(np.all(np.vstack([state_1_index[key], state_2_index[location]]), axis=0))
            print(f"initial: {key} ({initial_count}), final: {location} ({final_count})")
    return state_1_index
