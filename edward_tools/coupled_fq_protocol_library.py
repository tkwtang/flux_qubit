import numpy as np
import sys
import os
source_path = os.path.expanduser('~/Project/source/')
sys.path.append(source_path)
sys.path.append(os.path.expanduser('~/Project/source/simtools/'))

from .fq_potential import fq_pot, fq_default_param
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
import matplotlib.pyplot as plt

protocol_key = ['U0_1', 'U0_2', 'gamma_1', 'gamma_2', 'beta_1', 'beta_2', 'd_beta_1', 'd_beta_2', 'phi_1_x', 'phi_2_x', 'phi_1_dcx', 'phi_2_dcx', 'M_12']

def create_simple_protocol_parameter_dict(protocol_array):
    """
    simple means that the protocol doesn't have substeps.
    """
    result_dict = {}
    for i, k in enumerate(protocol_key):
        result_dict[k] = [protocol_array[i], protocol_array[i]]
    result_dict["t"] = [0, 1]
    return result_dict

def create_system(protocol_parameter_dict, domain = None):
    """
    This function is used to produce the storage and computation protocol

    input:
    1. comp_protocol_parameter_dict:
    - a dictionary contains the an array of time, which represents the time point at which the protocol is changed
    - the key is the name of the parameter
    - for parameters, they are arrays containing the value of the parameter at the particular time point

    output:
    1. comp_prototocl
    - the protocol for the computation system
    """
    # storage protocol, just take the first and last element of each term in the protocol key from the dict to form the storage protocol
    storage_t = (protocol_parameter_dict["t"][0], protocol_parameter_dict["t"][-1])
    storage_protocol_parameter_time_series = [np.array([protocol_parameter_dict[key][0], protocol_parameter_dict[key][-1]]) for key in protocol_key]
    storage_protocol_parameter_time_series = np.array(storage_protocol_parameter_time_series)
    storage_protocol = Protocol(storage_t, storage_protocol_parameter_time_series)

    # computation protocol, this part form the time series of the comp_protocol and join them to form Compound_Protocol
    comp_protocol_array = []
    comp_t = protocol_parameter_dict["t"]
    comp_protocol_parameter_time_series = [protocol_parameter_dict[key] for key in protocol_key]
    comp_protocol_parameter_time_series = np.array(comp_protocol_parameter_time_series).T

    for i in range(len(comp_t)-1):
        n_th_comp_time_array = (comp_t[i], comp_t[i+1])
        n_th_comp_protocol_parameter_array = np.array([comp_protocol_parameter_time_series[i], comp_protocol_parameter_time_series[i+1]]).T # in the form of array of [(p_n_i, p_n_f)]
        _p = Protocol(n_th_comp_time_array, n_th_comp_protocol_parameter_array)
        comp_protocol_array.append(_p)
    comp_protocol = Compound_Protocol(comp_protocol_array)

    return storage_protocol, comp_protocol


def create_system_from_storage_and_computation_protocol(storage_protocol_parameter_dict = None, comp_protocol_parameter_dict = None, domain = None):
    """
    This function is used to produce the storage and computation protocol

    input:
    1. input_parameters_dict:
    - a dictionary contains the an array of time, which represents the time point at which the protocol is changed
    - the key is the name of the parameter
    - for parameters, they are arrays containing the value of the parameter at the particular time point

    output:
    1. storage_protocol:
    - the protocol for the equilibrium system

    2. comp_prototocl
    - the protocol for the computation system
    """
    if comp_protocol_parameter_dict == None:
        print("please give me comp_protocol")


    if storage_protocol_parameter_dict is not None:
        print("storage_protocol_parameter_dict is not None")

    # storage protocol
    storage_t = storage_protocol_parameter_dict["t"]
    storage_protocol_parameter_time_series = [storage_protocol_parameter_dict[key] for key in protocol_key]
    storage_protocol_parameter_time_series = np.array(storage_protocol_parameter_time_series)
    storage_protocol = Protocol(storage_t, storage_protocol_parameter_time_series)

    # computation protocol
    comp_protocol_array = []
    comp_t = comp_protocol_parameter_dict["t"]
    comp_protocol_parameter_time_series = [comp_protocol_parameter_dict[key] for key in protocol_key]
    comp_protocol_parameter_time_series = np.array(comp_protocol_parameter_time_series).T

    for i in range(len(comp_t)-1):
        n_th_comp_time_array = (comp_t[i], comp_t[i+1])
        n_th_comp_protocol_parameter_array = np.array([comp_protocol_parameter_time_series[i], comp_protocol_parameter_time_series[i+1]]).T # in the form of array of [(p_n_i, p_n_f)]
        _p = Protocol(n_th_comp_time_array, n_th_comp_protocol_parameter_array)
        comp_protocol_array.append(_p)
    comp_protocol = Compound_Protocol(comp_protocol_array)

    return storage_protocol, comp_protocol


def get_potential_shot_at_different_t(simRunner, timeSeries, axis1 = 0, axis2 = 1, contours=10, resolution = 200, manual_domain=None, surface = False, cbar=False):

    numberOfColumns = 4

    numberOfPlots = len(timeSeries)
    offset = 0 if numberOfPlots % numberOfColumns == 0 else 1
    numberOfRows = numberOfPlots // numberOfColumns + offset

    if surface is False:
        fig, ax = plt.subplots(numberOfRows, numberOfColumns, figsize=(18,4 * numberOfRows))
        fig.tight_layout(h_pad=5, w_pad=2)
    else:
        fig = plt.figure(figsize=plt.figaspect(0.5))

    for i, t in enumerate(timeSeries):
        row = i // numberOfColumns
        column = i % numberOfColumns
        U, X_mesh = simRunner.system.lattice(t, resolution, axes=(axis1, axis2), manual_domain=manual_domain)
        X = X_mesh[0]
        Y = X_mesh[1]
        x_min, x_max = np.min(X), np.max(X)
        y_min, y_max = np.min(Y), np.max(Y)
        if surface is False:
            out = ax[row][column].contourf(X, Y, U, contours)
            if cbar:
                plt.colorbar(out)
            ax[row][column].set_title("t={:.2f}".format(t))

        if surface is True:

            ax = fig.add_subplot(row+1, numberOfColumns, column + 1, projection='3d')
            surf = ax.plot_surface(X, Y, U)
            ax.set_title("t={:.2f}".format(t))
    plt.show()
