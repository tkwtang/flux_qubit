import numpy as np
import sys
import os
source_path = os.path.expanduser('~/Project/source/')
sys.path.append(source_path)
sys.path.append(os.path.expanduser('~/Project/source/simtools/'))

from .fq_potential import fq_pot, fq_default_param
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol


def create_simple_protocol_parameter_dict(protocol_array):
    protocol_key = ["U_0", "gamma", "beta", "delta_beta", "phi_x", "phi_dcx"]
    result_dict = {}
    for i, k in enumerate(protocol_key):
        result_dict[k] = [protocol_array[i], protocol_array[i]]
    result_dict["t"] = [0, 1]
    return result_dict

def convert_potential_parameter_to_protocol_parameter(potential_dict):
    protocol_key = ["U_0", "gamma", "beta", "delta_beta", "phi_x", "phi_dcx"]
    result_dict = {}
    for k in protocol_key:
        result_dict[k] = [potential_dict[k], potential_dict[k]]
    result_dict["t"] = [0, 1]
    return result_dict

def create_system(storage_protocol_parameter_dict, comp_protocol_parameter_dict, domain = None):
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

    # to create the parameter list at different steps
    protocol_key = ["U_0", "gamma", "beta", "delta_beta", "phi_x", "phi_dcx"]

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
