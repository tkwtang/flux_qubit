import sys, os
sys.path.append(os.path.expanduser('~/Project/source'))
#
import numpy as np
import copy
from quick_sim import setup_sim
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from sus.protocol_designer.protocol import sequential_protocol
from edward_tools import coupled_fq_protocol_library, cfq_runner
from edward_tools.coupled_fq_potential import coupled_flux_qubit_pot, coupled_flux_qubit_force, coupled_fq_pot
import edward_tools.cfq_batch_sweep as cfq_batch_sweep
from edward_tools.visualization import animate_sim_flux_qubit
import kyle_tools as kt
import matplotlib.pyplot as plt
from PARAMETER_INPUT import *
from PARAMETER_INPUT import _lambda, _theta, _eta


create_simple_protocol_parameter_dict = coupled_fq_protocol_library.create_simple_protocol_parameter_dict
create_system_from_storage_and_computation_protocol = coupled_fq_protocol_library.create_system_from_storage_and_computation_protocol
create_system = coupled_fq_protocol_library.create_system

# print(PHI_0, _lambda, _theta, _eta,)
"""
# step 1: Define potential
"""
coupled_fq_default_param = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x_c0]
[phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound] = np.array([4, 4, 4, 4])/time_scale_factor

coupled_fq_domain = [[-phi_1_bound, -phi_2_bound, -phi_1dc_bound, -phi_2dc_bound], \
                     [phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound]]

coupled_fq_pot = Potential(coupled_flux_qubit_pot, coupled_flux_qubit_force, 14, 4,\
                           default_params = coupled_fq_default_param,  relevant_domain = coupled_fq_domain)

"""
# step 2: Define initial condition and protocol
"""
manual_domain=[np.array([-5, -5]), np.array([5, 5])]
# phi_1_dcx, phi_2_dcx = 3, 3
phi_1_dcx, phi_2_dcx = 0, 0
phi_1_dc, phi_2_dc = phi_1_dcx, phi_2_dcx

# gamma, beta_1, beta_2 = 0, 0, 0
gamma = 20
# d_beta_1, d_beta_2 = 0.6, 0.6
d_beta_1, d_beta_2 = 0, 0
params['sim_params'] = [_lambda, _theta, _eta]

initial_parameter_dict = {
        "U0_1": U0_1,     "U0_2": U0_2,     "gamma_1": gamma,  "gamma_2": gamma,
        "beta_1": beta_1,   "beta_2": beta_2,   "d_beta_1": d_beta_1 ,   "d_beta_2": d_beta_2,
        "phi_1_x": 0,  "phi_2_x": 0,  "phi_1_dcx": phi_1_dcx,  "phi_2_dcx": phi_2_dcx,
        "M_12": 0, 'x_c': x_c
}

load_initial_state = True
old_initial_state_name = "6ba93082ae3f794e8e44ec11a9d40878e864e525dd9732ac9fda2337f3c7f8c1" + "_final_state.npy"


sweep_parameter = "duration"
# sweep_parameter_list = np.linspace(2, 10, 6) /time_scale_factor
sweep_parameter_list = np.linspace(100, 200, 4) /time_scale_factor
# sweep_parameter_list = np.linspace(1, 2, 3) /time_scale_factor

initial_parameter_dict["phi_2_dcx"] = 0/time_scale_factor
initial_parameter_dict["M_12"] = 0
# initial_parameter_dict["phi_1_dcx"] = 3/time_scale_factor
sweep_parameter_prototype = [
    # {
    #     "duration": 10/time_scale_factor , "phi_2_dcx": 3/time_scale_factor, "name": "mix in y direction"
    # },
    # {"duration": 18/time_scale_factor, "M_12": -0.9, "name": "conditional tilt"}, # conditional tilt
    # {"duration": 2/time_scale_factor, "M_12": -0.9, "name": "conditional tilt (fix)"},
    # {"duration": 16/time_scale_factor, "phi_2_dcx": 0, "name": "raise the barrier"},
    # {"duration": 2/time_scale_factor, "phi_2_dcx": 0, "name": "fix"},
    # {"duration": 20/time_scale_factor, "M_12": 0, "name": "4 well potential (constant)"}, # 4 well potential
    {"duration": 12/time_scale_factor, "phi_1_dcx": 3/time_scale_factor, "name": "mix in x direction"}, # mix in x direction
    # {"duration": 50/time_scale_factor, "phi_1_dcx": 0, "M_12": 0, "name": "4 well potential "}, # 4 well potential

]



protocol_list_for_sweep = []
for val in sweep_parameter_list:
    _dummy_dict = copy.deepcopy(sweep_parameter_prototype)
    _last_item = copy.deepcopy(_dummy_dict[-1])
    _last_item[sweep_parameter] = val
    _dummy_dict[-1] = _last_item

    protocol_list_for_sweep.append(_dummy_dict)

"""
# step 3: create the relevant storage protocol and computation protocol
"""

# check if the simulation is a sweep or not
def multiprocessing_worker_function(protocol_list):
    # print(protocol_list)

    computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, protocol_list)
    storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict)

    """
    # step 4: create the coupled_fq_runner
    """
    cfqr = cfq_runner.coupledFluxQubitRunner(potential = coupled_fq_pot, params = params, storage_protocol= storage_protocol, computation_protocol= comp_protocol)
    cfqr.initialize_sim()
    cfqr.set_sim_attributes()
    init_state_saved = cfqr.init_state

    if load_initial_state:
        init_state_saved = np.load(f"coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{old_initial_state_name}")

    """
    step 5: Run sweep simulation in parallel
    """

    manual_domain=[np.array([-5, -5])/time_scale_factor, np.array([5, 5])/time_scale_factor]
    params['sim_params'] = [_lambda, _theta, _eta]

    """
    step 5a: single simulation
    """
    simResult = cfq_batch_sweep.simulateSingleCoupledFluxQubit(params, initial_parameter_dict, protocol_list, initial_state = init_state_saved, manual_domain = manual_domain, phi_1_dcx = phi_1_dcx,  phi_2_dcx = phi_2_dcx, frameRate = 10)

    cfqr = simResult["cfqr"]

    cfq_batch_sweep.saveSimulationResult(simResult, U0_1, timeOrStep = 'step', save = True, save_final_state = True)

    return simResult

from multiprocessing import Process
if __name__ == "__main__":  # confirms that the code is under main function
    procs = []
    # instantiating process with arguments
    for protocol_list in protocol_list_for_sweep:
        proc = Process(target=multiprocessing_worker_function, args=[protocol_list])
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
    print("process finished")
