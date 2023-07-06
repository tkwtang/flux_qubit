""" cell 1 """
import numpy as np
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from sus.protocol_designer.protocol import sequential_protocol
from IPython import display
from IPython.display import HTML
from quick_sim import setup_sim
from edward_tools.coupled_fq_potential import coupled_flux_qubit_pot, coupled_flux_qubit_force, coupled_fq_pot
import edward_tools.fq_runner as fq_runner
from edward_tools.visualization import animate_sim_flux_qubit

import kyle_tools as kt
import matplotlib.pyplot as plt

import importlib, os
from edward_tools import coupled_fq_protocol_library, cfq_runner
from edward_tools import coupled_fq_protocol_library
import edward_tools.cfq_batch_sweep as cfq_batch_sweep

coupled_fq_protocol_library = importlib.reload(coupled_fq_protocol_library)
create_system = coupled_fq_protocol_library.create_system
get_potential_shot_at_different_t = coupled_fq_protocol_library.get_potential_shot_at_different_t
get_potential_shot_at_different_t_1D = coupled_fq_protocol_library.get_potential_shot_at_different_t_1D
create_simple_protocol_parameter_dict = coupled_fq_protocol_library.create_simple_protocol_parameter_dict
create_system_from_storage_and_computation_protocol = coupled_fq_protocol_library.create_system_from_storage_and_computation_protocol
coupled_fq_runner = importlib.reload(cfq_runner)
coupled_fq_protocol_library = importlib.reload(coupled_fq_protocol_library)
create_system = coupled_fq_protocol_library.create_system

""" cell 2 """
import numpy as np
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from sus.protocol_designer.protocol import sequential_protocol
from IPython.display import HTML
from quick_sim import setup_sim
from edward_tools.coupled_fq_potential import coupled_flux_qubit_pot, coupled_flux_qubit_force, coupled_fq_pot
from edward_tools.visualization import animate_sim_flux_qubit
import kyle_tools as kt
import matplotlib.pyplot as plt
import importlib, os, hashlib, json
from edward_tools import coupled_fq_protocol_library, cfq_runner

base_path = "coupled_flux_qubit_protocol/coupled_flux_qubit_initial_state/"

""" cell 3 """
from edward_tools import coupled_fq_protocol_library
coupled_fq_protocol_library = importlib.reload(coupled_fq_protocol_library)
create_system = coupled_fq_protocol_library.create_system
get_potential_shot_at_different_t = coupled_fq_protocol_library.get_potential_shot_at_different_t
create_simple_protocol_parameter_dict = coupled_fq_protocol_library.create_simple_protocol_parameter_dict
create_system_from_storage_and_computation_protocol = coupled_fq_protocol_library.create_system_from_storage_and_computation_protocol
coupled_fq_runner = importlib.reload(cfq_runner)

""" cell 4 """
def saveInitialState(initial_state, initial_parameter_dict, sim_param, overwrite = False):
    base_path = "coupled_flux_qubit_protocol/coupled_flux_qubit_initial_state/"

    new_initial_parameter_dict = initial_parameter_dict.copy()
    new_initial_parameter_dict["N"] = sim_param["N"]
    new_initial_parameter_dict["beta"] = sim_param["beta"]


    key_value_tuple = [x[0] for x in zip(new_initial_parameter_dict.items())]
    string_key_value = [f"{x[0]}__{x[1]}" for x in key_value_tuple]
    final_string = "___".join(string_key_value)
    hash_string = hashlib.sha256(final_string.encode('utf-8')).hexdigest()
    npy_name = hash_string + ".npy"

    if npy_name in os.listdir(base_path) and overwrite is False:
        print("the file exist:" + npy_name)
    else:
        print("write the file:" + npy_name)

        new_initial_parameter_dict["hash_string"] = hash_string

        with open("coupled_flux_qubit_protocol/coupled_flux_qubit_initial_state/data.json") as f:
            jsonData = json.load(f)
            print(jsonData)
            jsonData.append(new_initial_parameter_dict)
            with open("coupled_flux_qubit_protocol/coupled_flux_qubit_initial_state/data.json", "w+") as fw:
                json.dump(jsonData, fw)

        np.save(base_path + npy_name, initial_state)

back_up_initial_state = None

""" cell 7 """
PHI_0 = 2.067833848 * 1e-15
k_B = 1.38e-23
T = 0.5
k_BT = k_B * T
time_scale_factor = 1

prefactor = 1
I_p_1, I_p_2 = 2e-6 * prefactor, 2e-6 * prefactor  # Amp
I_m_1, I_m_2 = 0, 0                                # Amp
R_1, R_2 = 371, 371                                # ohm
C_1, C_2 = 4e-9, 4e-9                              # F
L_1, L_2 = 1e-9, 1e-9                              # H

quick_doubler = lambda x1, x2: np.hstack([np.array([x1] * 2), np.array([x2]*2)])
I_p, I_m = quick_doubler(I_p_1, I_p_2), quick_doubler(I_m_1, I_m_2)
R, L, C = quick_doubler(R_1, R_2), quick_doubler(L_1, L_2), quick_doubler(C_1, C_2)
m = np.array([1, 1/4, 1, 1/4])
nu = np.array([2, 1/2, 2, 1/2])

nu_c = 1/R
t_c = time_scale_factor * np.sqrt(L * C)
x_c = time_scale_factor * PHI_0 / (2 * np.pi)
m_c = C
U0_1, _, U0_2, _ = m_c * x_c**2 / t_c**2 / k_BT
U0 = quick_doubler(U0_1, U0_2)
keppa = np.array([1/U0_1, 1/U0_1, 1/U0_2, 1/U0_2])

beta_1 = 2 * np.pi * L_1 * I_p_1 / PHI_0
beta_2 = 2 * np.pi * L_2 * I_p_2 / PHI_0

d_beta_1 = 2 * np.pi * L_1 * I_m_1 / PHI_0
d_beta_2 = 2 * np.pi * L_2 * I_m_2 / PHI_0

def get_lambda(m_c, nu_c, t_c, m, nu):
    return nu_c * t_c * nu / (m * m_c)

def get_theta(nu_c, U0, t_c, m_c, nu, keppa, m):
    return U0 * k_BT * t_c**2 / (m * m_c * x_c**2)
#     return 1/m

def get_eta(nu_c, U0, t_c, m_c, nu, keppa, m):
#     return np.sqrt(nu_c * U0 * t_c**3 * nu * keppa / x_c**2) / (m_c* m)
    return np.sqrt(nu_c * nu * U0 * k_BT * keppa * t_c**3) / (m_c* m * x_c)
#     return np.sqrt(_lambda * keppa / m)

_lambda = get_lambda(m_c, nu_c, t_c, m, nu)
_theta = get_theta(nu_c, U0, t_c, m_c, nu, keppa, m)
_eta = get_eta(nu_c, U0, t_c, m_c, nu, keppa, m)

params = {}
params['N'] = 10_000
params['dt'] = 1/10_000
params['lambda'] = 1
params['beta'] = 1
params['sim_params'] = [_lambda, _theta, _eta]
params['target_work'] = None
print(_lambda, _theta, _eta)

""" cell 8 """
manual_domain=[[-5, -5], [5, 5]]
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
        "M_12": 0
}

# initial_parameter_dict = {
#         "U0_1": U0_1,     "U0_2": U0_2,     "gamma_1": gamma,  "gamma_2": gamma,
#         "beta_1": beta_1,   "beta_2": beta_2,   "d_beta_1": d_beta_1 ,   "d_beta_2": d_beta_2,
#         "phi_1_x": 0,  "phi_2_x": 0,  "phi_1_dcx": phi_1_dcx,  "phi_2_dcx": 3,
#         "M_12": -0.9
# }
""" cell 9 """
params['sim_params'] = [_lambda, _theta, _eta]

# protocol_list = [
#         {"duration": 7, "phi_2_dcx": 3, "name": "mix in y direction"}, # mix in y direction
# #         {"duration": 7.5, "phi_2_dcx": 0, "name": "return"}, # return to initial state
# #     {"duration": 2, "name": "mix in y direction (constant)"},
#     {"duration": 8, "M_12": -0.9, "name": "conditional tilt"}, # conditional tilt
#     {"duration": 8, "phi_2_dcx": 0, "name": "conditional tilt"}, # conditional tilt
#     {"duration": 10, "phi_2_dcx": 0, "M_12": 0, "name": "4 well potential (constant)"}, # 4 well potential
# ]

protocol_list = [
        {"duration": 100, "phi_2_dcx": 3, "name": "mix in y direction"}, # mix in y direction
        {"duration": 20, "name": "fix"},
        {"duration": 100, "M_12": -0.9, "name": "conditional tilt"}, # conditional tilt
        {"duration": 20, "name": "fix"},
        {"duration": 100, "phi_2_dcx": 0, "name": "raise the barrier"}, # conditional tilt
        {"duration": 20, "name": "fix"},
        {"duration": 100, "M_12": 0, "name": "4 well potential (constant)"}, # 4 well potential
        {"duration": 20, "name": "fix"},
        {"duration": 100, "phi_1_dcx": 3, "name": "mix in x direction"}, # mix in x direction
        {"duration": 100, "name": "fix"},
        {"duration": 100, "phi_1_dcx": 0, "M_12": 0, "name": "4 well potential "}, # 4 well potential
        {"duration": 20, "name": "4 well potential (constant)"},
]

computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, \
                                                                    protocol_list)
storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict)
cfqr = coupled_fq_runner.coupledFluxQubitRunner(params = params, storage_protocol= storage_protocol, \
                                                computation_protocol= comp_protocol)
cfqr.initialize_sim()
cfqr.set_sim_attributes()
init_state_saved = cfqr.init_state

""" cell 9 """
vmin,vmax  = 0, 50
vmin, vmax = None, None
manual_domain=[[-5, -5], [5, 5]]

""" cell 10 """
loadInitialState = True
loadInitialState = False
base_path = "coupled_flux_qubit_protocol/coupled_flux_qubit_initial_state/"
fileHash = "7757abc78a0bfba3e170d6b4b72d30d3671706289e7b4a183f7d51307a18edc1"
initial_state_location = base_path + fileHash + ".npy"
initial_state = None if not loadInitialState else np.load(initial_state_location)

save_initial_state = False
# save_initial_state = True
overwrite = False
if save_initial_state:
    back_up_initial_state = cfqr.init_state.copy()
    saveInitialState(back_up_initial_state, initial_parameter_dict, params, overwrite)


""" cell 10 """
manual_domain=[[-5, -5], [5, 5]]
mapping_state_1_to_state_2_dict = {"00": ["00", "10"], "01": ["00", "10"], "10": ["01", "11"], "11": ["01", "11"]}

""" cell 10 """
params['sim_params'] = [_lambda, _theta, _eta]
# print(params['sim_params'])
simResult = cfq_batch_sweep.simulateSingleCoupledFluxQubit(params, initial_parameter_dict, protocol_list, \
                                        initial_state = init_state_saved, manual_domain = manual_domain, \
                                        phi_1_dcx = phi_1_dcx,  phi_2_dcx = phi_2_dcx)



""" cell 10 """
cfqr = simResult["cfqr"]
cfq_batch_sweep = importlib.reload(cfq_batch_sweep)
cfq_batch_sweep.saveSimulationResult(simResult, timeOrStep = 'step', save = True)

""" cell 10 """


""" cell 10 """


""" cell 10 """

""" cell 10 """
