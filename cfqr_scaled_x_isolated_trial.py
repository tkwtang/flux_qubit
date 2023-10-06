import sys, os
sys.path.append(os.path.expanduser('~/Project/source'))

import numpy as np
from quick_sim import setup_sim
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from sus.protocol_designer.protocol import sequential_protocol
from edward_tools import coupled_fq_protocol_library, cfq_runner
from edward_tools.coupled_fq_potential import coupled_flux_qubit_pot, coupled_flux_qubit_force, coupled_fq_pot
import edward_tools.cfq_batch_sweep as cfq_batch_sweep
from edward_tools.visualization import animate_sim_flux_qubit
import kyle_tools as kt
import matplotlib.pyplot as plt

create_simple_protocol_parameter_dict = coupled_fq_protocol_library.create_simple_protocol_parameter_dict
create_system_from_storage_and_computation_protocol = coupled_fq_protocol_library.create_system_from_storage_and_computation_protocol
create_system = coupled_fq_protocol_library.create_system



"""
parameter part
"""
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
x_c0 = PHI_0 / (2 * np.pi)
x_c = time_scale_factor * x_c0
m_c = C
U0_1, _, U0_2, _ = m_c * x_c**2 / t_c**2 / k_BT
U0 = quick_doubler(U0_1, U0_2)
keppa = np.array([1/U0_1, 1/U0_1, 1/U0_2, 1/U0_2])

beta_1 = 2 * np.pi * L_1 * I_p_1 / PHI_0
beta_2 = 2 * np.pi * L_2 * I_p_2 / PHI_0

d_beta_1 = 2 * np.pi * L_1 * I_m_1 / PHI_0
d_beta_2 = 2 * np.pi * L_2 * I_m_2 / PHI_0

def get_lambda(m_c, nu_c, t_c, m, nu):
    return nu * nu_c * t_c / (m * m_c)

def get_theta(nu_c, U0, t_c, m_c, nu, keppa, m):
#     return U0 * k_BT * t_c / (m * m_c * x_c**2)
#     return 1/m
    return U0 * k_BT * t_c**2 / (m * m_c * x_c**2)

def get_eta(nu_c, U0, t_c, m_c, nu, keppa, m):
#     return np.sqrt(nu_c * U0 * t_c**3 * nu * keppa / x_c**2) / (m_c* m)
#     return np.sqrt(nu_c * nu * U0 * k_BT * keppa * t_c**3) / (m_c* m * x_c)
    return np.sqrt(_lambda * keppa / m)

_lambda = get_lambda(m_c, nu_c, t_c, m, nu)
_theta = get_theta(nu_c, U0, t_c, m_c, nu, keppa, m)
_eta = get_eta(nu_c, U0, t_c, m_c, nu, keppa, m)

# _theta = _theta / time_scale_factor * 0
# _lambda, _eta = _lambda * 0, _eta * 0

params = {}
params['N'] = 10_000
params['dt'] = 1/10_00 / time_scale_factor**2
params['lambda'] = 1
params['beta'] = 1
params['sim_params'] = [_lambda, _theta, _eta]
params['target_work'] = None
params['time_scale_factor'] = time_scale_factor
print(_lambda, _theta, _eta)

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

# protocol_list = [
#         {"duration": 70, "phi_2_dcx": 3/time_scale_factor, "name": "mix in y direction"}, # mix in y direction
# #         {"duration": 7.5, "phi_2_dcx": 0, "name": "return"}, # return to initial state
# #     {"duration": 2, "name": "mix in y direction (constant)"},
#     {"duration": 80, "M_12": -0.9, "name": "conditional tilt"}, # conditional tilt
#     {"duration": 80, "phi_2_dcx": 0, "name": "conditional tilt"}, # conditional tilt
#     {"duration": 100, "phi_2_dcx": 0, "M_12": 0, "name": "4 well potential (constant)"}, # 4 well potential
# ]




# protocol_list = [
#         {"duration": 100/time_scale_factor , "phi_2_dcx": 0.0001/time_scale_factor, "name": "mix in y direction"}, # mix in y direction
# ]
# protocol_list = [
#         {"duration": 100/time_scale_factor , "phi_2_dcx": 3/time_scale_factor, "name": "mix in y direction"}, # mix in y direction
#         {"duration": 20/time_scale_factor, "name": "fix"},
#         {"duration": 100/time_scale_factor, "M_12": -0.9, "name": "conditional tilt"}, # conditional tilt
#         {"duration": 20/time_scale_factor, "name": "fix"},
#         {"duration": 100/time_scale_factor, "phi_2_dcx": 0, "name": "raise the barrier"}, # conditional tilt
#         {"duration": 20/time_scale_factor, "name": "fix"},
#         {"duration": 100/time_scale_factor, "M_12": 0, "name": "4 well potential (constant)"}, # 4 well potential
#         {"duration": 20/time_scale_factor, "name": "fix"},
#         {"duration": 100/time_scale_factor, "phi_1_dcx": 3/time_scale_factor, "name": "mix in x direction"}, # mix in x direction
#         {"duration": 100/time_scale_factor, "name": "fix"},
#         {"duration": 100/time_scale_factor, "phi_1_dcx": 0, "M_12": 0, "name": "4 well potential "}, # 4 well potential
#         {"duration": 20/time_scale_factor, "name": "4 well potential (constant)"},
# ]

protocol_list = [
    { "duration": 10/time_scale_factor , "phi_2_dcx": 3/time_scale_factor, "name": "mix in y direction"},
    {"duration": 18/time_scale_factor, "M_12": -0.9, "name": "conditional tilt"}, # conditional tilt
#     {"duration": 2/time_scale_factor, "M_12": -0.9, "name": "conditional tilt (fix)"},
    {"duration": 30/time_scale_factor, "phi_2_dcx": 0, "name": "raise the barrier"},
#     {"duration": 2/time_scale_factor, "phi_2_dcx": 0, "name": "fix"},
    {"duration": 20/time_scale_factor, "M_12": 0, "name": "4 well potential (constant)"}, # 4 well potential
    {"duration": 12/time_scale_factor, "phi_1_dcx": 3/time_scale_factor, "name": "mix in x direction"}, # mix in x direction
    {"duration": 50/time_scale_factor, "phi_1_dcx": 0, "M_12": 0, "name": "4 well potential "}, # 4 well potential
]

"""
# step 3: create the relevant storage protocol and computation protocol
"""
computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, \
                                                                    protocol_list)
storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict)

"""
# step 4: create the coupled_fq_runner
"""
cfqr = cfq_runner.coupledFluxQubitRunner(potential = coupled_fq_pot, params = params, \
                                                storage_protocol= storage_protocol, \
                                                computation_protocol= comp_protocol)
cfqr.initialize_sim()
cfqr.set_sim_attributes()
init_state_saved = cfqr.init_state

# cfqr.init_state[:, :, 1] = cfqr.init_state[:, :, 1]/time_scale_factor # to correct the velocity

"""
step 5: Run simulation
"""
#Define a worker — a function which will be executed in parallel
print(params)
manual_domain=[np.array([-5, -5])/time_scale_factor, np.array([5, 5])/time_scale_factor]
params['sim_params'] = [_lambda, _theta, _eta]


"""
step 5a: single simulation
"""
simResult = cfq_batch_sweep.simulateSingleCoupledFluxQubit(params, initial_parameter_dict, protocol_list, \
                                        initial_state = init_state_saved, manual_domain = manual_domain, \
                                        phi_1_dcx = phi_1_dcx,  phi_2_dcx = phi_2_dcx, frameRate = 1)

cfqr = simResult["cfqr"]
cfq_batch_sweep.saveSimulationResult(simResult, U0_1, timeOrStep = 'step', save = True)


"""
step 5: multiprocessing simulation
"""

# def worker(x):
#     simResult = cfq_batch_sweep.simulateSingleCoupledFluxQubit(params, initial_parameter_dict, protocol_list,  initial_state = init_state_saved, manual_domain = manual_domain, phi_1_dcx = phi_1_dcx,  phi_2_dcx = phi_2_dcx)
#     cfqr = simResult["cfqr"]
#     cfq_batch_sweep.saveSimulationResult(simResult, U0_1, timeOrStep = 'step', save = True)
#     return cfqr



# from multiprocessing import Pool

# import time

# work = (["A", 5], ["B", 2])

# def pool_handler():
#     p = Pool(2)
#     p.map(worker, work)


# if __name__ == '__main__':
#     pool_handler()



#
# if __name__ ==  '__main__':
#     set_start_method('fork')
#     lock = Lock()
#     proc_list = []
#     for values in [('immediate', 0), ('delayed', 2), ('eternity', 5)]:
#         p = Process(target=worker, args=values)
#         proc_list.append(p)
#         p.start()  # start execution of printer
#         p.join()

    # [p.join() for p in proc_list]

    # print('After processes...')
#     num_processors = 4
#     p=Pool(processes = num_processors)
#     output = p.map(worker,[i for i in range(0,3)])
#     print(output)
