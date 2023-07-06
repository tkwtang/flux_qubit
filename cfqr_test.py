import numpy as np
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from sus.protocol_designer.protocol import sequential_protocol
from IPython.display import HTML
from quick_sim import setup_sim
from edward_tools.coupled_fq_potential import coupled_flux_qubit_pot, coupled_flux_qubit_force, coupled_fq_pot
import edward_tools.fq_runner as fq_runner
from edward_tools.visualization import animate_sim_flux_qubit
import kyle_tools as kt
import matplotlib.pyplot as plt
import importlib
from edward_tools import coupled_fq_protocol_library, cfq_runner



I_p_1 = 2e-6       # Amp
I_p_2 = 2e-6       # Amp
I_m_1 = 7e-9       # Amp
I_m_2 = 7e-9       # Amp
R_1 = 371          # ohm
R_2 = 371          # ohm
C_1 = 4e-9         # F
C_2 = 4e-9         # F
L_1 = 1e-9         # H
L_2 = 1e-9         # H

M_12 = L_1 * L_2
PHI_0 = 2.067833848 * 1e-15


quick_doubler = lambda x1, x2: np.hstack([np.array([x1] * 2), np.array([x2]*2)])

I_p = quick_doubler(I_p_1, I_p_2)
I_m = quick_doubler(I_m_1, I_m_2)
R = quick_doubler(R_1, R_2)
L = quick_doubler(L_1, L_2)
C = quick_doubler(C_1, C_2)

m = np.array([1, 1/4, 1, 1/4])
nu = np.array([2, 1/2, 2, 1/2])
kepa = np.array([1, 1, 1, 1])


beta_1 = 2 * np.pi * L_1 * I_p_1 / PHI_0
beta_2 = 2 * np.pi * L_2 * I_p_2 / PHI_0

d_beta_1 = 2 * np.pi * L_1 * I_m_1 / PHI_0
d_beta_2 = 2 * np.pi * L_2 * I_m_2 / PHI_0



_lambda = np.sqrt(L*C)/(R*C)  * nu
_theta  = 1 / m
_eta = np.sqrt(_lambda * kepa / m)

params = {}
params['N'] = 100
params['dt'] = 1/100
params['lambda'] = 1
params['beta'] = 1
params['sim_params'] = [_lambda, _theta, _eta]
params['tau'] = 4
params['target_work'] = None

storage_parameter_dict = {
    "t": [0, 1],
    "U0_1": [1, 1],
    "U0_2": [1, 1],
    "gamma_1": [1, 0],
    "gamma_2": [1, 0],
    "beta_1": [0, 100],
    "beta_2": [0, 100],
    "d_beta_1": [0, 0],
    "d_beta_2": [0, 0],
    "phi_x_1": [0, 0],
    "phi_x_2": [0, 0],
    "phi_xdc_1": [0, 0],
    "phi_xdc_2": [0, 0]
}

computation_parameter_dict = {
    "t": [0, 0.5, 0.75, 1],
    "U0_1": [1, 1, 1, 1],
    "U0_2": [1, 1, 1, 1],
    "gamma_1": [0, 0, 10, 0],
    "gamma_2": [0, 0, 10, 0],
    "beta_1": [0, 0, 0, 0],
    "beta_2": [0, 0, 0, 0],
    "d_beta_1": [0, 0, 0, 0],
    "d_beta_2": [0, 0, 0, 0],
    "phi_x_1": [0, 9, 9, 0],
    "phi_x_2": [0, 9, 9, 0],
    "phi_xdc_1": [0, 2, 2, 0],
    "phi_xdc_2": [0, 2, 2, 0]
}



def parameter_generator(param_dict):
    """
    To generate a list of parameter
    """

    return [param_dict["U0_1"], param_dict["U0_2"], param_dict["gamma_1"], \
            param_dict["gamma_2"], param_dict["beta_1"], param_dict["beta_2"], param_dict["d_beta_1"], \
            param_dict["d_beta_2"], param_dict["phi_x_1"], param_dict["phi_x_2"], param_dict["phi_xdc_1"],\
            param_dict["phi_xdc_2"]
           ]


cfq_protocol_library = importlib.reload(coupled_fq_protocol_library)
create_simple_protocol_parameter_dict = cfq_protocol_library.create_simple_protocol_parameter_dict
create_system = cfq_protocol_library.create_system

coupled_fq_runner = importlib.reload(cfq_runner)

trial_Vstore_param = parameter_generator({
        "U0_1": 1,     "U0_2": 1,
        "gamma_1": 1,  "gamma_2": 1,
        "beta_1": beta_1,   "beta_2": beta_2,   "d_beta_1": d_beta_1,   "d_beta_2": d_beta_2,
        "phi_x_1": 0,  "phi_x_2": 0,  "phi_xdc_1": 0,  "phi_xdc_2": 0
    })

trial_Vcomp_param = parameter_generator({
        "U0_1": 1,     "U0_2": 1,
        "gamma_1": 0,  "gamma_2": 0,
        "beta_1": 0,   "beta_2": 0,   "d_beta_1": 0,   "d_beta_2": 0,
        "phi_x_1": 0,  "phi_x_2": 0,  "phi_xdc_1": 0,  "phi_xdc_2": 0
    })



realistic_param = parameter_generator({
        "U0_1": 1,     "U0_2": 1,
        "gamma_1": 12,  "gamma_2": 12,
        "beta_1": 6.2,   "beta_2": 6.2,   "d_beta_1": 0.2,   "d_beta_2": 0.2,
        "phi_x_1": 0.084,  "phi_x_2": 0.084,  "phi_xdc_1": -2.5,  "phi_xdc_2": -2.5
    })

t = [0, 0.5]



# to create the relevant protocols
storage_parameter_dict = create_simple_protocol_parameter_dict(trial_Vstore_param)
computation_parameter_dict = create_simple_protocol_parameter_dict(trial_Vcomp_param)


storage_protocol, comp_protocol = create_system(storage_parameter_dict, computation_parameter_dict)

# create the eq_system and computation_system
eq_system = System(storage_protocol, coupled_fq_pot)
computation_system = System(comp_protocol, coupled_fq_pot)


cfqr = coupled_fq_runner.coupledFluxQubitRunner(params = params, storage_protocol= storage_protocol, \
                                computation_protocol= comp_protocol)


cfqr.run_sim()

print("finish simulations.")
