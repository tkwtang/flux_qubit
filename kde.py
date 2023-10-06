#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

from edward_tools import coupled_fq_protocol_library
coupled_fq_protocol_library = importlib.reload(coupled_fq_protocol_library)
create_system = coupled_fq_protocol_library.create_system
get_potential_shot_at_different_t = coupled_fq_protocol_library.get_potential_shot_at_different_t
create_simple_protocol_parameter_dict = coupled_fq_protocol_library.create_simple_protocol_parameter_dict
create_system_from_storage_and_computation_protocol = coupled_fq_protocol_library.create_system_from_storage_and_computation_protocol
coupled_fq_runner = importlib.reload(cfq_runner)

back_up_initial_state = None

PHI_0 = 2.067833848 * 1e-15
k_B = 1.38e-23
T = 0.5
k_BT = k_B * T
time_scale_factor = 100

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
params['N'] = 10_00
params['dt'] = 1/1_000
params['lambda'] = 1
params['beta'] = 1
params['sim_params'] = [_lambda, _theta, _eta]
params['target_work'] = None
print(_lambda, _theta, _eta)

# step 1: import potential
coupled_fq_default_param = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x_c0]
[phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound] = np.array([4, 4, 4, 4])/time_scale_factor

coupled_fq_domain = [[-phi_1_bound, -phi_2_bound, -phi_1dc_bound, -phi_2dc_bound], \
                     [phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound]]

coupled_fq_pot = Potential(coupled_flux_qubit_pot, coupled_flux_qubit_force, 14, 4,\
                           default_params = coupled_fq_default_param,  relevant_domain = coupled_fq_domain)


# step 2: define initial condition and protocol

manual_domain=[np.array([-5, -5])/time_scale_factor, np.array([5, 5])/time_scale_factor]
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

protocol_list = [
        {"duration": 1, "phi_2_dcx": 3/time_scale_factor, "name": "mix in y direction"}, # mix in y direction
#         {"duration": 7.5, "phi_2_dcx": 0, "name": "return"}, # return to initial state
#     {"duration": 2, "name": "mix in y direction (constant)"},
    {"duration": 1, "M_12": -0.9, "name": "conditional tilt"}, # conditional tilt
    {"duration": 1, "phi_2_dcx": 0, "name": "conditional tilt"}, # conditional tilt
    {"duration": 1, "phi_2_dcx": 0, "M_12": 0, "name": "4 well potential (constant)"}, # 4 well potential
]


# step 3: create the relevant storage protocol and computation protocol
computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, \
                                                                    protocol_list)
storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict)

manual_domain=[np.array([-5, -5])/time_scale_factor, np.array([5, 5])/time_scale_factor]
mapping_state_1_to_state_2_dict = {"00": ["00", "10"], "01": ["00", "10"], "10": ["01", "11"], "11": ["01", "11"]}



# step 4: create the coupled_fq_runner
cfqr = coupled_fq_runner.coupledFluxQubitRunner(potential = coupled_fq_pot, params = params, \
                                                storage_protocol= storage_protocol, \
                                                computation_protocol= comp_protocol)
cfqr.initialize_sim()
# cfqr.set_sim_attributes()
# init_state_saved = cfqr.init_state


# In[4]:





# In[5]:





# In[30]:


from sklearn.neighbors import KernelDensity
import numpy as np
kernel = "tophat"
kernel = "gaussian"
resolution = 50

# tried to resolution = 10, 20, 50 and 100
x_range = np.linspace(-4, 4, resolution)
X, Y, Z, A = np.meshgrid(x_range, x_range, x_range, x_range)
meshgrid = np.transpose([X.ravel(), Y.ravel(), Z.ravel(), A.ravel()])


params = cfqr.protocol.get_params(0)
weight = np.exp(-cfqr.potential.potential(*meshgrid.T, params=params))

bandwidthArray = [0.2, 0.5, 1.0, 2.0]
kdeArray = [KernelDensity(kernel=kernel, bandwidth=_bandwidth).fit(meshgrid, weight) for \
           _bandwidth in bandwidthArray]

sampleArray = [_kde.sample(10000) for _kde in kdeArray]
# kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(meshgrid, weight)



def plotEstimation(kernel, resolution, samples, bandwidthArray):
    fig, ax = plt.subplots(4, 4, figsize=(18,11))
    fig.suptitle(f"{kernel}, resolution={resolution}", fontsize=16)
    plt.subplots_adjust(bottom=0.2)


    for i, _sample in enumerate(samples):
        ax[i][0].hist(_sample[:, 0], 100)
        ax[i][0].title.set_text(f'phi_1 ({bandwidthArray[i]})')

        ax[i][1].hist(_sample[:, 1], 100)
        ax[i][1].title.set_text(f'phi_1_dc ({bandwidthArray[i]})')

        ax[i][2].hist(_sample[:, 2], 100)
        ax[i][2].title.set_text(f'phi_2 ({bandwidthArray[i]})')

        ax[i][3].hist(_sample[:, 3], 100)
        ax[i][3].title.set_text(f'phi_2_dc ({bandwidthArray[i]})')
    fig.tight_layout()
    plt.savefig(f"{kernel}, resolution={resolution}.png")
    # plt.show()


# In[ ]:


plotEstimation(kernel, resolution, sampleArray, bandwidthArray)


# In[ ]:
