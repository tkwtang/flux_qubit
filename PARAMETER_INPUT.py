import numpy as np

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
fast = 0
params = {}
params['N'] = 1_000_000
params['dt'] = 1/1_000
params['lambda'] = 1
params['beta'] = 1
params['sim_params'] = [_lambda, _theta, _eta]
params['target_work'] = None
params['comment'] = ""
