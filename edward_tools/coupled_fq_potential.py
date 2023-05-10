import numpy as np
import os, sys

source_path = os.path.expanduser('~/Project/source/')
sys.path.append(source_path)
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol

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

M_12 = L_1 * L_2 * 0
PHI_0 = 2.067833848 * 1e-15


beta_1 = 2 * np.pi * L_1 * I_p_1 / PHI_0
beta_2 = 2 * np.pi * L_2 * I_p_2 / PHI_0

d_beta_1 = 2 * np.pi * L_1 * I_m_1 / PHI_0
d_beta_2 = 2 * np.pi * L_2 * I_m_2 / PHI_0


quick_doubler = lambda x1, x2: np.hstack([np.array([x1] * 2), np.array([x2]*2)])

I_p = quick_doubler(I_p_1, I_p_2)
I_m = quick_doubler(I_m_1, I_m_2)
R = quick_doubler(R_1, R_2)
L = quick_doubler(L_1, L_2)
C = quick_doubler(C_1, C_2)

m = np.array([1, 1/4, 1, 1/4])
nu = np.array([2, 1/2, 2, 1/2])
kepa = np.array([1, 1, 1, 1])


# phi_1_prefactor = 1
# phi_2_prefactor = 1

def coupled_flux_qubit_pot(phi_1, phi_2, phi_1dc, phi_2dc, params, phi_1_prefactor = 1, phi_2_prefactor = 1):
    """
    4D 4-well potential.

    Parmeters
    -------------
    phi: ndaray of dimension [N, 2]
    phi_dc: ndaray of dimension [N, 2]

    params: list / tuple
    - [U_0, g, beta, delta_beta, phi_x, phi_xdc ]: correspond to the energy scale, gamma
    - phi_x: associated with asymmetry in the informational subspace, and will only take a nonzero value to help
      offset asymmetry from the delta_beta term in U'
    """
    # (U0_1, U0_2), (g_1, g_2),  (beta_1, beta_2), (delta_beta_1, delta_beta_2), (phi_1x, phi_2x), (phi_1dcx, phi_2dcx) = params


    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12 = params

    u1_1 = 1/2 * (phi_1 - phi_1x)**2 * phi_1_prefactor
    u1_2 = 1/2 * (phi_2 - phi_2x)**2 * phi_2_prefactor
    u2_1 = 1/2 * g_1 * (phi_1dc - phi_1dcx)**2
    u2_2 = 1/2 * g_2 * (phi_2dc - phi_2dcx)**2
    u3_1 = beta_1 * np.cos(phi_1) * np.cos(phi_1dc/2)
    u3_2 = beta_2 * np.cos(phi_2) * np.cos(phi_2dc/2)
    u4_1 = delta_beta_1 * np.sin(phi_1) * np.sin(phi_1dc/2)
    u4_2 = delta_beta_2 * np.sin(phi_2) * np.sin(phi_2dc/2)
    u5 = M_12 * (phi_1 - phi_1x) * (phi_2 - phi_2x)

    U = U0_1 * ( u1_1 + u2_1 + u3_1 + u4_1 ) + U0_2 * ( u1_2 + u2_2 + u3_2 + u4_2 ) + u5

    # print("From coupled_fq_potential.py")
    # print("This is phi_1:", phi_1)
    # print("This is phi_2:", phi_2)
    # print("This is phi_1dc:", phi_1dc)
    # print("This is phi_2dc:", phi_2dc)

    return U


def coupled_flux_qubit_force(phi_1, phi_2, phi_1dc, phi_2dc, params, phi_1_prefactor = 1, phi_2_prefactor = 1):
    """
    2D 4-well potential.

    Parmeters
    -------------
    phi: ndaray of dimension [N, ]
    phi_dc: ndaray of dimension [N, ]

    params: list / tuple
    - [U_0, g, beta, delta_beta, phi_x, phi_xdc ]: correspond to the energy scale, gamma
    """

    U0_1, U0_2, g_1, g_2,  beta_1, beta_2, delta_beta_1, delta_beta_2, phi_1x, phi_2x , phi_1dcx, phi_2dcx, M_12 = params

    U_dp1 = U0_1* (
        (phi_1 - phi_1x) * phi_1_prefactor
        - beta_1 * np.sin(phi_1) * np.cos(phi_1dc / 2)
        + delta_beta_1 * np.cos(phi_1) * np.sin(phi_1dc/2)
    )  + M_12  *  (phi_2 - phi_2x)

    U_dp2 = U0_2* (
        (phi_2 - phi_2x) * phi_2_prefactor
        - beta_2 * np.sin(phi_2) * np.cos(phi_2dc / 2)
        + delta_beta_2 * np.cos(phi_2) * np.sin(phi_2dc/2)
    )  + M_12 *  (phi_1 - phi_1x)

    U_dp1dc = U0_1* (
        g_1 * (phi_1dc - phi_1dcx)
        - 1/2 * beta_1 * np.cos(phi_1) * np.sin(phi_1dc / 2)
        + 1/2 * delta_beta_1 * np.sin(phi_1) * np.cos(phi_1dc/2)
    )

    U_dp2dc = U0_2* (
        g_2 * (phi_2dc -  phi_2dcx)
        - 1/2 * beta_2 * np.cos(phi_2) * np.sin(phi_2dc / 2)
        + 1/2 * delta_beta_2 * np.sin(phi_2) * np.cos(phi_2dc/2)
    )



    return -1 * np.array([U_dp1, U_dp2, U_dp1dc, U_dp2dc])

[phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound] = [4, 4, 4, 4]
# [U_0, g, beta, delta_beta, phi_x, phi_xdc ]
# fq_default_param = [1, 1, beta, d_beta, 0, 0]
coupled_fq_default_param = [(1, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
coupled_fq_default_param = np.append(np.array(coupled_fq_default_param).reshape(1, -1), 0)

coupled_fq_domain = [[-phi_1_bound, -phi_2_bound, -phi_1dc_bound, -phi_2dc_bound], [phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound]]
coupled_fq_pot = Potential(coupled_flux_qubit_pot, coupled_flux_qubit_force, 13, 4, default_params = coupled_fq_default_param,  relevant_domain = coupled_fq_domain)
