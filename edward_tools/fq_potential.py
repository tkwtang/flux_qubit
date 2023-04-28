import numpy as np
import os, sys

source_path = os.path.expanduser('~/Project/source/')
sys.path.append(source_path)
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol

I_p = 2e-6       # Amp
I_m = 7e-9       # Amp
R = 371          # ohm
C = 4e-9         # F
L = 1e-9         # H
PHI_0 = 2.067833848 * 1e-15

beta = 2 * np.pi * L * I_p / PHI_0
d_beta = 2 * np.pi * L * I_m / PHI_0


def flux_qubit_pot(phi, phi_dc, params):
    """
    2D 4-well potential.

    Parmeters
    -------------
    phi: ndaray of dimension [N, ]
    phi_dc: ndaray of dimension [N, ]

    params: list / tuple
    - [U_0, g, beta, delta_beta, phi_x, phi_xdc ]: correspond to the energy scale, gamma
    - phi_x: associated with asymmetry in the informational subspace, and will only take a nonzero value to help
      offset asymmetry from the delta_beta term in U'
    """
    U_0, g, beta, delta_beta, phi_x, phi_xdc = params

    u1 = 1/2 * (phi - phi_x)**2
    u2 = 1/2 * g * (phi_dc - phi_xdc)**2
    u3 = beta * np.cos(phi) * np.cos(phi_dc/2)
    u4 = delta_beta * np.sin(phi) * np.sin(phi_dc/2)

    U = U_0 * ( u1 + u2 + u3 + u4 )

    return U


def flux_qubit_force(phi, phi_dc, params):
    """
    2D 4-well potential.

    Parmeters
    -------------
    phi: ndaray of dimension [N, ]
    phi_dc: ndaray of dimension [N, ]

    params: list / tuple
    - [U_0, g, beta, delta_beta, phi_x, phi_xdc ]: correspond to the energy scale, gamma
    """
    U_0, g, beta, delta_beta, phi_x, phi_xdc = params

    U_dp = U_0 * (
        (phi - phi_x)
        - beta * np.sin(phi) * np.cos(phi_dc / 2)
        + delta_beta * np.cos(phi) * np.sin(phi_dc/2)
    )
    U_dpdc = U_0 * (
        g * (phi_dc - phi_xdc)
        - 1/2 * beta * np.cos(phi) * np.sin(phi_dc / 2)
        + 1/2 * delta_beta * np.sin(phi) * np.cos(phi_dc/2)
    )

    return [-U_dp,  -U_dpdc]

xy_bound = 4
# [U_0, g, beta, delta_beta, phi_x, phi_xdc ]
fq_default_param = [1, 1, beta, d_beta, 0, 0]
fq_default_param = [1, 0, 0, 0, 0, 0]
fq_domain = [[-xy_bound, -xy_bound], [xy_bound, xy_bound]]
fq_pot = Potential(flux_qubit_pot, flux_qubit_force, 6, 2, default_params = fq_default_param, \
                   relevant_domain = fq_domain)



def coupled_flux_qubit_pot(phi, phi_dc, params):
    """
    2D 4-well potential.

    Parmeters
    -------------
    phi: ndaray of dimension [N, ]
    phi_dc: ndaray of dimension [N, ]

    params: list / tuple
    - [U_0, g, beta, delta_beta, phi_x, phi_xdc ]: correspond to the energy scale, gamma
    - phi_x: associated with asymmetry in the informational subspace, and will only take a nonzero value to help
      offset asymmetry from the delta_beta term in U'
    """
    U_0, g, beta, delta_beta, phi_x, phi_xdc = params

    u1 = 1/2 * (phi - phi_x)**2
    u2 = 1/2 * g * (phi_dc - phi_xdc)**2
    u3 = beta * np.cos(phi) * np.cos(phi_dc/2)
    u4 = delta_beta * np.sin(phi) * np.sin(phi_dc/2)

    U = U_0 * ( u1 + u2 + u3 + u4 )

    return U


def coupled_flux_qubit_force(phi, phi_dc, params):
    """
    2D 4-well potential.

    Parmeters
    -------------
    phi: ndaray of dimension [N, ]
    phi_dc: ndaray of dimension [N, ]

    params: list / tuple
    - [U_0, g, beta, delta_beta, phi_x, phi_xdc ]: correspond to the energy scale, gamma
    """
    U_0, g, beta, delta_beta, phi_x, phi_xdc = params

    U_dp = U_0 * (
        (phi - phi_x)
        - beta * np.sin(phi) * np.cos(phi_dc / 2)
        + delta_beta * np.cos(phi) * np.sin(phi_dc/2)
    )
    U_dpdc = U_0 * (
        g * (phi_dc - phi_xdc)
        - 1/2 * beta * np.cos(phi) * np.sin(phi_dc / 2)
        + 1/2 * delta_beta * np.sin(phi) * np.cos(phi_dc/2)
    )

    return [-U_dp,  -U_dpdc]

xy_bound = 4
# [U_0, g, beta, delta_beta, phi_x, phi_xdc ]
coupled_fq_default_param = [1, 1, beta, d_beta, 0, 0]
coupled_fq_default_param = [1, 0, 0, 0, 0, 0]
fq_domain = [[-xy_bound, -xy_bound], [xy_bound, xy_bound]]
fq_pot = Potential(coupled_flux_qubit_pot, coupled_flux_qubit_force, 6, 2, default_params = coupled_fq_default_param, relevant_domain = coupled_fq_domain)
