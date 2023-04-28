# Created 2019-09-23


from math import sqrt
from random import getrandbits
from numpy.random import RandomState




class RKDetermEulerStoch:
    """Runge Kutta Deterministic, Euler Stochastic Integrator."""

    def __init__(self, dynamic, rng=None, seed=None, nrandbits=32):

        if rng is None:
            if seed is None:
                seed = getrandbits(nrandbits)
            rng = RandomState(seed)

        self.seed = seed
        self.rng = rng
        self.get_determ_dsdt = dynamic.get_determ_dsdt
        self.get_stoch_dsdt = dynamic.get_stoch_dsdt

    def update_state(self, state, time, dt):

        get_determ_dsdt = self.get_determ_dsdt
        get_stoch_dsdt = self.get_stoch_dsdt
        rng = self.rng

        k1 = get_determ_dsdt(state,               time)
        k2 = get_determ_dsdt(state + k1 * dt / 2, time + dt / 2)
        k3 = get_determ_dsdt(state + k2 * dt / 2, time + dt / 2)
        k4 = get_determ_dsdt(state + k3 * dt,     time + dt)

        ds_determ = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        ds_stoch = get_stoch_dsdt(state, time, rng) * sqrt(dt)

        next_state = state + ds_determ + ds_stoch

        return next_state


import numpy as np

def SBS_step(f, g, x, dt, t, tol, max_order, max_subdivision):
    """
    SBS_step function implements the Super-Bulirsch-Stoer stochastic integration method. The function takes in the following inputs:

    Parameters:
    f: callable
    The deterministic function that describes the dynamics of the system.
    g: callable
    The stochastic function that describes the noise in the system.
    x: numpy.ndarray
    The current state of the system.
    dt: float
    The time step for the integration.
    t: float
    The current time.
    tol: float
    The tolerance level for the integration error.
    max_order: int
    The maximum order for the Bulirsch-Stoer extrapolation.
    max_subdivision: int
    The maximum number of subdivisions for the interval.

    Returns:
    x_new: numpy.ndarray
    The updated state of the system.
    error: float
    The estimated error of the integration.
    dt_new: float
    The recommended new time step for the next integration step.

    Note: This function uses the Super-Bulirsch-Stoer stochastic integration method to integrate the system dynamics, given the current state of the system and the specified parameters.
    """

    h = dt / max_order
    x_0 = x.copy()
    for j in range(max_subdivision):
        k = [np.zeros_like(x) for i in range(max_order+1)]
        k[0] = f(t, x)
        for i in range(max_order):
            t_i = t + i*h
            x_i = x_0 + h*sum(k[:i+1])/i
            k[i+1] = f(t_i, x_i)
        x_0 = x_0 + h*sum(k)/max_order

    # SBS stochastic step
    x1 = x.copy()
    x2 = x.copy()
    for j in range(max_subdivision):
        k = [np.zeros_like(x) for i in range(max_order+1)]
        k[0] = f(t, x1)
        for i in range(max_order):
            t_i = t + i*h
            w_i = np.random.normal(scale=np.sqrt(h), size=x.shape)
            x1_i = x1 + k[i]*h + g(t_i, x1)*w_i
            x2_i = x2 + (k[i]+k[i+1])/2*h + g(t_i+h/2, (x1+x2)/2)*(w_i+np.random.normal(scale=np.sqrt(h), size=x.shape))/np.sqrt(2)
            k[i+1] = f(t_i, x2_i)
            x1 = x1_i
            x2 = x2_i

    # error estimation
    err = np.linalg.norm((x1 - x2) / np.sqrt(2**max_subdivision - 1))

    # check if the error is smaller than the tolerance level
    if err < tol:
        return x1, err
    else:
        x_new1, err1 = SBS_step(f, g, x, dt/2, t, tol/2, max_order, max_subdivision)
        x_new2, err2 = SBS_step(f, g, x, dt/2, t + dt/2, tol/2, max_order, max_subdivision)
        return x_new1 + x_new2 - x, np.sqrt(err1**2 + err2**2)
