import sys
import os
source_path = os.path.expanduser('~/Project/source/')
sys.path.append(source_path)
sys.path.append(os.path.expanduser('~/Project/source/simtools/'))


import numpy as np
from .coupled_fq_potential import coupled_fq_pot, coupled_fq_default_param
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from kyle_tools.multisim import SimManager, FillSpace
from SimRunner import SaveParams, SaveSimOutput, SaveFinalWork
from infoenginessims.simprocedures import basic_simprocedures as sp
from infoenginessims.simprocedures import running_measurements as rp
from infoenginessims.simprocedures import trajectory_measurements as tp
from infoenginessims.simprocedures.basic_simprocedures import ReturnFinalState, MeasureWorkDone, MeasureStepValue
from quick_sim import setup_sim

default_params_dict = {}

class coupledFluxQubitRunner(SimManager):
    def __init__(self, potential = coupled_fq_pot, name_func = [None, None], params = default_params_dict, potential_default_param = coupled_fq_default_param, storage_protocol = None, computation_protocol = None ):
        """
        params: parameters for the simulation such as time, lambda, theta and eta
        override_potential_parameter: to override the default parameter for the potential
        """
        self.potential = potential
        self.params = params
        self.save_name = name_func
        self.has_velocity = True
        self.override_potential_parameter = potential_default_param
        self.storage_protocol = storage_protocol
        self.computation_protocol = computation_protocol
        self.save_procs =  [SaveParams(), SaveSimOutput(), SaveFinalWork()]


    def verify_param(self, key, val):
        return True

    def initialize_sim(self):
        self.potential.default_params = self.override_potential_parameter
        self.eq_protocol = self.storage_protocol or  self.potential.trivial_protocol().copy()

        self.potential.default_params = np.array(self.override_potential_parameter)
        self.protocol = self.computation_protocol or self.potential.trivial_protocol().copy()
        # print(f"from fq_runner.py: system.protocol.t_i = {self.protocol.t_i}, system.protocol.t_f = {self.protocol.t_f}")

        self.eq_system = System(self.eq_protocol, self.potential)
        self.eq_system.axes_label = ["phi_1", "phi_2", "phi_1_dc", "phi_2_dc"]
        self.eq_system.has_velocity = self.has_velocity

        self.system = System(self.protocol, self.potential)
        self.system.axes_label = ["phi_1", "phi_2", "phi_1_dc", "phi_2_dc"]
        self.system.has_velocity = self.has_velocity


        # self.system.protocol.normalize()
        # self.system.protocol.time_stretch(np.pi/np.sqrt(1))


    def set_sim_attributes(self, init_state = None, manual_domain = None, axes = None, percentage = 0.1, verbose = True):
        if init_state is not None:
            print("use old initial_state")
            self.init_state = init_state
        else:
            print("generating new initial_state")
            self.init_state = self.eq_system.eq_state(self.params['N'], t=0, beta=self.params['beta'], manual_domain = manual_domain, axes = axes)

        as_step = max(1, int((self.system.protocol.t_f/self.params['dt'])/500))

        sampleSize = round(self.params['N'] * percentage)

        self.procs = self.set_simprocs(as_step, sampleSize)

        if verbose:
            print("from cfq_runner.py, The as_tep is", as_step)
            print("from cfq_runner.py, The dt is", self.params['dt'],)

        # edward added this, to override the 200 states only in all states.
        # self.procs[1] = sp.MeasureAllState()

        sim_kwargs = {
                        'damping':self.params['lambda'],
                        'temp':1/self.params['beta'],
                        'dt':self.params['dt'],
                        'procedures':self.procs,
                        'sim_params': self.params['sim_params']
            }

        self.sim = setup_sim(self.system, self.init_state, verbose = verbose,**sim_kwargs)
        self.sim.reference_system = self.eq_system
        return

    # def analyze_output(self):
    #     if not hasattr(self.sim.output, 'final_W'):
    #         final_state = self.sim.output.final_state
    #         init_state = self.sim.initial_state
    #         U0 = self.system.get_potential(init_state, 0) - self.eq_system.get_potential(init_state, 0)
    #         UF = self.eq_system.get_potential(final_state, 0) - self.system.get_potential(final_state, 0)
    #         final_W = U0 + UF
    #         setattr(self.sim.output, 'final_W', final_W)

    def set_simprocs(self, as_step, sampleSize):
        return [
            sp.ReturnFinalState(),
            sp.MeasureAllState(trial_request=np.s_[:sampleSize], step_request=np.s_[::as_step]),
            sp.MeasureWorkDone2(rp.get_dW)
            ]
