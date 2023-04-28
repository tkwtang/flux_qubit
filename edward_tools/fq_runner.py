import sys
import os
source_path = os.path.expanduser('~/Project/source/')
sys.path.append(source_path)
sys.path.append(os.path.expanduser('~/Project/source/simtools/'))

import numpy as np
from .fq_potential import fq_pot, fq_default_param
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from kyle_tools.multisim import SimManager, FillSpace
from SimRunner import SaveParams, SaveSimOutput, SaveFinalWork
from infoenginessims.simprocedures import basic_simprocedures as sp
from infoenginessims.simprocedures import running_measurements as rp
from infoenginessims.simprocedures import trajectory_measurements as tp
from infoenginessims.simprocedures.basic_simprocedures import ReturnFinalState
from quick_sim import setup_sim

default_params_dict = {}

class fluxQubitRunner(SimManager):
    def __init__(self, name_func = [None, None], params = default_params_dict, potential_default_param = fq_default_param):
        self.potential = fq_pot
        self.params = params
        self.save_name = name_func
        self.has_velocity = True
        self.override_potential = potential_default_param

        self.save_procs =  [SaveParams(), SaveSimOutput(), SaveFinalWork()]

    def verify_param(self, key, val):
        return True

    def initialize_sim(self):
        self.potential.default_params = self.override_potential
        self.eq_protocol = self.potential.trivial_protocol().copy()

        self.potential.default_params = np.array(self.override_potential)
        self.protocol = self.potential.trivial_protocol().copy()
        print(f"from fq_runner.py: system.protocol.t_i = {self.protocol.t_i}, system.protocol.t_f = {self.protocol.t_f}")

        self.eq_system = System(self.eq_protocol, self.potential)
        self.eq_system.has_velocity = self.has_velocity

        self.system = System(self.protocol, self.potential)
        self.system.has_velocity = self.has_velocity

        self.system.protocol.normalize()
        # self.system.protocol.time_stretch(np.pi/np.sqrt(1))


    def set_sim_attributes(self):
        self.init_state = self.eq_system.eq_state(self.params['N'], t=0, beta=self.params['beta'])

        as_step = max(1, int((self.system.protocol.t_f/self.params['dt'])/500))

        self.procs = self.set_simprocs(as_step)

        print()
        print("The as_tep is", as_step)
        print("The dt is", self.params['dt'],)

        # edward added this, to override the 200 states only in all states.
        self.procs[2] = sp.MeasureAllState()

        sim_kwargs = {
                        'damping':self.params['lambda'],
                        'temp':1/self.params['beta'],
                        'dt':self.params['dt'],
                        'procedures':self.procs,
                        'sim_params': self.params['sim_params']
            }

        self.sim = setup_sim(self.system, self.init_state, **sim_kwargs)
        self.sim.reference_system = self.eq_system
        return

    def analyze_output(self):
        if not hasattr(self.sim.output, 'final_W'):
            final_state = self.sim.output.final_state
            init_state = self.sim.initial_state
            U0 = self.system.get_potential(init_state, 0) - self.eq_system.get_potential(init_state, 0)
            UF = self.eq_system.get_potential(final_state, 0) - self.system.get_potential(final_state, 0)
            final_W = U0 + UF
            setattr(self.sim.output, 'final_W', final_W)


    def set_simprocs(self, as_step):
        return [
            sp.ReturnFinalState(),
            sp.TerminateOnMean(rp.get_time_constant_work, target=self.params['target_work'], step_request=np.s_[::as_step], output_name='all_W'),
            sp.MeasureAllState(trial_request=np.s_[:200], step_request=np.s_[::as_step]),
            tp.CountJumps(output_name='jump_trajectories', state_slice=np.s_[...,0]),
            ]
