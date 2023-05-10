# Created 2019-09-30 from OLD4_simulation.py
#
# Part of September 2019 refactor.
import sys
from math import sqrt
from numpy import array, empty
from random import Random, getrandbits

class Bunch:
    """For making generic objects with arbitrary attribute assignments."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_hist_mean(hist):

    bins = hist[1]
    vals = hist[0]

    midpoints = bins[:-1] + (bins[1] - bins[0]) / 2

    mean = (vals * midpoints).sum() / vals.sum()

    return mean


def attempt(func, *args, **kwargs):
    """Attempts to run func with args and kwargs.

    Will escape if the exception is an AttributeError."""

    try:
        func(*args, **kwargs)

    except AttributeError:
        pass

def get_normalized_exponent(a):
    '''Finds the normalized scientific notation exponent of a number.

    Specifically, the significand would be a number between 1 and 10, left
    inclusive, right exclusive.  If the number is zero, returns None.
    '''

    if a == 0:
        return None

    b = abs(a)

    if floor(b) > 0:
        n = 0
        while floor(b / 10 ** n) > 10:
            n += 1
    else:
        n = -1
        while floor(b / 10 ** n) < 1:
            n -= 1
    return n

def get_min_normalized_exponent(A):
    '''Finds the minimum of the normalized notation exponents of numbers.

    Considers only non-zero elements of the numbers given.  Returns None if all
    numbers are 0.'''

    min_exp = None
    for a in A:
        exp = get_normalized_exponent(a)
        if exp is not None:
            min_exp = exp if min_exp is None else min(exp, min_exp)

    return min_exp

def get_significand(a, exp=None):
    '''Returns the sci-notation significand of a number given an exponent.

    If exp is None, the exponent is the normalized notation exponent.
    '''

    if exp is None:
        exp = get_normalized_exponent(a)

    return '{:.2f}'.format(a / 10 ** exp)

def get_common_exp_significands(A, exp=None):
    '''Returns the sci-notation significands of numbers given an exponent.

    If exp is None, the exponent is the minimum of the normalized notation
    exponents of the numbers.
    '''

    if exp is None:
        exp = get_min_normalized_exponent(A)

    return [get_significand(a, exp) for a in A]

def get_scinotation_rep(a, exp=None):
    '''Returns the number in sci-notation given an exponent.

    If exp is None, the exponent is the normalized notation exponent.
    '''

    if exp is None:
        exp = get_normalized_exponent(a)

    return '{:.2f}e{}'.format(a / 10 ** exp, exp)

def get_common_exp_scinotation_reps(A, exp=None):
    '''Returns the numbers in sci-notation given an exponent.

    If exp is None, the exponent is the minimum of the normalized notation
    exponents of the numbers.
    '''

    if exp is None:
        exp = get_min_normalized_exponent(A)

    return [get_scinotation_rep(a, exp) for a in A]



class Simulation:
    """The main simulation object.  Initializing runs a sim by default."""

    def __init__(self, update_state, procedures, nsteps, dt,
                 initial_state=None,
                 initial_dist=None, ntrials=None, sample_seed=None,
                 # dynamic=dynamic, system=None, protocol=None, infospace=None,
                 to_auto_run=False, nrandbits=32):
        """Initializes a simulation instance.

        Should provide at least one of initial_state and initial_dist.
        Providing both leaves initial_dist without any particular function.
        If only initial_dist is provided, should also provide ntrials.
        """
        print("This line is for the initializatoin of the simulation object.")
        print(f"number of steps: {nsteps}, dt: {dt}")

        if initial_state is not None:

            ntrials = initial_state.shape[0]

        else:  # Assume initial_dist and ntrials is given

            if sample_seed is None:
                sample_seed = getrandbits(nrandbits)

            initial_state = initial_dist.sample(ntrials, sample_seed)

            self.sample_seed = sample_seed

        if initial_dist is not None:
            self.initial_dist = initial_dist

        self.update_state = update_state
        self.procedures = procedures
        self.nsteps = nsteps
        self.dt = dt
        self.ntrials = ntrials
        self.initial_state = initial_state

        # if dynamic is not None:
        #     self.dynamic = dynamic
        # if system is not None:
        #     self.system = system
        # if protocol is not None:
        #     self.protocol = protocol
        # if infospace is not None:
        #     self.infospace = infospace

        if to_auto_run:
            self.output = self.run()

    def run(self, verbose=False):

        initial_state = self.initial_state
        update_state = self.update_state
        procedures = self.procedures

        # this line is to solve the problem of the number of steps
        # nsteps = round(1/self.dt)
        # print("The nsteps in simulation is ", nsteps)
        nsteps = self.nsteps
        dt = self.dt
        self.current_time = 0

        self.current_state = array(initial_state)
        self.next_state = empty(initial_state.shape)

        current_state = self.current_state
        next_state = self.next_state

        # Do initial tasks
        for procedure in procedures:
            procedure.do_initial_task(self)

        # Main evolution loop
        # self.current_step = 0
        # while self.current_step < nsteps:
        for step in range(nsteps):
            if (step > nsteps - 10):
                print("From simulation.py", step, nsteps)
            time = step * dt

            self.current_step = step
            self.current_time = time

            next_state[...] = update_state(current_state, time, dt)

            # Do intermediate tasks
            for procedure in procedures:
                procedure.do_intermediate_task()

            current_state[...] = next_state[...]

            if verbose:
                print("\r step {} of {} complete".format(step+1, nsteps),end="")

            if any( [p.terminate for p in procedures if hasattr(p,'terminate')] ):
                self.final_time = time+dt
                self.nsteps = step
                break

        # Do final tasks and get outputs
        outputs = Bunch()
        for procedure in procedures:

            output = procedure.do_final_task()

            if hasattr(procedure, 'output_name'):
                setattr(outputs, procedure.output_name, output)

        return outputs
