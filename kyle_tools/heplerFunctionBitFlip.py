import numpy as np
import os, json
import matplotlib.pyplot as plt

def evaluateFidelityRate(allstate):
    # indices to indicate which particles have initial bit 0 and initial bit 1
    bit_0_indices_initial = np.where(allstate[:, 0, 0, 0] < 0)
    bit_1_indices_initial = np.where(allstate[:, 0, 0, 0] > 0)

    # group 0 bits are the particles with initial x position < 0
    # group 1 bits are the particles with inital x position > 0
    group_0_bits = allstate[bit_0_indices_initial]
    group_1_bits = allstate[bit_1_indices_initial]

    # in the following code, it compares the initial and final states of the particles of group 0 and group 1
    # particles, and then find out the total successful bit flip rate
    total_trials = allstate.shape[0]
    group_0_initial = group_0_bits[:, 0, 0, 0] < 0
    group_0_final = group_0_bits[:, -1, 0, 0] > 0
    goodNumber_group_0 = np.sum(group_0_initial == group_0_final)

    group_1_initial = group_1_bits[:, 0, 0, 0] > 0
    group_1_final = group_1_bits[:, -1, 0, 0] < 0
    goodNumber_group_1 = np.sum(group_1_initial == group_1_final)

    goodNumber = goodNumber_group_0 + goodNumber_group_1
    fidelity_rate = goodNumber / total_trials * 100
    return fidelity_rate

beta = 1.0
def checkJE(final_W):
    return np.sum(np.exp(-beta * final_W))/len(final_W)


def singleSimulation(sweepObject, system, label):
    system.change_params(_sweepObject)
    system.run_sim(verbose=True)

    _sweepObject_save = _sweepObject.copy()
    _sweepObject_save["fidelity_rate"] = evaluateFidelityRate(system.sim.output.all_state["states"])
    _sweepObject_save["meanFinalWork"] = np.mean(system.sim.output.final_W)
    _sweepObject_save["avg_exp_neg_beta_W"] = checkJE(bfr.sim.output.final_W)




def sweepFunction(sweepObject, system, label, save = True, overwrite = False, jsonPath = "bitFlipSimData/bitFlipResult.json", txtPath = "bitFlipSimData/sim_data.txt"):
    """
    sweepObject = a dictionary in the format of
        "_time_stretch": 4,
        "_damping": [float(x) for x in np.linspace(1, 10, 10)],
        "_gamma": 4,
        "_theta": 80,
        "_eta": 10

    save = Boolean, if False, do not save the object as gif and json

    overwrite = Boolean, to overwrite the existing parameter list already simulated
    """

    _sweepObject = sweepObject.copy()
    sweepParameter = ""
    sweepList = []

    # to find out which parameter is the sweepParameter
    for key, val in _sweepObject.items():
        if type(val) is list or type(val) is np.ndarray:
            sweepParameter = key
            sweepList = val.copy()

    # genereate sets of parameter lists to do simulations
    for index, x in enumerate(sweepList):
        _sweepObject[sweepParameter] = x
        system.change_params(_sweepObject)

        print("=" * 10 + f" start ({index+1}/{len(sweepList)}) " + "=" * 10)

        parameterList = [f"{parameter}_{value:.5g}" for (parameter, value) in list(_sweepObject.items())]
        fileName = "_".join(parameterList)
        print(fileName)


        with open("bitFlipSimData/sim_data.txt", "r+") as rf:
            lines = rf.readlines()
            if fileName + "\n" in lines:
                print("- This set of parameter is already simulated.")
                print("- Nothing is saved.")
                print("- Skip to the next paramter.")
                # continue
            else:
                    #
                    with open("bitFlipSimData/sim_data.txt", "a+") as wf:
                        wf.write(fileName + "\n")

        system.run_sim(verbose=True)

        _sweepObject_save = _sweepObject.copy()
        _sweepObject_save["fidelity_rate"] = evaluateFidelityRate(system.sim.output.all_state["states"])
        _sweepObject_save["meanFinalWork"] = np.mean(system.sim.output.final_W)
        _sweepObject_save["avg_exp_neg_beta_W"] = checkJE(bfr.sim.output.final_W)
        _sweepObject_save["label"] = label

        print(f"fidelity_rate = {_sweepObject_save['fidelity_rate']}")
        print(f"meanFinalWork = {_sweepObject_save['meanFinalWork']}")
        print(f"<exp(-beta_W)> = {_sweepObject_save['avg_exp_neg_beta_W']}")

        print("=" * 10 + f" finished ({index+1}/{len(sweepList)}) " + "=" * 10)
        if save:
            with open(jsonPath, "r") as rf:
                data = json.load(rf)
                data.append(_sweepObject_save)

                with open(jsonPath, "w") as wf:
                    json.dump(data, wf)
    # if notification == True:
    #     from notifypy import Notify
    #     notification = Notify()
    #     notification.title = "Finish Simulation"
    #     notification.message = "The simulation is finished."
    #     notification.send()

def sweepAnalysisFunction(parameter, label, jsonPath = "bitFlipSimData/bitFlipResult.json"):
    print(label)
    with open(jsonPath, "r") as rf:
        data = json.load(rf)
        filteredObject = filter(lambda x: x["label"] == label, data)

    parameter_value_array = []
    fidelity_rate_array = []
    meanFinalWork_array = []

    for parameterDict in filteredObject:
        parameter_value_array.append(parameterDict[parameter])
        fidelity_rate_array.append(parameterDict["fidelity_rate"])
        meanFinalWork_array.append(parameterDict["meanFinalWork"])
    print("parameters: ", parameter_value_array)
    print("fidelity: ", fidelity_rate_array)
    print("meanFinalWork: ", meanFinalWork_array)


    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    ax[0].plot(parameter_value_array, fidelity_rate_array)
    ax[0].title.set_text(f'fiderity x vs {label}')
#     ax[0].legend()

    ax[1].plot(parameter_value_array, meanFinalWork_array)
    ax[1].title.set_text(f'mean final work x vs {label}')
#     ax[1].legend()
#     plt.plot(parameter_value_array, fidelity_rate_array)

#     plt.plot(parameter_value_array, meanFinalWork_array)


def get_energy_time_series(system, all_state):
    """
    input:
    - all_state: the states of all the particles in the system at different time

    output:
    - a time series of shape (ntrial, nsteps, dim, (KE, PE, TE))
    """
    number_of_particles = all_state.shape[0]
    number_of_steps = all_state.shape[1]

    # create an empty energy time series. The last dimension is KE, PE and Total E
    energy_time_series_shape = list(all_state.shape)
    energy_time_series_shape[-1] = 3
    energy_time_series = np.empty(energy_time_series_shape)


    reshape_array = list(all_state.shape)[:2]
    reshape_array.append(1)

    # t = x * system.sim.dt
    energy_time_series[:, :, :, 0] = system.system.get_kinetic_energy(all_state).reshape(reshape_array) # KE
    energy_time_series[:, :, :, 1] = system.system.get_potential(all_state, 0).T.reshape(reshape_array) # PE
    energy_time_series[:, :, :, 2] = energy_time_series[:, :, :, 0] + energy_time_series[:, :, :, 1] # TE

    return energy_time_series

def plot_energy_time_series(number_of_particles, energy_time_series, step = 100):
    """
    plot then energy time series
    """
    fig, ax = plt.subplots(1, 3, figsize=(18,4))

    ax[0].set_xlabel("steps")
    ax[0].set_ylabel("KE")
    for x in range(number_of_particles)[0::step]:
        ax[0].plot(energy_time_series[x, :, :, 0])

    ax[1].set_xlabel("steps")
    ax[1].set_ylabel("PE")
    for x in range(number_of_particles)[0::step]:
        ax[1].plot(energy_time_series[x, :, :, 1]) # PE graph

    ax[2].set_xlabel("steps")
    ax[2].set_ylabel("TE")
    for x in range(number_of_particles)[0::step]:
        ax[2].plot(energy_time_series[x, :, :, 2]) # TE graph

    plt.show()
