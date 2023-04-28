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
