import sys, os
sys.path.append(os.path.expanduser('~/Project/source'))
#
import numpy as np
import importlib, os, hashlib, json, copy, time
from quick_sim import setup_sim
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
from sus.protocol_designer.protocol import sequential_protocol
from edward_tools import coupled_fq_protocol_library, cfq_runner
from edward_tools.coupled_fq_potential import coupled_flux_qubit_pot, coupled_flux_qubit_force, coupled_fq_pot
import edward_tools.cfq_batch_sweep as cfq_batch_sweep
from edward_tools.visualization import animate_sim_flux_qubit
import kyle_tools as kt
import matplotlib.pyplot as plt
from PARAMETER_INPUT import *
from PARAMETER_INPUT import _lambda, _theta, _eta
import edward_tools.visualization as visualization
import matplotlib.animation as animation

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

create_simple_protocol_parameter_dict = coupled_fq_protocol_library.create_simple_protocol_parameter_dict
create_system_from_storage_and_computation_protocol = coupled_fq_protocol_library.create_system_from_storage_and_computation_protocol
create_system = coupled_fq_protocol_library.create_system

# print(PHI_0, _lambda, _theta, _eta,)
"""
# step 1: Define potential
"""
coupled_fq_default_param = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x_c0]
[phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound] = np.array([4, 4, 4, 4])/time_scale_factor

coupled_fq_domain = [[-phi_1_bound, -phi_2_bound, -phi_1dc_bound, -phi_2dc_bound], \
                     [phi_1_bound, phi_2_bound, phi_1dc_bound, phi_2dc_bound]]

coupled_fq_pot = Potential(coupled_flux_qubit_pot, coupled_flux_qubit_force, 14, 4,\
                           default_params = coupled_fq_default_param,  relevant_domain = coupled_fq_domain)

"""
# step 2: Define initial condition and protocol
"""
manual_domain=[np.array([-5, -5]), np.array([5, 5])]
# phi_1_dcx, phi_2_dcx = 3, 3
phi_1_dcx, phi_2_dcx = 0, 0
phi_1_dc, phi_2_dc = phi_1_dcx, phi_2_dcx

# gamma, beta_1, beta_2 = 0, 0, 0
gamma = 20
# d_beta_1, d_beta_2 = 0.6, 0.6
d_beta_1, d_beta_2 = 0, 0
params['sim_params'] = [_lambda, _theta, _eta]

N =  params['N'] = 50_000
params['dt'] = 1/5_00


max_n_per_batch = 10_000
batch_list = []

while N > max_n_per_batch:
    batch_list.append(max_n_per_batch)
    N -= max_n_per_batch
batch_list.append(N)

initial_parameter_dict = {
        "U0_1": U0_1,     "U0_2": U0_2,     "gamma_1": gamma,  "gamma_2": gamma,
        "beta_1": beta_1,   "beta_2": beta_2,   "d_beta_1": d_beta_1 ,   "d_beta_2": d_beta_2,
        "phi_1_x": 0,  "phi_2_x": 0,  "phi_1_dcx": phi_1_dcx,  "phi_2_dcx": phi_2_dcx,
        "M_12": 0, 'x_c': x_c
}

# protocol_list = [
#     {
#         "duration": 100/time_scale_factor , "phi_2_dcx": 3/time_scale_factor, "name": "mix in y direction"
#     },
#
#
#
#     {"duration": 18/time_scale_factor, "M_12": -0.9, "name": "conditional tilt"}, # conditional tilt
#
#     {"duration": 16/time_scale_factor, "phi_2_dcx": 0, "name": "raise the barrier"}, # raise the barrier
#     {"duration":130,"phi_1_dcx":3,"name":"mix in x direction"},
#     {"duration":100,"name":"fix"},
#     {"duration":100,"phi_1_dcx":0,"M_12":0,"name":"4 well potential "},
#     {"duration":20,"name":"4 well potential (constant)"}
#
#
# ]

#
# protocol_list = [
#     {"duration":10,"phi_2_dcx":3,"name":"mix in y direction"},
#     {"duration":2,"name":"fix"},
#     {"duration":10,"M_12":-0.9,"name":"conditional tilt"},
#     {"duration":2,"name":"fix"},
#     {"duration":10,"phi_2_dcx":0,"name":"raise the barrier"},
#     {"duration":2,"name":"fix"},
#     {"duration":10,"M_12":0,"name":"4 well potential (constant)"},
#     {"duration":2,"name":"fix"},
#     {"duration":30,"phi_1_dcx":3,"name":"mix in x direction"},
#     {"duration":10,"name":"fix"},
#     {"duration":10,"phi_1_dcx":0,"M_12":0,"name":"4 well potential "},
#     {"duration":2,"name":"4 well potential (constant)"}
# ]
params["comment"] = "test for longer fixed time to see if the statistical curve have a flat line when the protocol is fixed"
protocol_list = [
    {"duration":2,"phi_2_dcx":3,"name":"mix in y direction"},
    {"duration":10,"name":"fix"},
    {"duration":2,"M_12":-0.9,"name":"conditional tilt"},
    {"duration":10,"name":"fix"},
    {"duration":2,"phi_2_dcx":0,"name":"raise the barrier"},
    {"duration":10,"name":"fix"},
    {"duration":2,"M_12":0,"name":"4 well potential (constant)"},
    {"duration":10,"name":"fix"},
    {"duration":2,"phi_1_dcx":3,"name":"mix in x direction"},
    {"duration": 10, "name":"fix"},
    {"duration":2,"phi_1_dcx":0,"M_12":0,"name":"4 well potential "},
    {"duration":10,"name":"4 well potential (constant)"}
]


"""
# step 3: create the relevant storage protocol and computation protocol
"""

# check if the simulation is a sweep or not
def multiprocessing_worker_function(_N):
    # print(protocol_list)
    _params = copy.deepcopy(params)
    _params["N"] = _N
    computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, protocol_list)
    storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict)

    """
    # step 4: create the coupled_fq_runner
    """
    cfqr = cfq_runner.coupledFluxQubitRunner(potential = coupled_fq_pot, params = _params, storage_protocol= storage_protocol, computation_protocol= comp_protocol)
    cfqr.initialize_sim()
    cfqr.set_sim_attributes()
    init_state_saved = cfqr.init_state

    """
    step 5: Run sweep simulation in parallel
    """
    manual_domain=[np.array([-5, -5])/time_scale_factor, np.array([5, 5])/time_scale_factor]
    params['sim_params'] = [_lambda, _theta, _eta]

    """
    step 5a: single simulation
    """
    simResult = cfq_batch_sweep.simulateSingleCoupledFluxQubit(_params, initial_parameter_dict, protocol_list, initial_state = init_state_saved, manual_domain = manual_domain, phi_1_dcx = phi_1_dcx,  phi_2_dcx = phi_2_dcx, frameRate = -1)
    #
    # cfqr = simResult["cfqr"]
    # cfq_batch_sweep.saveSimulationResult(simResult, U0_1, timeOrStep = 'step', save = True)
    # simResult["cfqr"] = 0
    simResult["ani"] = 0



    return simResult

def test_function(_N):
    print("testing")
    return "finish"


from multiprocessing import Process, Pool, cpu_count
if __name__ == "__main__":  # confirms that the code is under main function
    number_of_cpu = int(cpu_count() * 0.4)
    print(f"number of cpu: {number_of_cpu}")

    # print(f"number of cpu: {number_of_cpu}")
    p = Pool(processes=number_of_cpu)
    print("testing")
    print(batch_list)
    dataArray = p.map(multiprocessing_worker_function, batch_list)
    # dataArray = p.map(test_function, batch_list)
    p.close()
    print("finish all")
    print(dataArray)

    parameterArray = np.array([item["params"] for item in dataArray])
    np.save("parameterArray.npy", parameterArray)

    # new simualtion data
    new_simulation_id = hashlib.sha256(bytes(str(time.time()), encoding='utf8')).hexdigest()
    simulation_data = dataArray[0]["simulation_data"]
    simulation_data["simulation_id"] = new_simulation_id
    simulation_data["simulation_date"] = str(simulation_data["simulation_date"])

    n_array = np.array([len(item["work_distribution"]) for item in dataArray])
    total_N = np.sum(n_array)



    # combine animation
    ani_max_N = 10000
    if total_N < ani_max_N:
        sampleNumber = n_array
    else:
        sampleNumber = n_array / total_N * ani_max_N

    sampleRatio = np.repeat(1, len(n_array))
    sampleTrajectoryArray = [data["cfqr"].sim.output.all_state["states"] for data in dataArray]
    trajectoryNumber = np.array([len(_t) for _t in sampleTrajectoryArray])
    sampleNumber = trajectoryNumber * sampleRatio
    sampleIndexArray = [np.random.randint(int(tn), size=int(sn)) for tn, sn in zip(trajectoryNumber, sampleNumber)]

    allSampleTrajectory = np.concatenate(sampleTrajectoryArray)

    # animations
    vmin, vmax = 0, 100
    frameRate = 10

    manual_domain=[[-5, -5], [5, 5]]
    time_range = [0, np.sum([item["duration"] for item in dataArray[0]["protocol_list_item"]])]
    cfqr = dataArray[0]["cfqr"]

    ani,_,_ = visualization.animate_sim_flux_qubit(allSampleTrajectory, system = cfqr.system ,
                        times = time_range, frame_skip=frameRate, color_by_state=True,
                                                       vmin = vmin, vmax = vmax,
                                                       manual_domain = manual_domain)

    FFwriter = animation.FFMpegWriter(fps=10)

    # video = ani.to_html5_video()
    # html = display.HTML(video)
    # display.display(html)

    ani.save(f'coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{new_simulation_id}_szilard_engine.mp4', writer = FFwriter)
    # simResult["ani"].save(f'coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{simResult["simulation_data"]["simulation_id"]}_szilard_engine.mp4', writer = FFwriter)

    work_distribution_array =  [item["work_distribution"] for item in dataArray]

    [np.save(f"coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{new_simulation_id}_work_distribution_{i}.npy", wd) for i, wd in enumerate(work_distribution_array)]

    combinedWorkDistributionArray = np.concatenate(work_distribution_array)
    np.save(f"coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{new_simulation_id}_work_distribution.npy", combinedWorkDistributionArray)

    # *** jarzynski_term ***
    new_jarzynski_term = np.mean(np.exp(-combinedWorkDistributionArray))

    # save figure
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(combinedWorkDistributionArray, bins = 30)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, f"Jarzyn: {new_jarzynski_term: .3g}", transform=ax.transAxes, fontsize=14, verticalalignment='top',bbox=props)
    # plt.hist(work_distribution)
    # if save:
        # np.save(f'coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{new_simulation_id}_work_distribution.npy', work_distribution)
    plt.savefig(f'coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{new_simulation_id}_work_distribution.png')

    # *** work statistics ***
    work_statistic_array =  np.array([item["work_statistic"]  for item in dataArray])
    [np.save(f"coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{new_simulation_id}_work_statistic_{i}.npy", ws) for i, ws in enumerate(work_statistic_array)]

    ratio = n_array / total_N
    weightedArray = [np.multiply(item,  r)  for r, item in zip(ratio, work_statistic_array)]
    combinedWorkStatistic = np.sum(weightedArray, axis = 0)

    # plot graphs
    work_mean, work_std = combinedWorkStatistic[:, 0], combinedWorkStatistic[:, 1]
    skip_step = int(len(work_mean) * 0.05)
    step_array = np.arange(0, work_mean.shape[0])
    plt.figure(figsize=(10, 7))
    plt.errorbar(step_array[::skip_step], work_mean[::skip_step], yerr = work_std[::skip_step])
    plt.savefig(f'coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{new_simulation_id}_work_statistic.png')

    # fidelity
    fidelityData = [x["fidelity"] for x in dataArray]

    with open("fidelityJSON.json", "w") as f:
        json.dump(fidelityData, f)

    combinedFidelity = []

    def processFidelityDict(array):
        resultDict = {"location": "", "count": 0}

        for x in array:
            resultDict["location"] = x["location"]
            resultDict["count"] += x["count"]

        return resultDict

    for item in np.array(fidelityData).T:
        _initial = [x["initial"] for x in item]
        _final =  [x["final"] for x in item]
        combinedFidelity.append({
            "initial": processFidelityDict(_initial),
            "final":   [processFidelityDict(x) for x in np.array(_final).T]
        })

    with open("combinedFidelity.json", "w") as f:
        json.dump(combinedFidelity, f)

    # new save object
    newSaveObject = {
        "params":                           dataArray[0]["params"],
        "initial_parameter_dict":   dataArray[0]["initial_parameter_dict"],
        "protocol_list_item":         dataArray[0]["protocol_list_item"],
        "jarzynski_term":              new_jarzynski_term,
        "fidelity":                           combinedFidelity,
        "sim_params":                 dataArray[0]["params"]["sim_params"],
        "simulation_data":            simulation_data,
        # "comment":                       dataArray[0]["comment"]
    }

    print(newSaveObject)

    with open("coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/gallery.json") as f:
        jsonData = json.load(f)
        jsonData.append(newSaveObject)
        with open("coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/gallery.json", "w+") as fw:
            json.dump(jsonData, fw, cls = NumpyArrayEncoder)

    # print(combinedWorkDistributionArray, combinedWorkStatistic, combinedFidelity)
    # procs = []
    # # instantiating process with arguments
    # for i, _N in enumerate(batch_list):
    #     proc = Process(target=multiprocessing_worker_function, args=[_N])
    #     procs.append(proc)
    #     print(f"Process {i}, number_of_particle = {_N}")
    #     proc.start()
    #
    # # complete the processes
    # for proc in procs:
    #     proc.join()
    # print("process finished")
