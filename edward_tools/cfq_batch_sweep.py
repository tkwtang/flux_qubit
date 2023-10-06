import copy, datetime, socket
import numpy as np
import edward_tools.couple_flux_qubit_metrics as couple_flux_qubit_metrics
import edward_tools.coupled_fq_protocol_library as coupled_fq_protocol_library
import edward_tools.visualization as visualization
from edward_tools import coupled_fq_protocol_library, cfq_runner
coupled_fq_runner = cfq_runner
from IPython import display
from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import time, datetime, json, hashlib

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

create_system = coupled_fq_protocol_library.create_system
get_potential_shot_at_different_t = coupled_fq_protocol_library.get_potential_shot_at_different_t
create_simple_protocol_parameter_dict = coupled_fq_protocol_library.create_simple_protocol_parameter_dict
create_system_from_storage_and_computation_protocol = coupled_fq_protocol_library.create_system_from_storage_and_computation_protocol
mapping_state_1_to_state_2_dict = {"00": ["00", "10"], "01": ["00", "10"], "10": ["01", "11"], "11": ["01", "11"]}

def simulateSingleCoupledFluxQubit(params, initial_parameter_dict, protocol_list, mapping_state_1_to_state_2_dict = mapping_state_1_to_state_2_dict, phi_1_dcx = 0, phi_2_dcx = 0, percentage = 0.1, initial_state = None, manual_domain = None, frameRate = 10, comment = ""):
    """
    The main object to perform simulations.
    """
    
    # base information
    start_time = time.time()
    now = str(start_time)
    sim_id = hashlib.sha256(bytes(now, encoding='utf8')).hexdigest()

    computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, protocol_list)
    storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict)
    cfqr = coupled_fq_runner.coupledFluxQubitRunner(params = params, storage_protocol= storage_protocol, \
                                    computation_protocol= comp_protocol)
    cfqr.initialize_sim()
    cfqr.run_sim(init_state = initial_state, percentage = percentage)
    cfqr.system.protocol_list = protocol_list

    all_state = cfqr.sim.output.all_state['states']
    final_state = cfqr.sim.output.final_state
    phi_1_and_phi_2_all_state = all_state[:, :, (0, 1), :]

    # fidelity_test
    initial_phi_1_phi_2 = all_state[:, 0, (0, 1), :]
    final_phi_1_phi_2   = all_state[:, -1, (0, 1), :]
    fidelity = couple_flux_qubit_metrics.fidelityEvaluation(initial_phi_1_phi_2, final_phi_1_phi_2, mapping_state_1_to_state_2_dict)

    # animations
    vmin, vmax = 0, 100
    phi_1_dc, phi_2_dc = phi_1_dcx, phi_2_dcx
    cfqr.system.protocol_list = protocol_list
    time_range = (computation_protocol_parameter_dict["t"][0], computation_protocol_parameter_dict["t"][-1])
    ani,_,_ = visualization.animate_sim_flux_qubit(all_state, system = cfqr.system ,
                                                   times = time_range, frame_skip=frameRate, color_by_state=True,
                                                   vmin = vmin, vmax = vmax,
                                                   manual_domain = manual_domain)
    # return {
    #     "cfqr": cfqr,
    #     "fidelity": fidelity,
    #     "work_distribution": cfqr.sim.work_dist_array_2, # work_dist_array_2 is for get_dW and work_dist_array is not the correct work done
    #     "work_statistic": cfqr.sim.work_statistic_array,
    #     "ani": ani,
    #     "params": params,
    #     "initial_parameter_dict": initial_parameter_dict,
    #     "protocol_list_item": protocol_list,
    #     "simulation_time": time.time() - start_time,
    #     "simulation_date": datetime.date.today(),
    #     "simulation_id": sim_id
    # }

    return {
        "cfqr": cfqr,
        "fidelity": fidelity,
        "work_distribution": cfqr.sim.work_dist_array_2, # work_dist_array_2 is for get_dW and work_dist_array is not the correct work done
        "work_statistic": cfqr.sim.work_statistic_array,
        "ani": ani,
        "params": params,
        "initial_parameter_dict": initial_parameter_dict,
        "protocol_list_item": protocol_list,
        "simulation_data":{
            "simulation_time": time.time() - start_time,
            "simulation_date": str(datetime.date.today()),
            "simulation_id": sim_id
        },
        "comment": comment
    }

def simulateCoupledFluxQubit(params, initial_parameter_dict, protocol_list_item, init_state = False, phi_1_dcx = 0, phi_2_dcx = 0, percentage = 0.1, frameRate = 10, verbose = False, comment = ""):
    """
    The main object to perform simulations.
    """
    subStepIndex =  protocol_list_item["substepIndex"]
    sweepKey =  protocol_list_item["sweepKey"]
    sweepParameter =  protocol_list_item["sweepParameter"]
    protocol_list = protocol_list_item["protocol_list"]

    computation_protocol_parameter_dict = coupled_fq_protocol_library.customizedProtocol(initial_parameter_dict, protocol_list)
    storage_protocol, comp_protocol = create_system(computation_protocol_parameter_dict)
    cfqr = coupled_fq_runner.coupledFluxQubitRunner(params = params, storage_protocol= storage_protocol,  computation_protocol= comp_protocol)

    cfqr.run_sim(init_state = init_state, percentage = percentage, verbose = False)

    all_state = cfqr.sim.output.all_state['states']
    final_state = cfqr.sim.output.final_state
    # final_W = cfqr.sim.output.final_W
    # all_W = cfqr.sim.output.all_W

    phi_1_and_phi_2_all_state = all_state[:, :, (0, 1), :]

    # calculate fidelity
    mapping_state_1_to_state_2_dict = {"00": ["00", "10"], "01": ["00", "10"], "10": ["01", "11"], "11": ["01", "11"]}
    initial_phi_1_phi_2 = all_state[:, 0, (0, 1), :]
    final_phi_1_phi_2   = all_state[:, -1, (0, 1), :]
    fidelity = couple_flux_qubit_metrics.fidelityEvaluation(initial_phi_1_phi_2, final_phi_1_phi_2, mapping_state_1_to_state_2_dict)

    # unmodified_jarzyn = np.mean(np.exp(-cfqr.sim.work_dist_array))

    # work statistic

    # visualization

    gif_save_path = "coupled_flux_qubit_protocol/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".gif"
    gif_save_path = None

    x_range = (-12, 12)
    y_range = (-6, 6)
    manual_domain = [[x_range[0], y_range[0]], [x_range[1], y_range[1]]]
    manual_domain=[[-5, -5], [5, 5]]
    manual_domain = None
    manual_domain=[[-5, -5], [5, 5]]

    vmin, vmax = 0, 100
    phi_1_dc, phi_2_dc = phi_1_dcx, phi_2_dcx
    cfqr.system.protocol_list = protocol_list
    time_range = (computation_protocol_parameter_dict["t"][0], computation_protocol_parameter_dict["t"][-1])

    if frameRate > 0:
        ani,_,_ = visualization.animate_sim_flux_qubit(all_state, system = cfqr.system , times = time_range, frame_skip=frameRate, color_by_state=True, vmin = vmin, vmax = vmax, manual_domain = manual_domain, save_path = gif_save_path, save_dict = computation_protocol_parameter_dict)
    else:
        ani = 0

    return {
        "fidelity": fidelity,
        "work_distribution": work_distribution,
        # "jarzyn_term": unmodified_jarzyn,
        "ani": ani,
        "substepIndex":  protocol_list_item["substepIndex"],
        "sweepKey":  protocol_list_item["substageName"],
        "sweepParameter":  protocol_list_item["sweepParameter"],
        "params": params,
        "initial_parameter_dict": initial_parameter_dict,
        "protocol_list_item": protocol_list_item,
        "comment": comment
    }


def generateSweepProtocolArray(protocol_list_wanted_to_sweep):
    """
    To create a list of protocol with a particular parameter you want to sweep.
    """

    sweep_protocol_list = []
    index = 0
    sweepKey = ""
    sweepArray = None

    for i, substep in enumerate(protocol_list_wanted_to_sweep):
        for key, elem in substep.items():
            if type(elem) is np.ndarray:
                index = i
                sweepKey = key
                sweepArray = elem
                substageName = substep["name"]
                # sweepArray = elem
                # sweepKey = substep["name"]

    for x in sweepArray:
        newProtocolList = copy.deepcopy(protocol_list_wanted_to_sweep)
        newProtocolList[index][sweepKey] = x
        sweep_protocol_list.append({"protocol_list": newProtocolList, "substepIndex": i, "sweepKey": key, "sweepParameter": x, "substageName": substageName})
    return sweep_protocol_list


def showResut(resultArray, itemsWantToShow = None, bins = 10):
    """
    To show the results such as fidelity and work_distribution
    """

    _resultArray = []

    if itemsWantToShow:
        for x in itemsWantToShow:
            _resultArray.append(resultArray[x])
    else:
        _resultArray = resultArray

    for item in _resultArray:
        fidelity = item["fidelity"]
        work_distribution = item["work_distribution"]
        jarzyn_term = np.mean(np.exp(-work_distribution))
        ani = item["ani"]
        substepIndex = item["substepIndex"]
        sweepKey = item["sweepKey"]
        sweepParameter = item["sweepParameter"]

        print(f"substep: {substepIndex}, key: {sweepKey}, value: {sweepParameter}")
        video = ani.to_html5_video()
        html = display.HTML(video)
        display.display(html)

    #     fidelity analysis
        for x in fidelity:
            initialLocation = x["initial"]["location"]
            initialCount = x["initial"]["count"]
            summaryText = f"initial: {initialLocation} ({initialCount}), final: "
            rightLocationCount = sum([y["count"] for y in x["final"]])
            goodRatio = rightLocationCount/initialCount * 100
            for y in x["final"]:
                summaryText += f"{y['location']} ({y['count']}/{rightLocationCount},{y['count']/rightLocationCount * 100: .3g}%),"
            summaryText += f" goodRatio:{goodRatio: .3g}%"
            # print(summaryText)

        plt.hist(work_distribution, bins = bins)
        plt.show()
        unmodified_jarzyn = np.mean(np.exp(work_distribution))
        print(f"jarzyn_term: {jarzyn_term}")
        print("-" * 100)


def getProtocolSubstepName(protocol_list, t):
    """
    Return the name of the substep at time t.
    """
    time_array = [item["duration"] for item in protocol_list]
    name_array = [item["name"] for item in protocol_list]
    cumulative_time_array = list(itertools.accumulate(time_array, operator.add))

    targetIndex = 0

    for i, x in enumerate(cumulative_time_array):
        if i == len(cumulative_time_array) - 1:
            targetIndex = i
            break
        elif i == 0 and t < cumulative_time_array[i]:
            print("case 2")
            targetIndex = i
            break
        else:
            if t >= cumulative_time_array[i] and t <= cumulative_time_array[i+1]:
                targetIndex = i + 1
                break

    print(time_array, cumulative_time_array, name_array[targetIndex])


def saveSimulationResult(simResult, U0_1, timeOrStep = "time", save = False, save_final_state = False, comment = ""):
    """U0_1"""
    #  fidelity
    fidelity = simResult["fidelity"]

    # work_distribution
    work_distribution = simResult["work_distribution"]
    unmodified_jarzyn = float(np.mean(np.exp(-work_distribution)))

    # plt.figure(figsize=(10, 7))
    # plt.text( f"{unmodified_jarzyn: .3g}", horizontalalignment="right",
    #     verticalalignment="top")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(work_distribution, bins = 30)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, f"Jarzyn: {unmodified_jarzyn: .3g}", transform=ax.transAxes, fontsize=14, verticalalignment='top',bbox=props)
    # plt.hist(work_distribution)
    if save:
        np.save(f'coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{simResult["simulation_data"]["simulation_id"]}_work_distribution.npy', work_distribution)
        plt.savefig(f'coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{simResult["simulation_data"]["simulation_id"]}_work_distribution.png')
        pass

    # work statistics
    work_statistic = simResult["work_statistic"]
    work_mean, work_std = work_statistic[:, 0], work_statistic[:, 1]

    skip_step = int(len(work_mean) * 0.01)
    step_array = np.arange(0, work_mean.shape[0])
    if timeOrStep == "time":
        step_array *= np.array(simResult["cfqr"].sim.dt)
    # plt.plot(step_array, work_mean)
    plt.figure(figsize=(10, 7))
    plt.errorbar(step_array[::skip_step], work_mean[::skip_step], yerr = work_std[::skip_step])
    # print(step_array)
    substep_array = np.cumsum([substep["duration"]/simResult["cfqr"].sim.dt for substep in simResult["protocol_list_item"]])
    for _t in substep_array[:-1]:
        plt.vlines(x=_t, ymin = np.min(work_mean), ymax = np.max(work_mean), ls="--", colors = "purple")

    if save:
        np.save(f'coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{simResult["simulation_data"]["simulation_id"]}_work_statistic.npy', work_statistic)
        plt.savefig(f'coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{simResult["simulation_data"]["simulation_id"]}_work_statistic.png')
        pass


    if save_final_state:
        final_state = simResult["cfqr"].sim.output.final_state
        np.save(f'coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{simResult["simulation_data"]["simulation_id"]}_final_state.npy', final_state)

        plt.figure(figsize=(10, 7))
        plt.scatter(final_state[:, 0, 0], final_state[:, 1, 0])
        plt.savefig(f'coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{simResult["simulation_data"]["simulation_id"]}_final_state.png')


    FFwriter = animation.FFMpegWriter(fps=10)
    if save:
        simResult["ani"].save(f'coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/{simResult["simulation_data"]["simulation_id"]}_szilard_engine.mp4', writer = FFwriter)

    simResult["simulation_data"]["simulation_computer"] = socket.gethostname()
    simResult["simulation_data"]["saveTime"] = str(datetime.datetime.timestamp( datetime.datetime.now()))


    saveData = {
        "params":                              simResult["params"],
        "initial_parameter_dict":      simResult["initial_parameter_dict"],
        "protocol_list_item":            simResult["protocol_list_item"],
        "simulation_data":                simResult["simulation_data"],
        "jarzynski_term":                 unmodified_jarzyn,
        "fidelity":                             simResult["fidelity"],
        "comment":                             simResult["comment"]
    }

    saveData["params"]["sim_params"] = [list(item) for item in simResult["params"]["sim_params"]]

    with open("coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/gallery.json") as f:
        jsonData = json.load(f)
        jsonData.append(saveData)
        with open("coupled_flux_qubit_protocol/coupled_flux_qubit_data_gallery/gallery.json", "w+") as fw:
            json.dump(jsonData, fw, cls = NumpyArrayEncoder)
