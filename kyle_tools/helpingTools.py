import json, os
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
# from visualization import animate_sim


source_path = os.path.expanduser('~/Project/source/')
sys.path.append(source_path)
from quick_sim import setup_sim
import kyle_tools as kt
import info_space

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))




colorCode = {"00": "blue", "01": "orange", "10": "green", "11": "red"}

def ratio_of_good_points(keys, _dict, all_state):
    _returnDict = {

    }

    initial_final_phase_space_dict = get_initial_and_final_xy_coord(all_state[:,:,:,0],
                                    frame_skip=10, color_by_state=True)

    for _k in keys:
        total = len(_dict[_k]["final"][1])
        number_of_good_points = 0

        goodPointsIndex = []

        if _k in ["00", "01"]:
            goodPointsIndex = np.where(initial_final_phase_space_dict[_k]["final"][1] < 0)[0]
            number_of_good_points = len(goodPointsIndex)
            print(_k, "< 0")

        elif  _k in ["10", "11"]:
            goodPointsIndex = np.where(initial_final_phase_space_dict[_k]["final"][1] > 0)[0]
            number_of_good_points = len(goodPointsIndex)
            print(_k, "> 0")

        goodPoints_x_coordinates = initial_final_phase_space_dict[_k]["final"][0][goodPointsIndex]



        print("number of good points: ", number_of_good_points)
        _returnDict[_k] = {
            "goodRatio": number_of_good_points/total * 100,
            "mean_x": np.mean(goodPoints_x_coordinates),
            "var_x": np.var(goodPoints_x_coordinates)
        }
        print(f"key: {_k} ({colorCode[_k]}) = {number_of_good_points/total * 100}")
        print(f"mean: {np.mean(goodPoints_x_coordinates)}, var: {np.var(goodPoints_x_coordinates)}")
        print("=" * 100)
    return _returnDict


def get_initial_and_final_xy_coord(all_state, times=[0,1], system=None, frame_skip=30, which_axes=None, axes_names=None, color_by_state=None, key_state=None, color_key=None, legend=True, alpha=None, fig_ax=None, **pot_kwargs):

    if color_by_state is not None:
        if key_state is not None:
            state_lookup = info_space.separate_by_state(key_state)
        else:
            state_lookup = info_space.separate_by_state(all_state[:, 0, ...])

    N, nsteps, N_dim = np.shape(all_state)[0], np.shape(all_state)[1], np.shape(all_state)[2]

    for i in range(N_dim):
        if np.size(np.shape(all_state)) == 4:
            which_axes = [np.s_[..., i, 0] for i in range(N_dim)]
        if np.size(np.shape(all_state)) == 3:
            which_axes = [np.s_[..., i] for i in range(N_dim)]

    x_array = [all_state[item] for item in which_axes]

    samples = np.linspace(0, nsteps-1, nsteps)[::frame_skip]
    time = np.linspace(times[0], times[1], nsteps + 1)


    x = x_array[0]
    y = x_array[1]

    initial_final_phase_space_dict = {}

    for key in ["00", "01", "10", "11"]:
        initial_final_phase_space_dict[key] = {
            "initial": (x[state_lookup[key], 0], y[state_lookup[key], 0]),
            "final": (x[state_lookup[key], int(samples[-1])], y[state_lookup[key], int(samples[-1])]),
        }

    return initial_final_phase_space_dict


def time_stretch_simulation(_time_stretch, _damping, _sim_parameters, system, init_state, procedures = None, speedyRun = False):
    dt = .005 if speedyRun is True else .0005
    system.protocol.normalize()
    system.protocol.time_stretch(_time_stretch)
    sim = setup_sim(system, init_state, procedures, sim_params=_sim_parameters, dt=dt, damping=_damping)
    sim.output = sim.run(verbose=True)
    return sim

def runSimulationFunction(parameterList: [float, float, [float]], system, init_state, procedures = None, label = "",   overwrite = False, save = False, showAnimation = False, speedyRun = False)->None:
    """
        a function that can run simulations with parameter input
        - label (string): give a label to the simulation result. It is useful when you do sweep.
        - overwrite (bool): If true, then overwrite the existing data
        - save(bool): If true, then save the simulation statistics results into json
        - showAnimation (bool): If true, then show the animation after simulation

    """
    _time_stretch, _damping, _sim_parameters = parameterList


    sim = time_stretch_simulation(_time_stretch, _damping, _sim_parameters, system, init_state, procedures, speedyRun)
    all_state = sim.output.all_state['states']
    # print(sim.output.all_statistic_with_time)
    time_dependent_statistic =  sim.output.all_statistic_with_time

    # to massage the statistics output so that they can be converted into json

    time_dependent_statistic.pop('trial_indices', None)
    time_dependent_statistic["step_indices"] = list(time_dependent_statistic["step_indices"]) # convert the slice object into a list
    time_dependent_statistic["values"] = np.round(time_dependent_statistic["values"], 4).tolist() # convert a numpy object into a list with 3 dp

    initial_final_phase_space_dict = get_initial_and_final_xy_coord(all_state[:,:,:,0],
                                        frame_skip=10, color_by_state=True)

    # to create a result dict and put into json for record
    # statistics: this data is for the final state only
    # time_dependent_statistic: this monitor the changes of the statistics with time. state only
    result_dict = {
        "label": label,
        "date": str(datetime.now()),
        "ts": _time_stretch,
        "damping": _damping,
        "gamma": _sim_parameters[0],
        "theta": _sim_parameters[1],
        "eta": _sim_parameters[2],
        "statistics": ratio_of_good_points(["00", "01", "10", "11"], initial_final_phase_space_dict, all_state),
        "time_dependent_statistic": time_dependent_statistic
    }



    ani,_,_ = kt.animate_sim(all_state[:,:,:,0], frame_skip=10, color_by_state=True)
    parameter_list = ','.join([str(x) for x in _sim_parameters])
    filename = f"parameter/animation_ts_{_time_stretch}_damping_{_damping}_parameter_{parameter_list}_times.gif"


    # writing the things into the json file

    if not os.path.isfile(filename) or overwrite:
        if save:
            with open("parameter/result.json", "r") as rf:
                data = json.load(rf)
                data.append(result_dict)

                with open("parameter/result.json", "w") as wf:
                    json.dump(data, wf)
                    # try:
                    #     json.dumps(data)

                    # except:
                    #     print(data)
                    #     print("json output error")
                    # # dumped = json.dumps(data)
            print(filename, "is saved.")
            ani.save(filename,writer='imagemagick', fps=60)
        print("You are not saving data")
    else:
        print("The set of parameters has already been simulated.")

    if showAnimation:
        ani,_,_ = kt.animate_sim(all_state[:,:,:,0], frame_skip=10, color_by_state=True)
        return ani

def sweepFunction(sweepObject, system, init_state, label, procedures = None, save = True, overwrite = False, notification = True, speedyRun = False):
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
        if type(val) is list:
            sweepParameter = key
            sweepList = val.copy()

    # genereate sets of parameter lists to do simulations
    for index, x in enumerate(sweepList):
        _sweepObject[sweepParameter] = x
        _time_stretch = _sweepObject['_time_stretch']
        _damping = _sweepObject['_damping']
        _gamma = _sweepObject['_gamma']
        _theta = _sweepObject['_theta']
        _eta = _sweepObject['_eta']

        print("=" * 10 + f" start ({index+1}/{len(sweepList)}) " + "=" * 10)
        print(f"time_stretch: {_time_stretch}, _damping: {_damping}, _gamma: {_gamma}, " +
        f"_theta: {_theta}, _eta: {_eta}")
        print(procedures)
        runSimulationFunction([_time_stretch, _damping, [_gamma, _theta, _eta]], system, init_state, procedures = procedures,  label = label, save = save, overwrite = overwrite, speedyRun = speedyRun)
        print("=" * 10 + f" finished ({index+1}/{len(sweepList)}) " + "=" * 10)

    # if notification == True:
    #     from notifypy import Notify
    #     notification = Notify()
    #     notification.title = "Finish Simulation"
    #     notification.message = "The simulation is finished."
    #     notification.send()
    #



################## measure.py #################
from measure import binary_partition, get_default_names, get_default_values

def position_classification(position_data):
    """
    This function is used to classification of the categories of the data points belongs to.
    categories examples: [(0, 0), (0, 1), (1, 0), (1, 1)]


    problem: this is just for 2D, need to be generalized to 1D and 3D too

    """
    initial_position = position_data
    print(position_data, position_data.shape)
    dim = position_data.shape[1]
    ntrials = position_data.shape[0]

    # category_list =  [(0, 0), (0, 1), (1, 0), (1, 1)]
    #  category_label = ["00", "01", "10", "11"]
    category_list = get_default_values(dim)
    category_label = get_default_names(dim)

    # for index, category in enumerate(category_list):
    #     # fill up the bool_list
    #     bool_list[category_label[index]] = np.all(measure_data == category, axis = 1)
    bool_list = {"00": [], "01": [], "10": [], "11": []}

    for index, category in enumerate(category_list):
        # fill up the bool_list
        measure_data = binary_partition(position_data[:, :, 0]) == category
        # reshape the data from (ntrial, 1, 2) -> (ntrial, 2)
        reshaped_measure_data = measure_data.reshape(ntrials, 2)

        index_column = [np.logical_and.reduce(x) for x in reshaped_measure_data]
        bool_list[category_label[index]] = index_column

    return bool_list

def plotColoredMapping(initial_state, position_categories_index_dict, alpha = 0.2):
    plt.scatter(*initial_state[position_categories_index_dict["00"]].T, color = colorCode["00"], alpha = alpha)
    plt.scatter(*initial_state[position_categories_index_dict["01"]].T, color = colorCode["01"], alpha = alpha)
    plt.scatter(*initial_state[position_categories_index_dict["10"]].T, color = colorCode["10"], alpha = alpha)
    plt.scatter(*initial_state[position_categories_index_dict["11"]].T, color = colorCode["11"], alpha = alpha)
    plt.show()


def plotStatisticsChangingWithTime(step_indices, overall_statistics_result, parameterDict = None):
    fig, ax = plt.subplots(1, 4, figsize=(18,4))

    if parameterDict is not None:
        print(parameterDict)

    for key in colorCode:
        index = list(colorCode.keys()).index(key)

        ax[0].plot(step_indices, overall_statistics_result[:, index, 0],\
                   label=f"{key}", color = colorCode[key])
        ax[0].title.set_text(f'mean x vs t')
        ax[0].legend()

        ax[1].plot(step_indices, overall_statistics_result[:, index, 1],\
                  label=f"{key}", color = colorCode[key])
        ax[1].title.set_text(f'var_x vs t')
        ax[1].legend()

        ax[2].plot(step_indices, overall_statistics_result[:, index, 2],\
                  label=f"{key}", color = colorCode[key])
        ax[2].title.set_text(f'mean y vs t')
        ax[2].legend()

        ax[3].plot(step_indices, overall_statistics_result[:, index, 3],\
                   label=f"{key}", color = colorCode[key])
        ax[3].title.set_text(f'var y vs t')
        ax[3].legend()
    plt.show()


def analysisFunction(label, parameterName):
    with open("parameter/result.json", "r") as rf:
        data = json.load(rf)
        filteredData = [x for x in data if x["label"] == _label]

        statisticObject = {}

        gamma_list = []
        statisticsDict = {
            "00": {"goodRatio": [], "mean_x": [], "var_x": []},
            "01": {"goodRatio": [], "mean_x": [], "var_x": []},
            "10": {"goodRatio": [], "mean_x": [], "var_x": []},
            "11": {"goodRatio": [], "mean_x": [], "var_x": []}
        }

        for _item in filteredData:
            _item_statistics = _item["statistics"]
            gamma_list.append(_item[parameterName])

            for _key in KEY:
                statisticsDict[_key]["goodRatio"].append(_item_statistics[_key]["goodRatio"])
                statisticsDict[_key]["mean_x"].append(_item_statistics[_key]["mean_x"])
                statisticsDict[_key]["var_x"].append(_item_statistics[_key]["var_x"])

        fig, ax = plt.subplots(1, 3, figsize=(18,4))
        fixedParameterList = [f"{x}: {filteredData[0][x]}" for x in parameterList if parameterName != x]
        print(fixedParameterList)


        for key in KEY:
    #         fig.tight_layout()
            #create subplots
            ax[0].plot(gamma_list, statisticsDict[key]["goodRatio"], label=f"{key}", color = colorMapping[key])
            ax[0].title.set_text(f'goodRatio vs {_label}')
            ax[0].legend()

            ax[1].plot(gamma_list, statisticsDict[key]["mean_x"], label=f"{key}", color = colorMapping[key])
            ax[1].title.set_text(f'mean_x vs {_label}')
            ax[1].legend()

            ax[2].plot(gamma_list, statisticsDict[key]["var_x"], label=f"{key}", color = colorMapping[key])
            ax[2].set_ylim(0, 3)
            ax[2].title.set_text(f'var_x vs {_label}')
            ax[2].legend()
        plt.show()

def get_statistics_for_all_categories(simulation, time, trial_request=np.s_[:]):
    """
    This function returns the mean and var for x and y coordinates at different steps

    return: an array of (x_mean, x_var, y_mean, y_var) for each categories (e.g. in 2d, it is 00, 01, 10, 11)
    """
    # state = simulation.current_state[trial_request]
    time = simulation.current_time
    state = simulation.current_state
    trials_number = trial_request # if trial_request exists, use it instead of the whole trials number

    # it is an array of [x_mean, x_var, y_mean, y_var]
    result_array = np.empty([4, 4])

    if time == 0:
        # to get the index for the categories of the initial positions
        simulation.position_categories_index_dict = position_classification(state)

        # simulation.position_categories_index_dict = helpingTools.position_classification(state)
        # print(simulation.position_categories_index_dict)

    index = 0
    for label, val in simulation.position_categories_index_dict.items():
        # label = 00, 01, 10, 11
        # val = [False, True, ...] index column to indicate whether the positions are 00, 01, 10, 11
        # plt.scatter(*state[val].T)
        x_coordinates = state[val][:, 0, 0]
        y_coordinates = state[val][:, 1, 0]

        statistics_result = np.array([np.mean(x_coordinates), np.var(x_coordinates), np.mean(y_coordinates), np.var(y_coordinates)])
        result_array[index:] = statistics_result
        index += 1

    return result_array
