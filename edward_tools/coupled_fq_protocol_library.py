import numpy as np
import sys
import os
source_path = os.path.expanduser('~/Project/source/')
sys.path.append(source_path)
sys.path.append(os.path.expanduser('~/Project/source/simtools/'))

from .fq_potential import fq_pot, fq_default_param
from sus.protocol_designer import System, Protocol, Potential, Compound_Protocol
import matplotlib.pyplot as plt

protocol_key = ['U0_1', 'U0_2', 'gamma_1', 'gamma_2', 'beta_1', 'beta_2', 'd_beta_1', 'd_beta_2', 'phi_1_x', 'phi_2_x', 'phi_1_dcx', 'phi_2_dcx', 'M_12']

def create_simple_protocol_parameter_dict(protocol_array):
    """
    simple means that the protocol doesn't have substeps.
    """
    result_dict = {}
    for i, k in enumerate(protocol_key):
        result_dict[k] = [protocol_array[i], protocol_array[i]]
    result_dict["t"] = [0, 1]
    return result_dict

def create_system(protocol_parameter_dict, domain = None):
    """
    This function is used to produce the storage and computation protocol

    input:
    1. comp_protocol_parameter_dict:
    - a dictionary contains the an array of time, which represents the time point at which the protocol is changed
    - the key is the name of the parameter
    - for parameters, they are arrays containing the value of the parameter at the particular time point

    output:
    1. comp_prototocl
    - the protocol for the computation system
    """
    # storage protocol, just take the first and last element of each term in the protocol key from the dict to form the storage protocol
    storage_t = (protocol_parameter_dict["t"][0], protocol_parameter_dict["t"][-1])
    storage_protocol_parameter_time_series = [np.array([protocol_parameter_dict[key][0], protocol_parameter_dict[key][-1]]) for key in protocol_key]
    storage_protocol_parameter_time_series = np.array(storage_protocol_parameter_time_series)
    storage_protocol = Protocol(storage_t, storage_protocol_parameter_time_series)

    # computation protocol, this part form the time series of the comp_protocol and join them to form Compound_Protocol
    comp_protocol_array = []
    comp_t = protocol_parameter_dict["t"]
    comp_protocol_parameter_time_series = [protocol_parameter_dict[key] for key in protocol_key]
    comp_protocol_parameter_time_series = np.array(comp_protocol_parameter_time_series).T

    for i in range(len(comp_t)-1):
        n_th_comp_time_array = (comp_t[i], comp_t[i+1])
        n_th_comp_protocol_parameter_array = np.array([comp_protocol_parameter_time_series[i], comp_protocol_parameter_time_series[i+1]]).T # in the form of array of [(p_n_i, p_n_f)]
        _p = Protocol(n_th_comp_time_array, n_th_comp_protocol_parameter_array)
        comp_protocol_array.append(_p)
    comp_protocol = Compound_Protocol(comp_protocol_array)

    return storage_protocol, comp_protocol


def create_system_from_storage_and_computation_protocol(storage_protocol_parameter_dict = None, comp_protocol_parameter_dict = None, domain = None):
    """
    This function is used to produce the storage and computation protocol

    input:
    1. input_parameters_dict:
    - a dictionary contains the an array of time, which represents the time point at which the protocol is changed
    - the key is the name of the parameter
    - for parameters, they are arrays containing the value of the parameter at the particular time point

    output:
    1. storage_protocol:
    - the protocol for the equilibrium system

    2. comp_prototocl
    - the protocol for the computation system
    """
    if comp_protocol_parameter_dict == None:
        print("please give me comp_protocol")


    if storage_protocol_parameter_dict is not None:
        print("storage_protocol_parameter_dict is not None")

    # storage protocol
    storage_t = storage_protocol_parameter_dict["t"]
    storage_protocol_parameter_time_series = [storage_protocol_parameter_dict[key] for key in protocol_key]
    storage_protocol_parameter_time_series = np.array(storage_protocol_parameter_time_series)
    storage_protocol = Protocol(storage_t, storage_protocol_parameter_time_series)

    # computation protocol
    comp_protocol_array = []
    comp_t = comp_protocol_parameter_dict["t"]
    comp_protocol_parameter_time_series = [comp_protocol_parameter_dict[key] for key in protocol_key]
    comp_protocol_parameter_time_series = np.array(comp_protocol_parameter_time_series).T

    for i in range(len(comp_t)-1):
        n_th_comp_time_array = (comp_t[i], comp_t[i+1])
        n_th_comp_protocol_parameter_array = np.array([comp_protocol_parameter_time_series[i], comp_protocol_parameter_time_series[i+1]]).T # in the form of array of [(p_n_i, p_n_f)]
        _p = Protocol(n_th_comp_time_array, n_th_comp_protocol_parameter_array)
        comp_protocol_array.append(_p)
    comp_protocol = Compound_Protocol(comp_protocol_array)

    return storage_protocol, comp_protocol

def customizedProtocol(initial_values_dict, protocol_list, normalized = False):
    protocol_key_array = ['U0_1', 'U0_2', 'gamma_1', 'gamma_2', 'beta_1', 'beta_2', 'd_beta_1', \
                    'd_beta_2', 'phi_1_x', 'phi_2_x', 'phi_1_dcx', 'phi_2_dcx', 'M_12']

    protocol_parameter_dict = {key: [value] for key, value in initial_values_dict.items()}
    protocol_parameter_dict["t"] = [0]


    for item in protocol_list:
        # add the duration to the time entry of the protocol_parameter_dict
        protocol_parameter_dict["t"].append(protocol_parameter_dict["t"][-1] + item["duration"])

        for key in protocol_key_array:
            if key in item.keys(): # to check which key is present in the protocol_list_item.
                protocol_parameter_dict[key].append(item[key])
            else:
                protocol_parameter_dict[key].append(protocol_parameter_dict[key][-1])

    if normalized:
        protocol_parameter_dict["t"] = np.array(protocol_parameter_dict["t"])/ np.max(protocol_parameter_dict["t"])

    return protocol_parameter_dict


# def sliceThroughPhi_dc(simRunner, time = None, axis1 = 0, axis2 = 1, contours=10, resolution = 200, manual_domain=None, slice_values = None, surface = False, cbar=False, numberOfColumns = 3, vmin = None, vmax = None, steps = None):
#



def get_potential_shot_at_different_t(simRunner, protocol_parameter_dict, timeStep = None, axis1 = 0, axis2 = 1, contours=10, resolution = 200, manual_domain=None, slice_values = None, surface = False, cbar=False, numberOfColumns = 3, vmin = None, vmax = None):
    # print(protocol_parameter_dict)
    # to figure out which parameter has changed, and which have not been changed.
    changing_parameter_key = [key for key, value in protocol_parameter_dict.items() \
                            if len(set(value)) != 1]

    # to create the title of each subplot

    if timeStep:
        timeSeries = np.arange(protocol_parameter_dict["t"][0], protocol_parameter_dict["t"][-1] + timeStep, timeStep)
        # timeSeries = np.arange(0, 1 + timeStep, timeStep)
        changing_parameter_dict = {}


        for key in changing_parameter_key:
            if key != "t":
                keyIndex = protocol_key.index(key)
                changing_parameter_dict[key] = [simRunner.system.protocol.get_params(t)[keyIndex] for t in timeSeries]

    else:
        timeSeries = protocol_parameter_dict["t"]
        changing_parameter_dict = {key: protocol_parameter_dict[key] for key in changing_parameter_key}

    # print(changing_parameter_dict)
    # create the subplot_title
    subplot_title_array = []
    for key, value in changing_parameter_dict.items():
        array = [f"{key}: {v:.3g}" for v in value]
        subplot_title_array.append(array)
    subplot_title_array = list(zip(*subplot_title_array))

    # to create the subgraph with correct number of rows
    numberOfPlots = len(timeSeries)
    offset = 0 if numberOfPlots % numberOfColumns == 0 else 1
    numberOfRows = numberOfPlots // numberOfColumns + offset

    fig, ax = plt.subplots(numberOfRows, numberOfColumns, figsize=(18,4 * numberOfRows))


    def drawParameterGraphs(fig, ax, vmin, vmax):
        # vmin, vmax = 0, 0
        modified_manual_domain = [(manual_domain[0][1], manual_domain[0][0]), (manual_domain[1][1], manual_domain[1][0])]
        for i, t in enumerate(timeSeries):
            row = i // numberOfColumns
            column = i % numberOfColumns
            phi_1_dcx_index = protocol_key.index('phi_1_dcx')
            phi_2_dcx_index = protocol_key.index('phi_2_dcx')
            phi_1_dc_i = simRunner.system.protocol.get_params(t)[phi_1_dcx_index]
            phi_2_dc_i = simRunner.system.protocol.get_params(t)[phi_2_dcx_index]
            slice_values = [0, 0, phi_1_dc_i, phi_2_dc_i]

            U, X_mesh = simRunner.system.lattice(t, resolution, axes=(0, 1), manual_domain=modified_manual_domain, slice_values = slice_values)

            if (i==0) and not vmin and not vmax:
                vmin = np.min(U)
                vmax = np.max(U)
            # U, X_mesh = simRunner.system.lattice(t, resolution, axes=(axis1, axis2), manual_domain=manual_domain, slice_values = slice_values)
            X = X_mesh[0]
            Y = X_mesh[1]
            x_min, x_max = np.min(X), np.max(X)
            y_min, y_max = np.min(Y), np.max(Y)

            if surface is False:
                # subplot = fig.add_subplot(row+1, numberOfColumns, column + 1)
                # out = subplot.contourf(X, Y, U, contours)
                if cbar:
                    plt.colorbar(out)

                # This part is to prevent index error
                if len(timeSeries) > numberOfColumns: # when the number of graph is more than one row
                    subplot = ax[row][column]
                elif len(timeSeries) <= numberOfColumns and len(timeSeries) > 1: # when the number of graph is just one row
                    subplot = ax[column]
                elif len(timeSeries) == 1: # when the number of graph is 1
                    subplot = ax
                subplot.set_aspect(1)

                if len(subplot_title_array) > 0:
                    subplot.set_title(f"t = {t:.3g}, " + ", ".join(subplot_title_array[i]))
                else:
                    subplot.set_title(f"t = {t:.3g}")
                out = subplot.contourf(X, Y, U, contours, vmin = vmin, vmax = vmax)

                # cfqr.system.protocol.get_params(0)
            if surface is True:

                ax = fig.add_subplot(row+1, numberOfColumns, column + 1, projection='3d')
                surf = ax.plot_surface(X, Y, U)
                # ax.set_title(", ".join(subplot_title_array[i]))

    drawParameterGraphs(fig, ax, vmin, vmax)
    plt.show()


def get_potential_shot_at_different_t_1D(simRunner, protocol_parameter_dict, timeStep = None, axis1 = 0, axis2 = 1, targetAxis = 0, cutlineDirection = "v", cutlineValue = 0, contours=10, resolution = 200, manual_domain=None, slice_values = None, surface = False, cbar=False, numberOfColumns = 3, vmin = None, vmax = None):
    # print(protocol_parameter_dict)
    # to figure out which parameter has changed, and which have not been changed.
    changing_parameter_key = [key for key, value in protocol_parameter_dict.items() \
                            if len(set(value)) != 1]

    # to create the title of each subplot
    plotResultArray = []

    if timeStep:
        timeSeries = np.arange(protocol_parameter_dict["t"][0], protocol_parameter_dict["t"][-1] + timeStep, timeStep)
        # timeSeries = np.arange(0, 1 + timeStep, timeStep)
        changing_parameter_dict = {}


        for key in changing_parameter_key:
            if key != "t":
                keyIndex = protocol_key.index(key)
                changing_parameter_dict[key] = [simRunner.system.protocol.get_params(t)[keyIndex] for t in timeSeries]

    else:
        timeSeries = protocol_parameter_dict["t"]
        changing_parameter_dict = {key: protocol_parameter_dict[key] for key in changing_parameter_key}

    # print(changing_parameter_dict)
    # create the subplot_title
    subplot_title_array = []
    for key, value in changing_parameter_dict.items():
        array = [f"{key}: {v:.3g}" for v in value]
        subplot_title_array.append(array)
    subplot_title_array = list(zip(*subplot_title_array))

    # to create the subgraph with correct number of rows
    numberOfPlots = len(timeSeries * 2)
    offset = 0 if numberOfPlots % numberOfColumns == 0 else 2
    numberOfRows = numberOfPlots // numberOfColumns + offset

    fig, ax = plt.subplots(numberOfRows, numberOfColumns, figsize=(18,4 * numberOfRows))


    def drawParameterGraphs(fig, ax, vmin, vmax):
        # vmin, vmax = 0, 0
        modified_manual_domain = [(manual_domain[0][1], manual_domain[0][0]), (manual_domain[1][1], manual_domain[1][0])]

        for i, t in enumerate(timeSeries):
            contour_row = 2 * (i // numberOfColumns)
            cutline_row = contour_row + 1
            column = i % numberOfColumns
            phi_1_dcx_index = protocol_key.index('phi_1_dcx')
            phi_2_dcx_index = protocol_key.index('phi_2_dcx')
            phi_1_dc_i = simRunner.system.protocol.get_params(t)[phi_1_dcx_index]
            phi_2_dc_i = simRunner.system.protocol.get_params(t)[phi_2_dcx_index]
            slice_values = [0, 0, phi_1_dc_i, phi_2_dc_i]

            U, X_mesh = simRunner.system.lattice(t, resolution, axes=(0, 1), manual_domain=modified_manual_domain, slice_values = slice_values)

            if (i==0) and not vmin and not vmax:
                vmin = np.min(U)
                vmax = np.max(U)
            # U, X_mesh = simRunner.system.lattice(t, resolution, axes=(axis1, axis2), manual_domain=manual_domain, slice_values = slice_values)
            X = X_mesh[0]
            Y = X_mesh[1]
            x_min, x_max = np.min(X), np.max(X)
            y_min, y_max = np.min(Y), np.max(Y)


            if surface is False:
                # This part is to prevent index error
                if len(timeSeries) > numberOfColumns: # when the number of graph is more than one row
                    contour_subplot = ax[contour_row][column]
                    cutline_subplot = ax[cutline_row][column]
                elif len(timeSeries) <= numberOfColumns and len(timeSeries) > 1: # when the number of graph is just one row
                    contour_subplot = ax[contour_row][column]
                    cutline_subplot = ax[cutline_row][column]

                if len(subplot_title_array) > 0:
                    contour_subplot.set_title(f"t = {t:.3g}" + ", ".join(subplot_title_array[i]))
                else:
                    contour_subplot.set_title(f"t = {t:.3g}")

                # subplot.set_aspect(1)
                plotResult = plotCutlines(X, Y, U, cutlineDirection = cutlineDirection, cutlineValue = cutlineValue, contour_plt = contour_subplot, cutline_plt = cutline_subplot, contours = contours, time = t)
                plotResult["parameters"] = subplot_title_array
                plotResultArray.append(plotResult)
                # subplot.show()
                # plt.scatter(plotAxis, targetU[0])
                # if cutlineDirection == "v":
                #     targetAxis = Y
                #     targetIndex = np.sum(np.mean(targetAxis, axis=1) < cutline) - 1
                #     plotAxis = X[targetIndex, :]
                # if cutlineDirection == "h":
                #     targetAxis = X
                #     targetIndex = np.sum(np.mean(targetAxis, axis=1) < cutline) - 1
                #     plotAxis = Y[:, targetIndex]
                # # print(X[targetIndex, :])
                # # out1 = subplot.contourf(X, Y, U, contours, vmin = vmin, vmax = vmax)
                #
                #
                # if cutlineDirection == "v":
                #     subplot.axvline(x=cutline, ymin=np.min(plotAxis), ymax=np.max(plotAxis))
                # if cutlineDirection == "h":
                #     subplot.axhline(y=cutline, xmin=np.min(plotAxis), xmax=np.max(plotAxis))
                #
                # # targetIndex = np.sum(np.mean(targetAxis, axis=1) < cutline) - 1
                # # subplot.set_aspect(0)
                # out2 = subplot.plot(plotAxis, U[targetIndex, :])

                # cfqr.system.protocol.get_params(0)

    drawParameterGraphs(fig, ax, vmin, vmax)
    plt.show()
    return plotResultArray

def plotCutlines(X, Y, U, cutlineDirection, cutlineValue, contour_plt = plt, cutline_plt = plt, contours = 5, time = None):
    if cutlineDirection == "h":
        _plotAxis = X
        _targetAxis = Y
        _plotU = U

    if cutlineDirection == "v":
        _plotAxis = Y.T
        _targetAxis = X.T
        _plotU = U.T

    plotAxis = _plotAxis[0]

    targetAxis = np.mean(_targetAxis, axis = 1)
    # to find out the resolution of the target axis
    targetRange = (targetAxis[-1] - targetAxis[-2])/2
    targetIndex = np.where(np.abs(targetAxis - cutlineValue) <= targetRange)[0][0]
    targetU = _plotU[targetIndex]
    # print(targetAxis, targetIndex, targetU)
    cont = contour_plt.contourf(X, Y, U,  contours)

    if cutlineDirection == "h":
        contour_plt.hlines(y = _targetAxis[targetIndex], xmin = np.min(_plotAxis), xmax = np.max(_plotAxis), colors= "red")
    if cutlineDirection == "v":
        contour_plt.vlines(x = _targetAxis[targetIndex], ymin = np.min(_plotAxis), ymax = np.max(_plotAxis), colors= "red")
    # _plt.show()

    cutline_plt.scatter(plotAxis, targetU)
    return {
        "contour_plot": {"X": X, "Y": Y, "U": U, "contours": contours, "time": time},
        "cutline_plot": {"plotAxis": plotAxis, "targetU": targetU, "time": time, "cutlineDirection": cutlineDirection, "cutlineValue": cutlineValue}
    }


def eq_state(self, Nsample, t=None, resolution=500, beta=1, manual_domain=None, axes=None, slice_vals=None, verbose=True):
        resolution = 80
        NT = Nsample
        state = np.zeros((max(100, int(2*NT)), self.potential.N_dim, 2))

        def get_prob(self, state):
            E_curr = self.get_energy(state, t)
            Delta_U = E_curr-U0
            return np.exp(-beta * Delta_U)

        if t is None:
            t = self.protocol.t_i

        U, X = self.lattice(t, resolution, axes=axes, slice_values=slice_vals, manual_domain=manual_domain)

        mins = []
        maxes = []
        # for item in X:
        #     mins.append(np.min(item))
        #     maxes.append(np.max(item))
        #
        # U0 = np.min(U)
        # i = 0

#         if axes is None:
#             axes = [_ for _ in range(1, self.potential.N_dim + 1)]
#         axes = np.array(axes) - 1

#         count = 0
#         while i < Nsample:
#             count += 1
#             test_coords = np.zeros((NT, self.potential.N_dim, 2))
#             if slice_vals is not None:
#                 test_state[:, :, 0] = slice_vals

#             test_coords[:, axes, 0] = np.random.uniform(mins, maxes, (NT, len(axes)))

#             p = get_prob(self, test_coords)

#             decide = np.random.uniform(0, 1, NT)
#             n_sucesses = np.sum(p > decide)
#             if i == 0:
#                 ratio = max(n_sucesses/NT, .05)

#             state[i:i+n_sucesses, :, :] = test_coords[p > decide, :, :]
#             i = i + n_sucesses
#             if verbose:
#                 print("\r found {} samples out of {}".format(i, Nsample), end="")

#             NT = max(int((Nsample-i)/ratio), 100)

#         print("from system: finish the while loop.")
#         state = state[0:Nsample, :, :]
#         # print("the state is", state)

#         if self.has_velocity:
#             state[:, :, 1] = np.random.normal(0, np.sqrt(1/(self.mass*beta)), (Nsample, self.potential.N_dim))
#         else:
#             return state[:, :, 0]

#         return state
# cutlineValue, contour_plt = contour_subplot, cutline_plt = cutline_subplot)
