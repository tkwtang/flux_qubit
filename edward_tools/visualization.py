import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numba import njit

@njit
def binary_partition(positions, boundary=0):
    '''
    takes a set of position coordinates and sets each value to either 0 or 1 depending on if it is below or above the boundary
    '''
    return (np.sign(positions-boundary)+1)/2


def separate_by_state(state, **kwargs):
    initial_state = state[:, 0, ...]

    bool_array_00 = binary_partition(initial_state[:, :, 0]) == np.array([0., 0.])
    # print(bool_array_00)
    index_of_00 = np.all(bool_array_00, axis=1)

    bool_array_01 = binary_partition(initial_state[:, :, 0]) == np.array([0., 1.])
    # print(bool_array_01)
    index_of_01 = np.all(bool_array_01, axis=1)

    bool_array_10 = binary_partition(initial_state[:, :, 0]) == np.array([1., 0.])
    # print(bool_array_10)
    index_of_10 = np.all(bool_array_10, axis=1)

    bool_array_11 = binary_partition(initial_state[:, :, 0]) == np.array([1., 1.])
    # print(bool_array_11)
    index_of_11 = np.all(bool_array_11, axis=1)

    return {
        "00": index_of_00,
        "01": index_of_01,
        "10": index_of_10,
        "11": index_of_11,
    }

def animate_sim_flux_qubit(all_state, times=[0,1], system=None, frame_skip=30, which_axes=None, axes_names=None, \
                color_by_state=None, key_state=None, color_key=None, legend=True, alpha=None, fig_ax=None, \
                **pot_kwargs):

    N, nsteps, N_dim = np.shape(all_state)[0], np.shape(all_state)[1], np.shape(all_state)[2]
    which_axes = [np.s_[..., i, 0] for i in range(N_dim)]
    x_array = [all_state[item] for item in which_axes]
    state_lookup = separate_by_state(all_state)

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    samples = np.linspace(0, nsteps-1, nsteps)[::frame_skip]
    time = np.linspace(times[0], times[1], nsteps + 1)
    opacity = min(1, 300/N)

    x = x_array[0]
    y = x_array[1]
    names = ('x', 'y')


    x_lim = (np.min(x), np.max(x))
    y_lim = (np.min(y), np.max(y))

    txt = fig.suptitle('t={:.2f}'.format(times[0]))

    scat_kwargs = {'alpha':opacity, 'zorder':10}

    scat = [ax.scatter(x[state_lookup[key], 0], y[state_lookup[key], 0], **scat_kwargs) for key in state_lookup]

    fig.legend(state_lookup)

    ax.set(xlim=x_lim, ylim=y_lim, xlabel=names[0], ylabel=names[1])

    def animate(i):
        index = int(samples[i])
        t_c = time[index]
        x_i = x[:, index]
        y_i = y[:, index]

        for i, item in enumerate(state_lookup):
            scat[i].set_offsets(np.c_[x_i[state_lookup[item]], y_i[state_lookup[item]]])
        txt.set_text(f't={t_c:.2g}')

    ani = FuncAnimation(fig, animate, interval=100, frames=len(samples), blit=False)
    plt.close()
    return ani, fig, ax
