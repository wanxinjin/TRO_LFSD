from CPDP import CPDP
from JinEnv import JinEnv
from casadi import *

import matplotlib.pyplot as plt

# plot the learning results
if True:
    params = {'axes.labelsize': 25,
              'axes.titlesize': 25,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'legend.fontsize': 20}
    plt.rcParams.update(params)

    # --------------------------- load the data-----------------------
    load = np.load('./quadrotor_case5.npy', allow_pickle=True).item()
    parameter_trace = np.array(load['parameter_trace'])
    loss_trace = load['loss_trace']

    print('the final total distance to the waypoints is: ', loss_trace[-1])

    # eliminate some spike due to the failure of optimal control solver.
    for i in range(1, len(loss_trace)):
        if abs(loss_trace[i]-loss_trace[i-1])>30:
            loss_trace[i]=loss_trace[i-1]


    fig = plt.figure(2, figsize=(6, 4))
    ax = fig.subplots(1, 1)
    ax.set_facecolor('#E6E6E6')
    ax.plot(loss_trace, lw=5,  color='tab:blue')
    ax.set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
    ax.set_xlabel('Iteration')
    ax.set_xlim([-5,200])
    ax.grid()
    ax.set_position([0.18, 0.18, 0.78, 0.80])

    plt.show()


