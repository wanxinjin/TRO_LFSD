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
    # Plot the learned trajectory
    load = np.load('./quadrotor_feature_case4.npy', allow_pickle=True).item()
    time_tau = load['time_tau']
    waypoints = load['waypoints']
    parameter_trace = np.array(load['parameter_trace'])
    loss_trace = load['loss_trace']

    # Plot the learned trajectory
    load2 = np.load('./quadrotor_feature_case4_2.npy', allow_pickle=True).item()
    parameter_trace2 = np.array(load2['parameter_trace'])
    loss_trace2 = load2['loss_trace']

    # eliminate some spike due to the failure of optimal control solver.
    for i in range(1, len(loss_trace)):
        if abs(loss_trace[i]-loss_trace[i-1])>30:
            loss_trace[i]=loss_trace[i-1]
    for i in range(1, len(loss_trace2)):
        if abs(loss_trace2[i]-loss_trace2[i-1])>30:
            loss_trace2[i]=loss_trace2[i-1]


    fig = plt.figure(2, figsize=(6, 4))
    ax = fig.subplots(1, 1)
    ax.set_facecolor('#E6E6E6')
    # ax.plot(loss_trace, lw=5,  color='tab:green', label=r'$p\in R^4$')
    ax.plot(loss_trace2, lw=5,  color='tab:orange',label=r'(41) with $p\geq0$')
    ax.legend()
    ax.set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
    ax.set_xlabel('Iteration')
    ax.set_xlim([-5,150])
    ax.grid()
    ax.set_position([0.18, 0.18, 0.78, 0.80])

    plt.show()


