from CPDP import CPDP
from JinEnv import JinEnv
from casadi import *

import matplotlib.pyplot as plt

# plot the learning true
if True:
    params = {'axes.labelsize': 25,
              'axes.titlesize': 25,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'legend.fontsize': 20}
    plt.rcParams.update(params)

    # --------------------------- load the data-----------------------
    trials=[0,1,2,3]
    loss_trace_list=[]
    for trial in trials:
        load = np.load('./case_3/quadrotor_trial_'+str(trial)+'.npy', allow_pickle=True).item()
        loss_trace_list.append(load['loss_trace'])

    loss_trace_mean=np.mean(np.array(loss_trace_list), axis=0)
    loss_trace_std=np.std(np.array(loss_trace_list), axis=0)



    print('final loss mean:', loss_trace_mean[-1])
    print('final loss std:', loss_trace_std[-1])

    print('the final total distance to the waypoints is (mean): ', loss_trace_mean[-1])
    print('the final total distance to the waypoints is: (std)', loss_trace_std[-1])





    fig = plt.figure(2, figsize=(6, 4))
    ax = fig.subplots(1, 1)
    # ax.set_facecolor('#E6E6E6')
    xvals=np.arange(len(loss_trace_mean))
    ax.plot(loss_trace_mean, lw=5,  color='tab:blue')
    ax.fill_between(xvals, loss_trace_mean - loss_trace_std,
                   loss_trace_mean + loss_trace_std,
                   alpha=0.40, linewidth=0, color='tab:blue')
    ax.set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
    ax.set_xlabel('Iteration')
    ax.grid()
    ax.set_position([0.18, 0.18, 0.80, 0.80])
    # plt.tight_layout()
    plt.show()


