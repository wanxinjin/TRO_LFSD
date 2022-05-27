import matplotlib.pyplot as plt
from casadi import *

# plot the learning true
if True:
    params = {'axes.labelsize': 25,
              'axes.titlesize': 25,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'legend.fontsize': 20}
    plt.rcParams.update(params)

    # --------------------------- load the data-----------------------
    trials = [0, 1, 2, 3, 4]
    subcase1_loss_trace_list = []
    subcase1_learned_param_list = []
    for trial in trials:
        load = np.load('./case_feature/subcase1_trial_' + str(trial) + '.npy', allow_pickle=True).item()
        load_trace = np.array(load['loss_trace'])
        subcase1_loss_trace_list.append(load_trace)
        subcase1_learned_param_list.append(load['parameter_trace'][-1])
    subcase1_learned_param_mean = np.mean(np.array(subcase1_learned_param_list), axis=0)
    subcase1_learned_param_std = np.std(np.array(subcase1_learned_param_list), axis=0)
    print('learned_param_mean:', subcase1_learned_param_mean)

    subcase1_loss_trace_mean = np.mean(np.array(subcase1_loss_trace_list), axis=0)
    subcase1_loss_trace_std = np.std(np.array(subcase1_loss_trace_list), axis=0)

    print('final loss mean (subcase1):', subcase1_loss_trace_mean[-1])
    print('final loss std (subcase1):', subcase1_loss_trace_std[-1])

    trials = [0, 1, 2, 3, 4]
    subcase2_loss_trace_list = []
    for trial in trials:
        load = np.load('./case_feature/subcase2_trial_' + str(trial) + '.npy', allow_pickle=True).item()
        load_trace = np.array(load['loss_trace'])
        subcase2_loss_trace_list.append(load_trace)

    subcase2_loss_trace_mean = np.mean(np.array(subcase2_loss_trace_list), axis=0)
    subcase2_loss_trace_std = np.std(np.array(subcase2_loss_trace_list), axis=0)

    print('final loss mean (subcase2):', subcase2_loss_trace_mean[-1])
    print('final loss std (subcase2):', subcase2_loss_trace_std[-1])

    fig = plt.figure(2, figsize=(6, 4))
    ax = fig.subplots(1, 1)
    # ax.set_facecolor('#E6E6E6')
    xvals = np.arange(len(subcase1_loss_trace_mean))
    ax.plot(subcase1_loss_trace_mean, lw=5, color='tab:green', label=r'(40) with $p\in R^4$')
    ax.fill_between(xvals, subcase1_loss_trace_mean - subcase1_loss_trace_std,
                    subcase1_loss_trace_mean + subcase1_loss_trace_std,
                    alpha=0.25, linewidth=0, color='tab:green')

    ax.plot(subcase2_loss_trace_mean, lw=5, color='tab:orange', label=r'(40) with $p\geq0$')
    ax.fill_between(xvals, subcase2_loss_trace_mean - subcase2_loss_trace_std,
                    subcase2_loss_trace_mean + subcase2_loss_trace_std,
                    alpha=0.25, linewidth=0, color='tab:orange')

    ax.set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
    ax.set_xlabel('Iteration')
    ax.legend()

    ax.grid()
    ax.set_position([0.18, 0.18, 0.78, 0.80])
    # plt.tight_layout()
    plt.show()
