import numpy as np
import matplotlib.pyplot as plt

num_trial = [0, 1, 2, 3, 4, 5, 6, 7, 8,9]

######################################
# load the results with num_waypoints 1
######################################
loss_all_trials = []
param_error_all_trials = []
for k in num_trial:
    load = np.load('./true/num_waypoints_' + str(1) + '_trial_' + str(k) + '.npy',
                   allow_pickle=True).item()
    parameter_trace = np.array(load['parameter_trace'])[0:-1]
    true_parameter = np.array(load['true_parameter'])
    parameter_error = np.linalg.norm(parameter_trace - true_parameter, axis=1) ** 2
    loss_all_trials.append(load['loss_trace'])
    param_error_all_trials.append(parameter_error)
param_error_all_trials = np.array(param_error_all_trials)
loss_all_trials = np.array(loss_all_trials)

nwp_1_loss_mean = loss_all_trials.mean(axis=0)
nwp_1_loss_std = loss_all_trials.std(axis=0)
nwp_1_param_error_mean = param_error_all_trials.mean(axis=0)
nwp_1_param_error_std = param_error_all_trials.std(axis=0)

######################################
# load the results with num_waypoints 2
######################################
loss_all_trials = []
param_error_all_trials = []
for k in num_trial:
    load = np.load('./true/num_waypoints_' + str(2) + '_trial_' + str(k) + '.npy',
                   allow_pickle=True).item()
    parameter_trace = np.array(load['parameter_trace'])[0:-1]
    true_parameter = np.array(load['true_parameter'])
    parameter_error = np.linalg.norm(parameter_trace - true_parameter, axis=1) ** 2
    loss_all_trials.append(load['loss_trace'])
    param_error_all_trials.append(parameter_error)
param_error_all_trials = np.array(param_error_all_trials)
loss_all_trials = np.array(loss_all_trials)

nwp_2_loss_mean = loss_all_trials.mean(axis=0)
nwp_2_loss_std = loss_all_trials.std(axis=0)
nwp_2_param_error_mean = param_error_all_trials.mean(axis=0)
nwp_2_param_error_std = param_error_all_trials.std(axis=0)

######################################
# load the results with num_waypoints 3
######################################
loss_all_trials = []
param_error_all_trials = []
for k in num_trial:
    load = np.load('./true/num_waypoints_' + str(3) + '_trial_' + str(k) + '.npy',
                   allow_pickle=True).item()
    parameter_trace = np.array(load['parameter_trace'])[0:-1]
    true_parameter = np.array(load['true_parameter'])
    parameter_error = np.linalg.norm(parameter_trace - true_parameter, axis=1) ** 2
    loss_all_trials.append(load['loss_trace'])
    param_error_all_trials.append(parameter_error)
param_error_all_trials = np.array(param_error_all_trials)
loss_all_trials = np.array(loss_all_trials)

nwp_3_loss_mean = loss_all_trials.mean(axis=0)
nwp_3_loss_std = loss_all_trials.std(axis=0)
nwp_3_param_error_mean = param_error_all_trials.mean(axis=0)
nwp_3_param_error_std = param_error_all_trials.std(axis=0)

######################################
# load the results with num_waypoints 4
######################################
loss_all_trials = []
param_error_all_trials = []
for k in num_trial:
    load = np.load('./true/num_waypoints_' + str(4) + '_trial_' + str(k) + '.npy',
                   allow_pickle=True).item()
    parameter_trace = np.array(load['parameter_trace'])[0:-1]
    true_parameter = np.array(load['true_parameter'])
    parameter_error = np.linalg.norm(parameter_trace - true_parameter, axis=1) ** 2
    loss_all_trials.append(load['loss_trace'])
    param_error_all_trials.append(parameter_error)
param_error_all_trials = np.array(param_error_all_trials)
loss_all_trials = np.array(loss_all_trials)

nwp_4_loss_mean = loss_all_trials.mean(axis=0)
nwp_4_loss_std = loss_all_trials.std(axis=0)
nwp_4_param_error_mean = param_error_all_trials.mean(axis=0)
nwp_4_param_error_std = param_error_all_trials.std(axis=0)

######################################
# load the results with num_waypoints 8
######################################
loss_all_trials = []
param_error_all_trials = []
for k in num_trial:
    load = np.load('./true/num_waypoints_' + str(8) + '_trial_' + str(k) + '.npy',
                   allow_pickle=True).item()
    parameter_trace = np.array(load['parameter_trace'])[0:-1]
    true_parameter = np.array(load['true_parameter'])
    parameter_error = np.linalg.norm(parameter_trace - true_parameter, axis=1) ** 2
    loss_all_trials.append(load['loss_trace'])
    param_error_all_trials.append(parameter_error)
param_error_all_trials = np.array(param_error_all_trials)
loss_all_trials = np.array(loss_all_trials)

nwp_8_loss_mean = loss_all_trials.mean(axis=0)
nwp_8_loss_std = loss_all_trials.std(axis=0)
nwp_8_param_error_mean = param_error_all_trials.mean(axis=0)
nwp_8_param_error_std = param_error_all_trials.std(axis=0)

#############################
# set the plotting parameters
#############################


params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'legend.fontsize': 18}
plt.rcParams.update(params)

fig = plt.figure(0, figsize=(10, 4))
ax = fig.subplots(1, 2)

xvals = np.arange(len(nwp_1_loss_mean))
loss1, = ax[0].plot(xvals, nwp_1_loss_mean, alpha=1, linewidth=3, color='tab:purple')
ax[0].fill_between(xvals, nwp_1_loss_mean - nwp_1_loss_std,
                   nwp_1_loss_mean + nwp_1_loss_std,
                   alpha=0.25, linewidth=0, color='tab:purple')

xvals = np.arange(len(nwp_2_loss_mean))
loss2, = ax[0].plot(xvals, nwp_2_loss_mean, alpha=1, linewidth=3, color='tab:orange')
ax[0].fill_between(xvals, nwp_2_loss_mean - nwp_2_loss_std,
                   nwp_2_loss_mean + nwp_2_loss_std,
                   alpha=0.25, linewidth=0, color='tab:orange', )

xvals = np.arange(len(nwp_3_loss_mean))
loss3, = ax[0].plot(xvals, nwp_3_loss_mean, alpha=1, linewidth=3, color='tab:green')
ax[0].fill_between(xvals, nwp_3_loss_mean - nwp_3_loss_std,
                   nwp_3_loss_mean + nwp_3_loss_std,
                   alpha=0.25, linewidth=0, color='tab:green')

xvals = np.arange(len(nwp_4_loss_mean))
loss4, = ax[0].plot(xvals, nwp_4_loss_mean, alpha=1, linewidth=3, color='tab:blue')
ax[0].fill_between(xvals, nwp_4_loss_mean - nwp_4_loss_std,
                   nwp_4_loss_mean + nwp_4_loss_std,
                   alpha=0.25, linewidth=0, color='tab:blue')

xvals = np.arange(len(nwp_8_loss_mean))
loss8, = ax[0].plot(xvals, nwp_8_loss_mean, alpha=1, linewidth=3, color='tab:red')
ax[0].fill_between(xvals, nwp_8_loss_mean - nwp_8_loss_std,
                   nwp_8_loss_mean + nwp_8_loss_std,
                   alpha=0.25, linewidth=0, color='tab:red')

ax[0].set_xlabel('Iteration')
ax[0].set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
ax[0].set_yscale('log')
ax[0].grid()

xvals = np.arange(len(nwp_1_param_error_mean))
error1, = ax[1].plot(xvals, nwp_1_param_error_mean, alpha=1, linewidth=3, color='tab:purple')
ax[1].fill_between(xvals, nwp_1_param_error_mean - nwp_1_param_error_std,
                   nwp_1_param_error_mean + nwp_1_param_error_std,
                   alpha=0.25, linewidth=0, color='tab:purple')

xvals = np.arange(len(nwp_2_param_error_mean))
error2, = ax[1].plot(xvals, nwp_2_param_error_mean, alpha=1, linewidth=3, color='tab:orange')
ax[1].fill_between(xvals, nwp_2_param_error_mean - nwp_2_param_error_std,
                   nwp_2_param_error_mean + nwp_2_param_error_std,
                   alpha=0.25, linewidth=0, color='tab:orange')

xvals = np.arange(len(nwp_3_param_error_mean))
error3, = ax[1].plot(xvals, nwp_3_param_error_mean, alpha=1, linewidth=3, color='tab:green')
ax[1].fill_between(xvals, nwp_3_param_error_mean - nwp_3_param_error_std,
                   nwp_3_param_error_mean + nwp_3_param_error_std,
                   alpha=0.25, linewidth=0, color='tab:green')

xvals = np.arange(len(nwp_4_param_error_mean))
error4, = ax[1].plot(xvals, nwp_4_param_error_mean, alpha=1, linewidth=3, color='tab:blue')
ax[1].fill_between(xvals, nwp_4_param_error_mean - nwp_4_param_error_std,
                   nwp_4_param_error_mean + nwp_4_param_error_std,
                   alpha=0.25, linewidth=0, color='tab:blue')

xvals = np.arange(len(nwp_8_param_error_mean))
error8, = ax[1].plot(xvals, nwp_8_param_error_mean, alpha=1, linewidth=3, color='tab:red')
ax[1].fill_between(xvals, nwp_8_param_error_mean - nwp_8_param_error_std,
                   nwp_8_param_error_mean + nwp_8_param_error_std,
                   alpha=0.25, linewidth=0, color='tab:red')

ax[1].set_xlabel('Iteration')
ax[1].set_ylabel(r'$||\theta-\theta^{true}||^2$')
ax[1].set_yscale('log')
ax[1].grid()

fig.legend([loss1, loss2, loss3, loss4, loss8],
           ['1 keyframe', '2 keyframes', '3 keyframes', '4 keyframes', '8 keyframes', ],
           ncol=5, bbox_to_anchor=(1.01, 1.0), handlelength=0.6, columnspacing=0.7, handletextpad=0.4)
plt.subplots_adjust(left=0.12, right=0.95, bottom=0.16, wspace=0.4, top=0.82)

# plt.tight_layout()
plt.show()
