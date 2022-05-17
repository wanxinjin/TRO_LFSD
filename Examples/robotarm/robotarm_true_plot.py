import numpy as np
from CPDP import CPDP
from JinEnv import JinEnv
import casadi

# ---------------------------------------load environment---------------------------------------
env = JinEnv.RobotArm()
env.initDyn(l1=1, m1=2, l2=1, m2=1, g=0)
env.initCost_WeightedDistance(wu=0.5)

# --------------------------- create optimal control object --------------------------
oc = CPDP.COCSys()
beta = casadi.SX.sym('beta')
dyn = beta * env.f
oc.setAuxvarVariable(casadi.vertcat(beta, env.cost_auxvar))
oc.setStateVariable(env.X)
oc.setControlVariable(env.U)
oc.setDyn(dyn)
path_cost = beta * env.path_cost
oc.setPathCost(path_cost)
oc.setFinalCost(env.final_cost)
oc.setIntegrator(n_grid=10)

import matplotlib.pyplot as plt

# set the plotting parameters
params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'legend.fontsize': 18}
plt.rcParams.update(params)

# load date
num_waypoints = [2, 3, 4, 8]
num_trial = [0, 1, 2, 3, 4, 5, 6, 7, 8]

loss_mean_list = []
loss_std_list = []
param_error_mean_list = []
param_error_std_list = []
for i in num_waypoints:
    loss_trace_all_trials = []
    param_error_trace_all_trials = []
    for k in num_trial:
        load = np.load('results/true_trial_' + str(k) + '_num_waypoints_' + str(i) + '.npy',
                       allow_pickle=True).item()
        parameter_trace = np.array(load['parameter_trace'])[1:]
        true_parameter = np.array(load['true_parameter'])
        parameter_error = np.linalg.norm(parameter_trace - true_parameter, axis=1) ** 2
        loss_trace_all_trials.append(load['loss_trace'])
        param_error_trace_all_trials.append(parameter_error)

    param_error_trace_all_trials = np.array(param_error_trace_all_trials)
    loss_mean_list.append(param_error_trace_all_trials.mean(axis=0))
    loss_std_list.append(param_error_trace_all_trials.std(axis=0))
    param_error_trace_all_trials = np.array(param_error_trace_all_trials)
    param_error_mean_list.append(param_error_trace_all_trials.mean(axis=0))
    param_error_std_list.append(param_error_trace_all_trials.std(axis=0))

fig = plt.figure(0, figsize=(10, 4))
ax = fig.subplots(1, 2)

# ax[0].set_facecolor('#E6E6E6')
xvals = np.arange(loss_mean_list[0].shape[0])
ax[0].fill_between(xvals, loss_mean_list[0] - loss_std_list[0],
                   loss_mean_list[0] + loss_std_list[0],
                   alpha=0.25, linewidth=3, color='tab:purple')
ax[0].fill_between(xvals, loss_mean_list[1] - loss_std_list[1],
                   loss_mean_list[1] + loss_std_list[1],
                   alpha=0.25, linewidth=3, color='tab:orange')

ax[0].set_xlabel('Iteration')
ax[0].set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
# ax[0].set_xlim([0,20])
ax[0].set_yscale('log')
ax[0].grid()

# # ax[1].set_facecolor('#E6E6E6')
ax[1].fill_between(xvals, param_error_mean_list[0] - param_error_std_list[0],
                   param_error_mean_list[0] + param_error_std_list[0],
                   alpha=0.25, linewidth=3, color='tab:purple')
ax[1].fill_between(xvals, param_error_mean_list[1] - param_error_std_list[1],
                   param_error_mean_list[1] + param_error_std_list[1],
                   alpha=0.25, linewidth=3, color='tab:purple')

# error2, = ax[1].plot(parameter_error2, linewidth=3, color='tab:orange')
# error3, = ax[1].plot(parameter_error3, linewidth=3, color='tab:green')
# error4, = ax[1].plot(parameter_error4, linewidth=3, color='tab:blue')
# error8, = ax[1].plot(parameter_error8, linewidth=3, color='tab:red')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel(r'$||\theta-\theta^{true}||^2$')
# ax[1].set_yscale('log')
ax[1].grid()
#
# plt.subplots_adjust(left=0.12, right=0.95, bottom=0.16, wspace=0.4, top=0.84)
# fig.legend([loss1, loss2, loss3, loss4, loss8],
#            ['1 keyframe', '2 keyframes', '3 keyframes',
#             '4 keyframes', '8 keyframes', ],
#            ncol=5, bbox_to_anchor=(1.00, 1.0), handlelength=0.7, columnspacing=0.5, handletextpad=0.4)
plt.show()
