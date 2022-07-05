from CPDP import CPDP
from JinEnv import JinEnv
from casadi import *
from scipy.integrate import solve_ivp
import scipy.io as sio

# ---------------------------------------load environment---------------------------------------
env = JinEnv.RobotArm()
env.initDyn(l1=1, m1=2, l2=1, m2=1, g=0)
env.initCost_WeightedDistance(wu=0.5)

# --------------------------- create optimal control object --------------------------
oc = CPDP.COCSys()
beta = SX.sym('beta')
dyn = beta * env.f
oc.setAuxvarVariable(vertcat(beta, env.cost_auxvar))
oc.setStateVariable(env.X)
oc.setControlVariable(env.U)
oc.setDyn(dyn)
path_cost = beta * env.path_cost
oc.setPathCost(path_cost)
oc.setFinalCost(env.final_cost)
oc.setIntegrator(n_grid=15)

print(oc.auxvar)

# --------------------------- plot --------------------------

import matplotlib.pyplot as plt

# set the plotting parameters
params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'legend.fontsize': 18}
plt.rcParams.update(params)

fig = plt.figure(0, figsize=(10, 4))
ax = fig.subplots(2, 3)

# load the data
trials = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# load data
loss_all_trials = []
param_error_all_trials = []
learned_param_all_trials = []
for k in trials:
    load = np.load('random/trial_' + str(k) + '.npy', allow_pickle=True).item()
    parameter_trace = np.array(load['parameter_trace'])[0:-1]
    true_parameter = np.array(load['true_parameter'])
    parameter_error = np.linalg.norm(parameter_trace - true_parameter, axis=1) ** 2
    loss_all_trials.append(load['loss_trace'])
    param_error_all_trials.append(parameter_error)
    learned_param_all_trials.append(parameter_trace[-1])
param_error_all_trials = np.array(param_error_all_trials)
loss_all_trials = np.array(loss_all_trials)
learned_param_all_trials = np.array(learned_param_all_trials)

loss_mean = loss_all_trials.mean(axis=0)
loss_std = loss_all_trials.std(axis=0)
param_error_mean = param_error_all_trials.mean(axis=0)
param_error_std = param_error_all_trials.std(axis=0)

learned_param_mean = learned_param_all_trials.mean(axis=0)

# --------------------------- plot the selection of the sparse demos
n_state = oc.n_state
ini_state = [-pi / 2, 3 * pi / 4, -5, 3]
T = 1
true_parameter = [5, 3, 3, 3, 3]
true_time_grid, true_opt_sol = oc.cocSolver(ini_state, T, true_parameter)
state_traj = true_opt_sol(true_time_grid)[:, 0:n_state]
time_tau = true_time_grid[[3, 5, 7, 9, 12, 14]]
waypoints = np.array([[-2., 2.5],
                      [-2., 2.],
                      [-1., 1.],
                      [-1., 1.],
                      [0., 1.],
                      [0., 1.]])

print('time_tau:\n', time_tau)
print('waypoints:\n', waypoints)

# for 1 waypoints
_, opt_sol = oc.cocSolver(ini_state, T, learned_param_mean)
regenerated_state_traj = opt_sol(true_time_grid)[:, 0:n_state]

# ax[0, 0].set_facecolor('#E6E6E6')
line_true, = ax[0, 0].plot(true_time_grid, state_traj[:, 0], linewidth=4, color='black', linestyle='-')
ax[0, 0].scatter(time_tau, waypoints[:, 0], marker="o", s=100, c='r', zorder=100)
reproduced_line, = ax[0, 0].plot(true_time_grid, regenerated_state_traj[:, 0], linewidth=4, color='tab:orange',
                                 linestyle='-')
ax[0, 0].set_ylabel(r'$q_1$')

ax[0, 0].set_ylim([-5.2, 2.5])
ax[0, 0].grid()
ax[0, 0].legend([line_true, reproduced_line],
                [r'$\theta^{true}}$', r'Learned $\theta$'],
                ncol=1, loc='lower right', columnspacing=.5, fontsize=14, handlelength=1)
ax[0, 0].plot(true_time_grid, np.pi / 2 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )
# ax[0, 0].text(1.15, np.pi/2-0.1, '$q^{g}_1$', fontsize=22)


# ax[1, 0].set_facecolor('#E6E6E6')
ax[1, 0].plot(true_time_grid, state_traj[:, 1], linewidth=4, color='black', linestyle='-')
ax[1, 0].scatter(time_tau, waypoints[:, 1], marker="o", s=100, c='r', zorder=100)
ax[1, 0].plot(true_time_grid, regenerated_state_traj[:, 1], linewidth=4, color='tab:orange', linestyle='-')
ax[1, 0].set_xlabel(r'$\tau$')
ax[1, 0].set_ylabel(r'$q_2$')
ax[1, 0].set_ylim([-0.5, 3.2])
ax[1, 0].grid()
ax[1, 0].plot(true_time_grid, 0 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )

# --------------------------- plot the loss and parameter error

xvals = np.arange(len(loss_mean))
ax[0, 1].plot(xvals, loss_mean, linewidth=4, color='tab:brown', linestyle='-')
ax[0, 1].fill_between(xvals, loss_mean - loss_std,
                      loss_mean + loss_std,
                      alpha=0.25, linewidth=0, color='tab:brown')
ax[0, 1].set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')

ax[0, 1].grid()

# ax[1, 1].set_facecolor('#E6E6E6')
xvals = np.arange(len(param_error_mean))
ax[1, 1].plot(xvals, param_error_mean, linewidth=4, color='tab:brown', linestyle='-')
ax[1, 1].fill_between(xvals, param_error_mean - param_error_std,
                      param_error_mean + param_error_std,
                      alpha=0.25, linewidth=0, color='tab:brown')
ax[1, 1].set_xlabel('Iteration')
ax[1, 1].set_ylabel(r'$||\theta-\theta^{true}||^2$')
ax[1, 1].grid()
ax[1, 1].set_ylim([6.5, 9])
print('final loss:', 'mean', loss_mean[-1], 'std', loss_std[-1])

# --------------------------- plot the generalization

# set the novel init state
ini_state = [-np.pi / 4, 0, 0, 0]
# set the novel horizon
T = 2
true_time_grid = np.linspace(0, T, 20)
# ground truth
true_parameter = [5, 3, 3, 3, 3]
_, true_opt_sol = oc.cocSolver(ini_state, T, true_parameter)
state_traj = true_opt_sol(true_time_grid)[:, 0:n_state]
# for 1 waypoints
_, opt_sol = oc.cocSolver(ini_state, T, learned_param_mean)
regenerated_state_traj = opt_sol(true_time_grid)[:, 0:n_state]

# ax[0, 2].set_facecolor('#E6E6E6')
true_line, = ax[0, 2].plot(true_time_grid, state_traj[:, 0], linewidth=6, color='black', linestyle='--')
generated_line, = ax[0, 2].plot(true_time_grid, regenerated_state_traj[:, 0], linewidth=5, color='tab:blue',
                                linestyle='-')

ax[0, 2].set_ylabel(r'$q_2$')
ax[0, 2].grid()
ax[0, 2].legend([true_line, generated_line],
                [r'$\theta^{true}$', r'learned $\theta$'],
                ncol=1, loc='lower right', columnspacing=1.5, fontsize=15, handlelength=1)
ax[0, 2].plot(true_time_grid, np.pi / 2 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )

# ax[1, 2].set_facecolor('#E6E6E6')
ax[1, 2].plot(true_time_grid, state_traj[:, 1], linewidth=6, color='black', linestyle='--')
ax[1, 2].plot(true_time_grid, regenerated_state_traj[:, 1], linewidth=4, color='tab:blue', linestyle='-')
ax[1, 2].set_xlabel(r'$\tau$')
ax[1, 2].set_ylabel(r'$q_2$')
ax[1, 2].grid()
ax[1, 2].plot(true_time_grid, 0 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )

plt.subplots_adjust(left=0.09, right=0.98, bottom=0.15, wspace=0.48, top=0.98, hspace=0.30)
# plt.tight_layout()
plt.show()

true_posa = np.array([np.pi / 2, 0, 0, 0])
print('final state:', regenerated_state_traj[-1], '|  error to the goal:',
      np.linalg.norm(regenerated_state_traj[-1] - true_posa))
print('final state for true theta:', state_traj[-1], '|  error to the goal:',
      np.linalg.norm(state_traj[-1] - true_posa))
