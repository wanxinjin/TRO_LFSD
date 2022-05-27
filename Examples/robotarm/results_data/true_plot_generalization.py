import numpy as np
import matplotlib.pyplot as plt
from CPDP import CPDP
from JinEnv import JinEnv
from casadi import *

# ---------------------------------------load data ---------------------------------------
if True:
    num_trial = [0, 1, 2, 3, 4, 5,6,7, 8,9]

    ######################################
    # load the results with num_waypoints 1
    ######################################
    learned_param_all_trials = []
    for k in num_trial:
        load = np.load('./true/num_waypoints_' + str(1) + '_trial_' + str(k) + '.npy',
                       allow_pickle=True).item()
        parameter_trace = np.array(load['parameter_trace'])[1:]
        learned_param_all_trials.append(parameter_trace[-1])
    learned_param_all_trials = np.array(learned_param_all_trials)
    nwp_1_learned_param_mean = learned_param_all_trials.mean(axis=0)

    ######################################
    # load the results with num_waypoints 2
    ######################################
    learned_param_all_trials = []
    for k in num_trial:
        load = np.load('./true/num_waypoints_' + str(2) + '_trial_' + str(k) + '.npy',
                       allow_pickle=True).item()
        parameter_trace = np.array(load['parameter_trace'])[1:]
        learned_param_all_trials.append(parameter_trace[-1])
    learned_param_all_trials = np.array(learned_param_all_trials)
    nwp_2_learned_param_mean = learned_param_all_trials.mean(axis=0)

    ######################################
    # load the results with num_waypoints 3
    ######################################
    learned_param_all_trials = []
    for k in num_trial:
        load = np.load('./true/num_waypoints_' + str(3) + '_trial_' + str(k) + '.npy',
                       allow_pickle=True).item()
        parameter_trace = np.array(load['parameter_trace'])[1:]
        learned_param_all_trials.append(parameter_trace[-1])
    learned_param_all_trials = np.array(learned_param_all_trials)
    nwp_3_learned_param_mean = learned_param_all_trials.mean(axis=0)

    ######################################
    # load the results with num_waypoints 4
    ######################################
    learned_param_all_trials = []
    for k in num_trial:
        load = np.load('./true/num_waypoints_' + str(4) + '_trial_' + str(k) + '.npy',
                       allow_pickle=True).item()
        parameter_trace = np.array(load['parameter_trace'])[1:]
        learned_param_all_trials.append(parameter_trace[-1])
    learned_param_all_trials = np.array(learned_param_all_trials)
    nwp_4_learned_param_mean = learned_param_all_trials.mean(axis=0)

    ######################################
    # load the results with num_waypoints 8
    ######################################
    learned_param_all_trials = []
    for k in num_trial:
        load = np.load('./true/num_waypoints_' + str(8) + '_trial_' + str(k) + '.npy',
                       allow_pickle=True).item()
        parameter_trace = np.array(load['parameter_trace'])[1:]
        learned_param_all_trials.append(parameter_trace[-1])
    learned_param_all_trials = np.array(learned_param_all_trials)
    nwp_8_learned_param_mean = learned_param_all_trials.mean(axis=0)

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
n_state = oc.n_state
print(oc.auxvar)

# ---------------------------- define loss function ------------------------------------
interface_pos_fn = Function('interface', [oc.state], [oc.state[0:2]])
diff_interface_pos_fn = Function('diff_interface', [oc.state], [jacobian(oc.state[0:2], oc.state)])


def getloss_pos_corrections(time_grid, target_waypoints, opt_sol, auxsys_sol):
    loss = 0
    diff_loss = numpy.zeros(oc.n_auxvar)
    for k, t in enumerate(time_grid):
        # solve loss
        target_waypoint = target_waypoints[k, :]
        target_position = target_waypoint[0:2]
        current_position = interface_pos_fn(opt_sol(t)[0:oc.n_state]).full().flatten()

        loss += numpy.linalg.norm(target_position - current_position) ** 2
        # solve gradient by chain rule
        dl_dpos = current_position - target_position
        dpos_dx = diff_interface_pos_fn(opt_sol(t)[0:oc.n_state]).full()
        dxpos_dp = auxsys_sol(t)[0:oc.n_state * oc.n_auxvar].reshape((oc.n_state, oc.n_auxvar))

        dl_dp = np.matmul(numpy.matmul(dl_dpos, dpos_dx), dxpos_dp)
        diff_loss += dl_dp
    return loss, diff_loss


# --------------------------- ground truth parameter ----------------------------------------
true_parameter = [5, 3, 3, 3, 3]

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

# new initial condition and new horizon
ini_state = [-np.pi / 4, 0, 0, 0]
T = 2
true_time_grid = np.linspace(0, T, 20)

_, true_opt_sol = oc.cocSolver(ini_state, T, true_parameter)
state_traj = true_opt_sol(true_time_grid)[:, 0:n_state]

# for 1 waypoints
_, true_opt_sol1 = oc.cocSolver(ini_state, T, nwp_1_learned_param_mean)
state_traj1 = true_opt_sol1(true_time_grid)[:, 0:n_state]

# for 2 waypoints
_, true_opt_sol2 = oc.cocSolver(ini_state, T, nwp_2_learned_param_mean)
state_traj2 = true_opt_sol2(true_time_grid)[:, 0:n_state]

# for 3 waypoints
_, true_opt_sol3 = oc.cocSolver(ini_state, T, nwp_3_learned_param_mean)
state_traj3 = true_opt_sol3(true_time_grid)[:, 0:n_state]

# for 4 waypoints
_, true_opt_sol4 = oc.cocSolver(ini_state, T, nwp_4_learned_param_mean)
state_traj4 = true_opt_sol4(true_time_grid)[:, 0:n_state]

# for 8 waypoints
_, true_opt_sol8 = oc.cocSolver(ini_state, T, nwp_8_learned_param_mean)
state_traj8 = true_opt_sol8(true_time_grid)[:, 0:n_state]





# ax[0].set_facecolor('#E6E6E6')
line_true, = ax[0].plot(true_time_grid, state_traj[:, 0], linewidth=6, color='black', linestyle='--')
line_1, = ax[0].plot(true_time_grid, state_traj1[:, 0], linewidth=3, color='tab:purple')
line_2, = ax[0].plot(true_time_grid, state_traj2[:, 0], linewidth=3, color='tab:orange')
line_3, = ax[0].plot(true_time_grid, state_traj3[:, 0], linewidth=3, color='tab:green')
line_4, = ax[0].plot(true_time_grid, state_traj4[:, 0], linewidth=4, color='tab:blue')
line_8, = ax[0].plot(true_time_grid, state_traj8[:, 0], linewidth=3, color='tab:red')
ax[0].plot(true_time_grid, np.pi / 2 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )
ax[0].text(2.15, np.pi / 2 - 0.1, '$q^{g}_1$', fontsize=22)
# ax[0].set_xticks(np.arange(0,2,5))
ax[0].set_xlabel(r'$\tau$')
ax[0].set_ylabel(r'$q_1$')
# ax[0].set_yticks(np.arange(-0.5,1.5,5))
ax[0].grid()

# ax[1].set_facecolor('#E6E6E6')
ax[1].plot(true_time_grid, state_traj[:, 1], linewidth=6, color='black', linestyle='--')
ax[1].plot(true_time_grid, state_traj1[:, 1], linewidth=3, color='tab:purple')
ax[1].plot(true_time_grid, state_traj2[:, 1], linewidth=3, color='tab:orange')
ax[1].plot(true_time_grid, state_traj3[:, 1], linewidth=3, color='tab:green')
ax[1].plot(true_time_grid, state_traj4[:, 1], linewidth=4, color='tab:blue')
ax[1].plot(true_time_grid, state_traj8[:, 1], linewidth=3, color='tab:red')
ax[1].plot(true_time_grid, 0 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )
ax[1].text(2.15, 0 - 0.005, '$q^{g}_2$', fontsize=22)
ax[1].set_xlabel(r'$\tau$')
ax[1].set_ylabel(r'$q_2$')
ax[1].grid()

plt.subplots_adjust(left=0.10, right=0.95, bottom=0.15, wspace=0.4, top=0.84)
fig.legend([line_1, line_2, line_3, line_4, line_8, line_true, ],
           ['1 keyframe', '2 keyframes', '3 keyframes',
            '4 keyframes', '8 keyframes', ],
           ncol=5, bbox_to_anchor=(1.00, 1.0), handlelength=0.7, columnspacing=0.5, handletextpad=0.4)
plt.show()

#   print the final pose in each learned case
true_posa = np.array([np.pi / 2, 0, 0, 0])
print('final state for 1 keyframes:', state_traj1[-1], '|  error to the goal:',
      np.linalg.norm(state_traj1[-1] - true_posa))
print('final state for 2 keyframes:', state_traj2[-1], '|  error to the goal:',
      np.linalg.norm(state_traj2[-1] - true_posa))
print('final state for 3 keyframes:', state_traj3[-1], '|  error to the goal:',
      np.linalg.norm(state_traj3[-1] - true_posa))
print('final state for 4 keyframes:', state_traj4[-1], '|  error to the goal:',
      np.linalg.norm(state_traj4[-1] - true_posa))
print('final state for 8 keyframes:', state_traj8[-1], '|  error to the goal:',
      np.linalg.norm(state_traj8[-1] - true_posa))
print('final state for true theta:', state_traj[-1], '|  error to the goal:',
      np.linalg.norm(state_traj[-1] - true_posa))
