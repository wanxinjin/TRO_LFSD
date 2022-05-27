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
          'legend.fontsize': 16}
plt.rcParams.update(params)

load = np.load('./keyframes/compare_keyframe.npy', allow_pickle=True).item()
time_tau = load['time_tau']
waypoints_q1 = load['waypoints_q1']
waypoints_q2 = load['waypoints_q2']
spl_q1 = load['spl_q1']
spl_q2 = load['spl_q2']
plot_time = load['plot_time']
waypoints = load['waypoints']

if True:
    fig = plt.figure(0, figsize=(3.8, 5))
    ax = fig.subplots(2, 1)

    # ax[0].set_facecolor('#E6E6E6')
    ax[0].plot(plot_time, spl_q1(plot_time), linewidth=4, color='tab:brown')
    ax[0].scatter(time_tau, waypoints_q1, marker="o", s=100, c='r', zorder=100)
    ax[0].set_xlabel(r'$\tau$')
    ax[0].set_ylabel(r'$q_1$', labelpad=-5)
    ax[0].grid()
    ax[0].plot(plot_time,np.pi/2 * np.ones_like(plot_time), color='gray', linestyle='--', linewidth=4, )

    # ax[1].set_facecolor('#E6E6E6')
    ax[1].plot(plot_time, spl_q2(plot_time), linewidth=4, color='tab:brown')
    ax[1].scatter(time_tau, waypoints_q2, marker="o", s=100, c='r', zorder=100)
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'$q_2$')
    ax[1].grid()
    ax[1].plot(plot_time, 0. * np.ones_like(plot_time), color='gray', linestyle='--', linewidth=4, )
    plt.subplots_adjust(left=0.16, right=0.98, bottom=0.12, top=0.99, hspace=0.40)
    plt.show()

# plot generalization
if True:
    ini_state_generalize = [-np.pi / 4, 0, 0, 0]
    T_generalize = 2
    # check which waypoint is nearest to the new initial state
    nearest_index = np.argmin(np.linalg.norm(waypoints - np.array(ini_state_generalize[0:2]), axis=1))
    time_interpolate = np.linspace(time_tau[nearest_index], time_tau[nearest_index] + T_generalize, 50)

    fig = plt.figure(1, figsize=(3.8, 5))
    ax = fig.subplots(2, 1)

    # ax[0].set_facecolor('#E6E6E6')
    ax[0].plot(time_interpolate - time_tau[nearest_index], spl_q1(time_interpolate), linewidth=4, color='tab:brown')
    ax[0].set_xlabel(r'$\tau$')
    ax[0].plot(time_interpolate - time_tau[nearest_index],np.pi/2 * np.ones_like(time_interpolate), color='gray', linestyle='--', linewidth=4, )

    ax[0].set_ylabel(r'$q_1$', labelpad=-5)
    ax[0].grid()

    # ax[1].set_facecolor('#E6E6E6')
    ax[1].plot(time_interpolate - time_tau[nearest_index], spl_q2(time_interpolate), linewidth=4, color='tab:brown')
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'$q_2$')
    ax[1].grid()
    ax[1].plot(time_interpolate - time_tau[nearest_index], 0 * np.ones_like(time_interpolate), color='gray', linestyle='--', linewidth=4, )

    plt.subplots_adjust(left=0.22, right=0.98, bottom=0.12, top=0.99, hspace=0.40)
    plt.show()

    true_pose = np.array([np.pi / 2, 0])
    final_state_spline=np.array([spl_q1(time_interpolate)[-1], spl_q2(time_interpolate)[-1]])
    print('final state:', final_state_spline, '|  error to the goal:',
          np.linalg.norm(final_state_spline - true_pose))


# plot the generalization
if True:
    T_generalize = 2
    ini_state = [-np.pi / 4, 0, 0, 0]

    learned_param_all_trials = []
    for k in range(0,10):
        load = np.load('../../robotarm/results_data/true/num_waypoints_' + str(8) + '_trial_' + str(k) + '.npy',
                       allow_pickle=True).item()
        parameter_trace = np.array(load['parameter_trace'])
        learned_param_all_trials.append(parameter_trace[-1])
    learned_param_all_trials = np.array(learned_param_all_trials)
    nwp_8_learned_param_mean = learned_param_all_trials.mean(axis=0)

    _, opt_sol_weighted = oc.cocSolver(ini_state, T_generalize, nwp_8_learned_param_mean)



    # load data
    opt_sol_generalize_list=[]
    trials=[0,2,3,4,5,6,7,8,9]
    for k in trials:
        load=np.load('../../robotarm/results_data/neural/num_waypoints_8_trial_'+str(k)+'.npy', allow_pickle=True).item()
        opt_sol_generalize_list.append(load['opt_sol_generalize'])

    time_grid_generalized=np.linspace(0,2,20)
    opt_state_traj_generalized_list=[]
    for opt_sol_generalized in opt_sol_generalize_list:
        opt_state_traj_generalized_list.append(opt_sol_generalized(time_grid_generalized)[:,0:4])
    opt_state_traj_generalized_mean_neural=np.mean(np.array(opt_state_traj_generalized_list), axis=0)
    opt_state_traj_generalized_std_neural=np.std(np.array(opt_state_traj_generalized_list), axis=0)

    fig = plt.figure(2, figsize=(3.8, 5))
    ax = fig.subplots(2, 1)

    # ax[0].set_facecolor('#E6E6E6')
    neural, = ax[0].plot(time_grid_generalized, opt_state_traj_generalized_mean_neural[:, 0],
                         linewidth=4,
                         color='tab:blue')
    weighted, = ax[0].plot(time_grid_generalized, opt_sol_weighted(time_grid_generalized)[:, 0],
                           linewidth=4,
                           color='black', linestyle='--')
    ax[0].legend([neural, weighted], ['Neural', 'Weighted', ], ncol=1, loc='lower right',
                 handlelength=1.2)
    ax[0].plot(time_grid_generalized, np.pi/2 * np.ones_like(time_grid_generalized), color='gray', linestyle='--', linewidth=4, )
    ax[0].set_xlabel(r'$\tau$')
    ax[0].set_ylim([-1.2, 1.8])
    ax[0].set_ylabel(r'$q_1$', labelpad=-5)
    ax[0].grid()

    # ax[1].set_facecolor('#E6E6E6')
    ax[1].plot(time_grid_generalized, opt_state_traj_generalized_mean_neural[:, 1], linewidth=4,
               color='tab:blue')
    ax[1].plot(time_grid_generalized, opt_sol_weighted(time_grid_generalized)[:, 1],
               linewidth=4,
               color='black', linestyle='--')
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'$q_2$',labelpad=-3)
    ax[1].grid()
    ax[1].plot(time_grid_generalized, 0 * np.ones_like(time_grid_generalized), color='gray',
               linestyle='--', linewidth=4, )

    plt.subplots_adjust(left=0.22, right=0.98, bottom=0.12, top=0.99, hspace=0.40)
    plt.show()

    true_pose = np.array([np.pi / 2, 0, 0, 0])
    print('final state for neural case:', opt_state_traj_generalized_mean_neural[-1,0:4], '|  error to the goal:',
          np.linalg.norm(opt_state_traj_generalized_mean_neural[-1,0:4] - true_pose))
    print('final state for weights theta:', opt_sol_weighted(time_grid_generalized)[-1,0:4], '|  error to the goal:',
          np.linalg.norm(opt_sol_weighted(time_grid_generalized)[-1,0:4] - true_pose))
