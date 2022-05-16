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

load = np.load('./compare_keyframe.npy', allow_pickle=True).item()
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

    true_posa = np.array([np.pi / 2, 0])
    final_state_spline=np.array([spl_q1(time_interpolate)[-1], spl_q2(time_interpolate)[-1]])
    print('final state:', final_state_spline, '|  error to the goal:',
          np.linalg.norm(final_state_spline - true_posa))


# plot the generalization
if True:

    T_generalize = 2
    ini_state = [-np.pi / 4, 0, 0, 0]
    true_time_grid_genrealize = np.linspace(0, T_generalize, 20)

    # load data
    load_neural = np.load('./robotarm_neural_9.npy', allow_pickle=True).item()
    time_tau_neural = load_neural['time_tau']
    waypoint_neural = load_neural['waypoints']
    opt_sol_neural = load_neural['opt_sol']
    opt_sol_generalize_neural = load_neural['opt_sol_generalize']
    loss_neural = load_neural['loss_trace']

    load_weighted = np.load('./robotarm_true_8.npy', allow_pickle=True).item()
    parameter_trace_weighted = np.array(load_weighted['parameter_trace'])
    _, opt_sol_weighted = oc.cocSolver(ini_state, T_generalize, parameter_trace_weighted[-1])



    fig = plt.figure(2, figsize=(3.8, 5))
    ax = fig.subplots(2, 1)

    # ax[0].set_facecolor('#E6E6E6')
    neural, = ax[0].plot(true_time_grid_genrealize, opt_sol_generalize_neural(true_time_grid_genrealize)[:, 0],
                         linewidth=4,
                         color='tab:blue')
    weighted, = ax[0].plot(true_time_grid_genrealize, opt_sol_weighted(true_time_grid_genrealize)[:, 0],
                           linewidth=4,
                           color='black', linestyle='--')
    ax[0].legend([neural, weighted], ['Neural', 'Weighted', ], ncol=1, loc='lower right',
                 handlelength=1.2)
    ax[0].plot(true_time_grid_genrealize, np.pi/2 * np.ones_like(true_time_grid_genrealize), color='gray', linestyle='--', linewidth=4, )
    ax[0].set_xlabel(r'$\tau$')
    ax[0].set_ylim([-1.2, 1.8])
    ax[0].set_ylabel(r'$q_1$', labelpad=-5)
    ax[0].grid()

    # ax[1].set_facecolor('#E6E6E6')
    ax[1].plot(true_time_grid_genrealize, opt_sol_generalize_neural(true_time_grid_genrealize)[:, 1], linewidth=4,
               color='tab:blue')
    ax[1].plot(true_time_grid_genrealize, opt_sol_weighted(true_time_grid_genrealize)[:, 1],
               linewidth=4,
               color='black', linestyle='--')
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'$q_2$',labelpad=-3)
    ax[1].grid()
    ax[1].plot(true_time_grid_genrealize, 0 * np.ones_like(true_time_grid_genrealize), color='gray',
               linestyle='--', linewidth=4, )

    plt.subplots_adjust(left=0.22, right=0.98, bottom=0.12, top=0.99, hspace=0.40)
    plt.show()

    true_posa = np.array([np.pi / 2, 0, 0, 0])
    print('final state for neural case:', opt_sol_generalize_neural(true_time_grid_genrealize)[-1,0:4], '|  error to the goal:',
          np.linalg.norm(opt_sol_generalize_neural(true_time_grid_genrealize)[-1,0:4] - true_posa))
    print('final state for weights theta:', opt_sol_weighted(true_time_grid_genrealize)[-1,0:4], '|  error to the goal:',
          np.linalg.norm(opt_sol_weighted(true_time_grid_genrealize)[-1,0:4] - true_posa))
