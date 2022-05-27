import numpy as np

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
          'legend.fontsize': 15}
plt.rcParams.update(params)

# produce ground true generalized data
ini_state_generalize = [-np.pi / 4, 0, 0, 0]
T_generalize = 2
true_time_grid_genrealize = np.linspace(0, T_generalize, 20)
true_parameter = [5, 3, 3, 3, 3]
_, true_opt_sol_genrealize = oc.cocSolver(ini_state_generalize, T_generalize, true_parameter)

# produce ground true generalized data
ini_state = [-np.pi / 2, 3 * np.pi / 4, -5, 3]
T = 1
true_time_grid = np.linspace(0, T, 20)
true_parameter = [5, 3, 3, 3, 3]
_, true_opt_sol = oc.cocSolver(ini_state, T, true_parameter)


# take 4 keyframes
if True:
    # load data
    time_tau=0
    waypoints=0
    opt_sol_list=[]
    opt_sol_generalize_list=[]
    loss_trace_list=[]

    max_trial=5
    for k in range(0,max_trial):
        load=np.load('./neural/num_waypoints_4_3_trial_'+str(k)+'.npy', allow_pickle=True).item()
        time_tau=load['time_tau']
        waypoints=load['waypoints']
        opt_sol_list.append(load['opt_sol'])
        opt_sol_generalize_list.append(load['opt_sol_generalize'])
        loss_trace_list.append(load['loss_trace'])


    fig = plt.figure(0, figsize=(3.8, 10))
    ax = fig.subplots(5, 1)

    # plot the loss function
    loss_trace_mean=np.mean(np.array(loss_trace_list), axis=0)
    loss_trace_std=np.std(np.array(loss_trace_list), axis=0)
    xvals=np.arange(len(loss_trace_mean))
    # ax[0].set_facecolor('#E6E6E6')
    ax[0].set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
    ax[0].set_xlabel('Iteration')
    ax[0].plot(loss_trace_mean, linewidth=3, color='tab:brown')
    ax[0].fill_between(xvals, loss_trace_mean - loss_trace_std,
                   loss_trace_mean + loss_trace_std,
                   alpha=0.5, linewidth=0, color='tab:brown')
    ax[0].grid()

    # plot the loss function
    time_grid=np.linspace(0,1,20)
    opt_state_traj_list=[]
    for opt_sol in opt_sol_list:
        opt_state_traj_list.append(opt_sol(time_grid)[:,0:4])
    opt_state_traj_mean=np.mean(np.array(opt_state_traj_list), axis=0)
    opt_state_traj_std=np.std(np.array(opt_state_traj_list), axis=0)



    ax[1].plot(time_grid, opt_state_traj_mean[:, 0], linewidth=4, color='tab:orange')
    ax[1].scatter(time_tau, waypoints[:, 0], marker="o", s=100, c='r', zorder=100)
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'$q_1$')
    ax[1].grid()
    ax[1].plot(time_grid, np.pi / 2 * np.ones_like(time_grid), color='gray', linestyle='--', linewidth=5, )
    # ax[2].set_facecolor('#E6E6E6')

    reproduced, = ax[2].plot(time_grid, opt_state_traj_mean[:, 1], linewidth=4, color='tab:orange')
    givewaypoints = ax[2].scatter(time_tau, waypoints[:, 1], marker="o", s=100, c='r', zorder=100)
    ax[2].set_xlabel(r'$\tau$')
    ax[2].set_ylabel(r'$q_2$')
    ax[2].legend([givewaypoints, reproduced], ['Keyframes', 'Reproduced', ], ncol=1, handlelength=1.2)
    ax[2].plot(time_grid, 0 * np.ones_like(time_grid), color='gray', linestyle='--', linewidth=5, )

    ax[2].grid()

    #plot generalized motion
    time_grid_generalized=np.linspace(0,2,20)
    opt_state_traj_generalized_list=[]
    for opt_sol_generalized in opt_sol_generalize_list:
        opt_state_traj_generalized_list.append(opt_sol_generalized(time_grid_generalized)[:,0:4])
    opt_state_traj_generalized_mean=np.mean(np.array(opt_state_traj_generalized_list), axis=0)
    opt_state_traj_generalized_std=np.std(np.array(opt_state_traj_generalized_list), axis=0)

    generalized, = ax[3].plot(time_grid_generalized, opt_state_traj_generalized_mean[:, 0],
                              linewidth=4,
                              color='tab:blue')
    groundtruth, = ax[3].plot(time_grid_generalized, true_opt_sol_genrealize(time_grid_generalized)[:, 0],
                              linewidth=4,
                              color='black', linestyle='--')
    ax[3].legend([generalized, groundtruth], [r'$\theta$ (neural)', r'$\theta^{true}$', ], ncol=1,
                 bbox_to_anchor=(0.41, 0.15, 0.5, 0.5), handlelength=1.2, framealpha=0.1)
    ax[3].set_xlabel(r'$\tau$')
    ax[3].set_ylabel(r'$q_1$')
    ax[3].set_ylim([-1.2, 2])
    ax[3].grid()
    ax[3].plot(time_grid_generalized, np.pi / 2 * np.ones_like(time_grid_generalized), color='gray', linestyle='--',
               linewidth=5, )

    # ax[4].set_facecolor('#E6E6E6')
    ax[4].plot(time_grid_generalized, opt_state_traj_generalized_mean[:, 1], linewidth=4,
               color='tab:blue')
    ax[4].plot(time_grid_generalized, true_opt_sol_genrealize(time_grid_generalized)[:, 1], linewidth=4,
               color='black', linestyle='--')
    ax[4].set_xlabel(r'$\tau$')
    ax[4].set_ylabel(r'$q_2$', labelpad=-3)
    # ax[4].set_ylim([-0.5, 0.25])
    ax[4].grid()
    ax[4].plot(time_grid_generalized, 0 * np.ones_like(time_grid_generalized), color='gray', linestyle='--', linewidth=5, )
    plt.subplots_adjust(left=0.25, right=0.98, bottom=0.06, top=0.99, hspace=0.50)

    print(opt_state_traj_generalized_mean[-1])

    plt.show()

    true_posa = np.array([np.pi / 2, 0, 0, 0])
    print('final state for first case:', opt_state_traj_generalized_mean[-1],
          '|  error to the goal:',
          np.linalg.norm(opt_state_traj_generalized_mean[-1] - true_posa))
    print('final state for true theta:', true_opt_sol_genrealize(time_grid_generalized)[-1, 0:4],
          '|  error to the goal:',
          np.linalg.norm(true_opt_sol_genrealize(time_grid_generalized)[-1, 0:4] - true_posa))

# take 4 keyframes
if True:
    # load data
    time_tau=0
    waypoints=0
    opt_sol_list=[]
    opt_sol_generalize_list=[]
    loss_trace_list=[]

    max_trial=10
    for k in range(0,max_trial):
        load=np.load('./neural/num_waypoints_4_4_trial_'+str(k)+'.npy', allow_pickle=True).item()
        time_tau=load['time_tau']
        waypoints=load['waypoints']
        opt_sol_list.append(load['opt_sol'])
        opt_sol_generalize_list.append(load['opt_sol_generalize'])
        loss_trace_list.append(load['loss_trace'])


    fig = plt.figure(0, figsize=(3.8, 10))
    ax = fig.subplots(5, 1)

    # plot the loss function
    loss_trace_mean=np.mean(np.array(loss_trace_list), axis=0)
    loss_trace_std=np.std(np.array(loss_trace_list), axis=0)
    xvals=np.arange(len(loss_trace_mean))
    # ax[0].set_facecolor('#E6E6E6')
    ax[0].set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
    ax[0].set_xlabel('Iteration')
    ax[0].plot(loss_trace_mean, linewidth=3, color='tab:brown')
    ax[0].fill_between(xvals, loss_trace_mean - loss_trace_std,
                   loss_trace_mean + loss_trace_std,
                   alpha=0.50, linewidth=0, color='tab:brown')
    # ax[0].set_yscale('log')
    ax[0].grid()

    # plot the loss function
    time_grid=np.linspace(0,1,20)
    opt_state_traj_list=[]
    for opt_sol in opt_sol_list:
        opt_state_traj_list.append(opt_sol(time_grid)[:,0:4])
    opt_state_traj_mean=np.mean(np.array(opt_state_traj_list), axis=0)
    opt_state_traj_std=np.std(np.array(opt_state_traj_list), axis=0)



    ax[1].plot(time_grid, opt_state_traj_mean[:, 0], linewidth=4, color='tab:orange')
    ax[1].scatter(time_tau, waypoints[:, 0], marker="o", s=100, c='r', zorder=100)
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'$q_1$')
    ax[1].grid()
    ax[1].plot(time_grid, np.pi / 2 * np.ones_like(time_grid), color='gray', linestyle='--', linewidth=5, )
    # ax[2].set_facecolor('#E6E6E6')

    reproduced, = ax[2].plot(time_grid, opt_state_traj_mean[:, 1], linewidth=4, color='tab:orange')
    givewaypoints = ax[2].scatter(time_tau, waypoints[:, 1], marker="o", s=100, c='r', zorder=100)
    ax[2].set_xlabel(r'$\tau$')
    ax[2].set_ylabel(r'$q_2$')
    ax[2].legend([givewaypoints, reproduced], ['Keyframes', 'Reproduced', ], ncol=1, handlelength=1.2)
    ax[2].plot(time_grid, 0 * np.ones_like(time_grid), color='gray', linestyle='--', linewidth=5, )

    ax[2].grid()

    #plot generalized motion
    time_grid_generalized=np.linspace(0,2,20)
    opt_state_traj_generalized_list=[]
    for opt_sol_generalized in opt_sol_generalize_list:
        opt_state_traj_generalized_list.append(opt_sol_generalized(time_grid_generalized)[:,0:4])
    opt_state_traj_generalized_mean=np.mean(np.array(opt_state_traj_generalized_list), axis=0)
    opt_state_traj_generalized_std=np.std(np.array(opt_state_traj_generalized_list), axis=0)

    generalized, = ax[3].plot(time_grid_generalized, opt_state_traj_generalized_mean[:, 0],
                              linewidth=4,
                              color='tab:blue')
    groundtruth, = ax[3].plot(time_grid_generalized, true_opt_sol_genrealize(time_grid_generalized)[:, 0],
                              linewidth=4,
                              color='black', linestyle='--')
    ax[3].legend([generalized, groundtruth], [r'$\theta$ (neural)', r'$\theta^{true}$', ], ncol=1,
                 bbox_to_anchor=(0.41, 0.15, 0.5, 0.5), handlelength=1.2, framealpha=0.1)
    ax[3].set_xlabel(r'$\tau$')
    ax[3].set_ylabel(r'$q_1$')
    ax[3].set_ylim([-1.2, 2])
    ax[3].grid()
    ax[3].plot(time_grid_generalized, np.pi / 2 * np.ones_like(time_grid_generalized), color='gray', linestyle='--',
               linewidth=5, )

    # ax[4].set_facecolor('#E6E6E6')
    ax[4].plot(time_grid_generalized, opt_state_traj_generalized_mean[:, 1], linewidth=4,
               color='tab:blue')
    ax[4].plot(time_grid_generalized, true_opt_sol_genrealize(time_grid_generalized)[:, 1], linewidth=4,
               color='black', linestyle='--')
    ax[4].set_xlabel(r'$\tau$')
    ax[4].set_ylabel(r'$q_2$', labelpad=-3)
    # ax[4].set_ylim([-0.5, 0.25])
    ax[4].grid()
    ax[4].plot(time_grid_generalized, 0 * np.ones_like(time_grid_generalized), color='gray', linestyle='--', linewidth=5, )
    plt.subplots_adjust(left=0.25, right=0.98, bottom=0.06, top=0.99, hspace=0.50)

    print(opt_state_traj_generalized_mean[-1])

    plt.show()

    true_posa = np.array([np.pi / 2, 0, 0, 0])
    print('final state for second case:', opt_state_traj_generalized_mean[-1],
          '|  error to the goal:',
          np.linalg.norm(opt_state_traj_generalized_mean[-1] - true_posa))
    print('final state for true theta:', true_opt_sol_genrealize(time_grid_generalized)[-1, 0:4],
          '|  error to the goal:',
          np.linalg.norm(true_opt_sol_genrealize(time_grid_generalized)[-1, 0:4] - true_posa))


if True:
    # load data
    time_tau=0
    waypoints=0
    opt_sol_list=[]
    opt_sol_generalize_list=[]
    loss_trace_list=[]

    trials=[0,2,3,4,5,6,7,8,9]
    for k in trials:
        load=np.load('./neural/num_waypoints_8_trial_'+str(k)+'.npy', allow_pickle=True).item()
        time_tau=load['time_tau']
        waypoints=load['waypoints']
        opt_sol_list.append(load['opt_sol'])
        opt_sol_generalize_list.append(load['opt_sol_generalize'])
        loss_trace_list.append(load['loss_trace'])


    fig = plt.figure(0, figsize=(3.8, 10))
    ax = fig.subplots(5, 1)

    # plot the loss function
    loss_trace_mean=np.mean(np.array(loss_trace_list), axis=0)
    loss_trace_std=np.std(np.array(loss_trace_list), axis=0)
    xvals=np.arange(len(loss_trace_mean))
    # ax[0].set_facecolor('#E6E6E6')
    ax[0].set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
    ax[0].set_xlabel('Iteration')
    ax[0].plot(loss_trace_mean, linewidth=3, color='tab:brown')
    ax[0].fill_between(xvals, loss_trace_mean - loss_trace_std,
                   loss_trace_mean + loss_trace_std,
                   alpha=0.5, linewidth=0, color='tab:brown')
    ax[0].grid()

    # plot the loss function
    time_grid=np.linspace(0,1,20)
    opt_state_traj_list=[]
    for opt_sol in opt_sol_list:
        opt_state_traj_list.append(opt_sol(time_grid)[:,0:4])
    opt_state_traj_mean=np.mean(np.array(opt_state_traj_list), axis=0)
    opt_state_traj_std=np.std(np.array(opt_state_traj_list), axis=0)



    ax[1].plot(time_grid, opt_state_traj_mean[:, 0], linewidth=4, color='tab:orange')
    ax[1].scatter(time_tau, waypoints[:, 0], marker="o", s=100, c='r', zorder=100)
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'$q_1$')
    ax[1].grid()
    ax[1].plot(time_grid, np.pi / 2 * np.ones_like(time_grid), color='gray', linestyle='--', linewidth=5, )
    # ax[2].set_facecolor('#E6E6E6')

    reproduced, = ax[2].plot(time_grid, opt_state_traj_mean[:, 1], linewidth=4, color='tab:orange')
    givewaypoints = ax[2].scatter(time_tau, waypoints[:, 1], marker="o", s=100, c='r', zorder=100)
    ax[2].set_xlabel(r'$\tau$')
    ax[2].set_ylabel(r'$q_2$')
    ax[2].legend([givewaypoints, reproduced], ['Keyframes', 'Reproduced', ], ncol=1, handlelength=1.2)
    ax[2].plot(time_grid, 0 * np.ones_like(time_grid), color='gray', linestyle='--', linewidth=5, )

    ax[2].grid()

    #plot generalized motion
    time_grid_generalized=np.linspace(0,2,20)
    opt_state_traj_generalized_list=[]
    for opt_sol_generalized in opt_sol_generalize_list:
        opt_state_traj_generalized_list.append(opt_sol_generalized(time_grid_generalized)[:,0:4])
    opt_state_traj_generalized_mean=np.mean(np.array(opt_state_traj_generalized_list), axis=0)
    opt_state_traj_generalized_std=np.std(np.array(opt_state_traj_generalized_list), axis=0)

    generalized, = ax[3].plot(time_grid_generalized, opt_state_traj_generalized_mean[:, 0],
                              linewidth=4,
                              color='tab:blue')
    groundtruth, = ax[3].plot(time_grid_generalized, true_opt_sol_genrealize(time_grid_generalized)[:, 0],
                              linewidth=4,
                              color='black', linestyle='--')
    ax[3].legend([generalized, groundtruth], [r'$\theta$ (neural)', r'$\theta^{true}$', ], ncol=1,
                 bbox_to_anchor=(0.41, 0.15, 0.5, 0.5), handlelength=1.2, framealpha=0.1)
    ax[3].set_xlabel(r'$\tau$')
    ax[3].set_ylabel(r'$q_1$')
    ax[3].set_ylim([-1.2, 2])
    ax[3].grid()
    ax[3].plot(time_grid_generalized, np.pi / 2 * np.ones_like(time_grid_generalized), color='gray', linestyle='--',
               linewidth=5, )

    # ax[4].set_facecolor('#E6E6E6')
    ax[4].plot(time_grid_generalized, opt_state_traj_generalized_mean[:, 1], linewidth=4,
               color='tab:blue')
    ax[4].plot(time_grid_generalized, true_opt_sol_genrealize(time_grid_generalized)[:, 1], linewidth=4,
               color='black', linestyle='--')
    ax[4].set_xlabel(r'$\tau$')
    ax[4].set_ylabel(r'$q_2$', labelpad=-3)
    # ax[4].set_ylim([-0.5, 0.5])
    ax[4].grid()
    ax[4].plot(time_grid_generalized, 0 * np.ones_like(time_grid_generalized), color='gray', linestyle='--', linewidth=5, )
    plt.subplots_adjust(left=0.25, right=0.98, bottom=0.06, top=0.99, hspace=0.50)

    print(opt_state_traj_generalized_mean[-1])

    plt.show()

    true_posa = np.array([np.pi / 2, 0, 0, 0])
    print('final state for thrid case:', opt_state_traj_generalized_mean[-1],
          '|  error to the goal:',
          np.linalg.norm(opt_state_traj_generalized_mean[-1] - true_posa))
    print('final state for true theta:', true_opt_sol_genrealize(time_grid_generalized)[-1, 0:4],
          '|  error to the goal:',
          np.linalg.norm(true_opt_sol_genrealize(time_grid_generalized)[-1, 0:4] - true_posa))
