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


if True:
    # load data
    load4 = np.load('./robotarm_neural_4.npy', allow_pickle=True).item()
    time_tau4 = load4['time_tau']
    waypoint4 = load4['waypoints']
    opt_sol4 = load4['opt_sol']
    opt_sol_generalize4 = load4['opt_sol_generalize']
    loss4 = load4['loss_trace']


    fig = plt.figure(0, figsize=(3.8, 10))
    ax = fig.subplots(5, 1)

    # ax[0].set_facecolor('#E6E6E6')
    ax[0].set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
    ax[0].set_xlabel('Iteration')
    ax[0].plot(loss4, linewidth=3, color='tab:brown')
    ax[0].grid()

    # ax[1].set_facecolor('#E6E6E6')
    ax[1].plot(true_time_grid, opt_sol4(true_time_grid)[:, 0], linewidth=4, color='tab:orange')
    ax[1].scatter(time_tau4, waypoint4[:, 0], marker="o", s=100, c='r', zorder=100)
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'$q_1$')
    ax[1].grid()
    ax[1].plot(true_time_grid, np.pi/2 * np.ones_like(true_time_grid), color='gray', linestyle='--',
               linewidth=5, )
    # ax[2].set_facecolor('#E6E6E6')
    reproduced_line4, = ax[2].plot(true_time_grid, opt_sol4(true_time_grid)[:, 1], linewidth=4, color='tab:orange')
    givewaypoints=ax[2].scatter(time_tau4, waypoint4[:, 1], marker="o", s=100, c='r', zorder=100)
    ax[2].legend([givewaypoints,reproduced_line4], ['Keyframes','Reproduced', ], ncol=1, handlelength=1.2)
    ax[2].plot(true_time_grid, 0 * np.ones_like(true_time_grid), color='gray', linestyle='--',
               linewidth=5, )
    ax[2].set_xlabel(r'$\tau$')
    ax[2].set_ylabel(r'$q_2$')
    ax[2].grid()

    # ax[3].set_facecolor('#E6E6E6')
    generalized,=ax[3].plot(true_time_grid_genrealize, opt_sol_generalize4(true_time_grid_genrealize)[:, 0], linewidth=4,
               color='tab:blue')
    groundtruth,=ax[3].plot(true_time_grid_genrealize, true_opt_sol_genrealize(true_time_grid_genrealize)[:, 0], linewidth=4,
               color='black', linestyle='--')
    ax[3].legend([generalized, groundtruth], [r'$\theta$ (neural)', r'$\theta^{true}$', ], ncol=1, bbox_to_anchor=(0.41, 0.15, 0.5, 0.5), handlelength=1.2, framealpha=0.1)
    ax[3].plot(true_time_grid_genrealize,np.pi/2 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )
    ax[3].set_xlabel(r'$\tau$')
    ax[3].set_ylabel(r'$q_1$')
    ax[3].set_ylim([-1.2, 1.8])
    ax[3].grid()

    # ax[4].set_facecolor('#E6E6E6')
    ax[4].plot(true_time_grid_genrealize, opt_sol_generalize4(true_time_grid_genrealize)[:, 1], linewidth=4,
               color='tab:blue')
    ax[4].plot(true_time_grid_genrealize, true_opt_sol_genrealize(true_time_grid_genrealize)[:, 1], linewidth=4,
               color='black', linestyle='--')
    ax[4].set_xlabel(r'$\tau$')
    ax[4].set_ylabel(r'$q_2$', labelpad=-2.5)
    ax[4].plot(true_time_grid_genrealize,0 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )
    ax[4].grid()
    plt.subplots_adjust(left=0.22, right=0.98, bottom=0.06, top=0.99, hspace=0.50)

    # print(opt_sol_generalize4(T_generalize)[0:2])

    plt.show()

    true_posa = np.array([np.pi / 2, 0, 0, 0])
    print('final state for first case:', opt_sol_generalize4(true_time_grid_genrealize)[-1,0:4], '|  error to the goal:',
          np.linalg.norm(opt_sol_generalize4(true_time_grid_genrealize)[-1,0:4] - true_posa))
    print('final state for true theta:', true_opt_sol_genrealize(true_time_grid_genrealize)[-1,0:4], '|  error to the goal:',
          np.linalg.norm(true_opt_sol_genrealize(true_time_grid_genrealize)[-1,0:4] - true_posa))


if True:
    # load data
    load5 = np.load('./robotarm_neural_5.npy', allow_pickle=True).item()
    time_tau5 = load5['time_tau']
    waypoint5 = load5['waypoints']
    opt_sol5 = load5['opt_sol']
    opt_sol_generalize5 = load5['opt_sol_generalize']
    loss5 = load5['loss_trace']

    fig = plt.figure(0, figsize=(3.8, 10))
    ax = fig.subplots(5, 1)

    # ax[0].set_facecolor('#E6E6E6')
    ax[0].set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
    ax[0].set_xlabel('Iteration')
    ax[0].plot(loss5, linewidth=3, color='tab:brown')
    ax[0].grid()

    # ax[1].set_facecolor('#E6E6E6')
    ax[1].plot(true_time_grid, opt_sol5(true_time_grid)[:, 0], linewidth=4, color='tab:orange')
    ax[1].scatter(time_tau5, waypoint5[:, 0], marker="o", s=100, c='r', zorder=100)
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'$q_1$')
    ax[1].grid()
    ax[1].plot(true_time_grid, np.pi / 2 * np.ones_like(true_time_grid), color='gray', linestyle='--',
               linewidth=5, )
    # ax[2].set_facecolor('#E6E6E6')
    reproduced_line5, = ax[2].plot(true_time_grid, opt_sol5(true_time_grid)[:, 1], linewidth=4, color='tab:orange')
    givewaypoints=ax[2].scatter(time_tau5, waypoint5[:, 1], marker="o", s=100, c='r', zorder=100)
    ax[2].set_xlabel(r'$\tau$')
    ax[2].legend([givewaypoints,reproduced_line5], ['Keyframes','Reproduced', ], ncol=1, handlelength=1.2)
    ax[2].set_ylabel(r'$q_2$')
    ax[2].grid()
    ax[2].plot(true_time_grid, 0 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )

    # ax[3].set_facecolor('#E6E6E6')
    generalized,=ax[3].plot(true_time_grid_genrealize, opt_sol_generalize5(true_time_grid_genrealize)[:, 0], linewidth=4,
               color='tab:blue')
    groundtruth,=ax[3].plot(true_time_grid_genrealize, true_opt_sol_genrealize(true_time_grid_genrealize)[:, 0], linewidth=4,
               color='black', linestyle='--')
    ax[3].legend([generalized, groundtruth], [r'$\theta$ (neural)', r'$\theta^{true}$', ], ncol=1, bbox_to_anchor=(0.41, 0.15, 0.5, 0.5), handlelength=1.2, framealpha=0.1)
    ax[3].set_xlabel(r'$\tau$')
    ax[3].set_ylabel(r'$q_1$')
    ax[3].set_ylim([-1.2, 1.8])
    ax[3].grid()
    ax[3].plot(true_time_grid_genrealize, np.pi / 2 * np.ones_like(true_time_grid), color='gray', linestyle='--',
               linewidth=5, )

    # ax[4].set_facecolor('#E6E6E6')
    ax[4].plot(true_time_grid_genrealize, opt_sol_generalize5(true_time_grid_genrealize)[:, 1], linewidth=4,
               color='tab:blue')
    ax[4].plot(true_time_grid_genrealize, true_opt_sol_genrealize(true_time_grid_genrealize)[:, 1], linewidth=4,
               color='black', linestyle='--')
    ax[4].set_xlabel(r'$\tau$')
    ax[4].set_ylabel(r'$q_2$', labelpad=-2.5)
    ax[4].grid()
    ax[4].plot(true_time_grid_genrealize, 0 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )
    plt.subplots_adjust(left=0.22, right=0.98, bottom=0.06, top=0.99, hspace=0.50)

    print(opt_sol_generalize5(T_generalize)[0:2])

    plt.show()

    true_posa = np.array([np.pi / 2, 0, 0, 0])
    print('final state for second case:', opt_sol_generalize5(true_time_grid_genrealize)[-1,0:4], '|  error to the goal:',
          np.linalg.norm(opt_sol_generalize5(true_time_grid_genrealize)[-1,0:4] - true_posa))
    print('final state for true theta:', true_opt_sol_genrealize(true_time_grid_genrealize)[-1,0:4], '|  error to the goal:',
          np.linalg.norm(true_opt_sol_genrealize(true_time_grid_genrealize)[-1,0:4] - true_posa))


if True:
    # load data
    load9 = np.load('./robotarm_neural_9.npy', allow_pickle=True).item()
    time_tau9 = load9['time_tau']
    waypoint9 = load9['waypoints']
    opt_sol9 = load9['opt_sol']
    opt_sol_generalize9 = load9['opt_sol_generalize']
    loss9 = load9['loss_trace']


    fig = plt.figure(0, figsize=(3.8, 10))
    ax = fig.subplots(5, 1)

    # ax[0].set_facecolor('#E6E6E6')
    ax[0].set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
    ax[0].set_xlabel('Iteration')
    ax[0].plot(loss9, linewidth=3, color='tab:brown')
    ax[0].grid()

    # ax[1].set_facecolor('#E6E6E6')
    ax[1].plot(true_time_grid, opt_sol9(true_time_grid)[:, 0], linewidth=4, color='tab:orange')
    ax[1].scatter(time_tau9, waypoint9[:, 0], marker="o", s=100, c='r', zorder=100)
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'$q_1$')
    ax[1].grid()
    ax[1].plot(true_time_grid, np.pi/2* np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )
    # ax[2].set_facecolor('#E6E6E6')
    reproduced_line9, =ax[2].plot(true_time_grid, opt_sol9(true_time_grid)[:, 1], linewidth=4, color='tab:orange')
    givewaypoints=ax[2].scatter(time_tau9, waypoint9[:, 1], marker="o", s=100, c='r', zorder=100)
    ax[2].set_xlabel(r'$\tau$')
    ax[2].set_ylabel(r'$q_2$')
    ax[2].legend([givewaypoints,reproduced_line9], ['Keyframes','Reproduced', ], ncol=1,handlelength=1.2)
    ax[2].plot(true_time_grid, 0 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )

    ax[2].grid()

    # ax[3].set_facecolor('#E6E6E6')
    generalized,=ax[3].plot(true_time_grid_genrealize, opt_sol_generalize9(true_time_grid_genrealize)[:, 0], linewidth=4,
               color='tab:blue')
    groundtruth,=ax[3].plot(true_time_grid_genrealize, true_opt_sol_genrealize(true_time_grid_genrealize)[:, 0], linewidth=4,
               color='black', linestyle='--')
    ax[3].legend([generalized, groundtruth], [r'$\theta$ (neural)', r'$\theta^{true}$', ], ncol=1, bbox_to_anchor=(0.41, 0.15, 0.5, 0.5), handlelength=1.2, framealpha=0.1)
    ax[3].set_xlabel(r'$\tau$')
    ax[3].set_ylabel(r'$q_1$')
    ax[3].set_ylim([-1.2, 1.8])
    ax[3].grid()
    ax[3].plot(true_time_grid_genrealize, np.pi/2 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )

    # ax[4].set_facecolor('#E6E6E6')
    ax[4].plot(true_time_grid_genrealize, opt_sol_generalize9(true_time_grid_genrealize)[:, 1], linewidth=4,
               color='tab:blue')
    ax[4].plot(true_time_grid_genrealize, true_opt_sol_genrealize(true_time_grid_genrealize)[:, 1], linewidth=4,
               color='black', linestyle='--')
    ax[4].set_xlabel(r'$\tau$')
    ax[4].set_ylabel(r'$q_2$', labelpad=-3)
    ax[4].grid()
    ax[4].plot(true_time_grid_genrealize, 0 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )
    plt.subplots_adjust(left=0.21, right=0.98, bottom=0.06, top=0.99, hspace=0.50)

    print(opt_sol_generalize9(T_generalize)[0:2])

    plt.show()

    true_posa = np.array([np.pi / 2, 0, 0, 0])
    print('final state for thrid case:', opt_sol_generalize9(true_time_grid_genrealize)[-1,0:4], '|  error to the goal:',
          np.linalg.norm(opt_sol_generalize9(true_time_grid_genrealize)[-1,0:4] - true_posa))
    print('final state for true theta:', true_opt_sol_genrealize(true_time_grid_genrealize)[-1,0:4], '|  error to the goal:',
          np.linalg.norm(true_opt_sol_genrealize(true_time_grid_genrealize)[-1,0:4] - true_posa))
