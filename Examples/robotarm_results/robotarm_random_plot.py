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

# load date
load = np.load('./robotarm_random.npy', allow_pickle=True).item()
loss_trace = load['loss_trace']
parameter_trace = np.array(load['parameter_trace'])
print('learned', parameter_trace[-1])
true_parameter = np.array(load['true_parameter'])
init_parameter = load['initial_parameter']
parameter_error = np.linalg.norm(parameter_trace - true_parameter, axis=1) ** 2
true_opt_sol = load['true_opt_sol']
true_time_grid = np.linspace(0, 1, 20)
n_state = load['n_state']
state_traj = true_opt_sol(true_time_grid)[:, 0:n_state]
time_tau = load['time_tau']
waypoints = load['waypoints']
T = load['T']

# plot the selection of the sparse demos
if False:
    fig = plt.figure(0, figsize=(10, 4))
    ax = fig.subplots(1, 2)

    # ax[0].set_facecolor('#E6E6E6')
    ax[0].plot(true_time_grid, state_traj[:, 0], linewidth=3)
    ax[0].set_xlabel(r'$\tau$')
    ax[0].set_ylabel(r'$q_1$')
    ax[0].scatter(time_tau, waypoints[:, 0], marker="o", s=100, c='r', zorder=100)
    ax[0].grid()

    # ax[1].set_facecolor('#E6E6E6')
    ax[1].plot(true_time_grid, state_traj[:, 1], linewidth=3)
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'$q_2$')
    ax[1].scatter(time_tau, waypoints[:, 1], marker="o", s=100, c='r', zorder=100)
    ax[1].grid()

    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.2, wspace=0.3, top=0.85)
    plt.suptitle('Selection of sparse waypoints', fontsize=20)
    plt.show()

# plot the loss trace
if False:

    fig = plt.figure(0, figsize=(8, 6))
    ax = fig.subplots(2, 2)

    ax[0, 0].set_facecolor('#E6E6E6')
    line_true, = ax[0, 0].plot(true_time_grid, state_traj[:, 0], linewidth=4, color='black', linestyle='-')
    points = ax[0, 0].scatter(time_tau, waypoints[:, 0], marker="o", s=100, c='r', zorder=100)
    ax[0, 0].set_ylabel(r'$q_1$')
    ax[0, 0].set_xlabel(r'$\tau$')
    # ax[0, 0].axes.xaxis.set_ticklabels([])
    ax[0, 0].grid()

    ax[0, 1].set_facecolor('#E6E6E6')
    ax[0, 1].plot(true_time_grid, state_traj[:, 1], linewidth=4, color='black', linestyle='-')
    ax[0, 1].scatter(time_tau, waypoints[:, 1], marker="o", s=100, c='r', zorder=100)
    ax[0, 1].set_xlabel(r'$\tau$')
    ax[0, 1].set_ylabel(r'$q_2$')
    ax[0, 1].grid()
    # ax[0, 1].legend([line_true, points],
    #                 [r'Trajectory by $\theta^*$', 'Selected waypoints'],
    #                 ncol=1, loc='upper right',  columnspacing=1.5)

    ax[1, 0].set_facecolor('#E6E6E6')
    ax[1, 0].plot(loss_trace, linewidth=4, color='tab:blue', linestyle='-')
    ax[1, 0].set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
    ax[1, 0].set_xlabel('Iteration')
    ax[1, 0].grid()

    ax[1, 1].set_facecolor('#E6E6E6')
    ax[1, 1].plot(parameter_error,  linewidth=4, color='tab:blue', linestyle='-')
    ax[1, 1].set_xlabel('Iteration')
    ax[1, 1].set_ylabel(r'$||\theta-\theta^*||^2$')
    ax[1, 1].grid()

    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.12, wspace=0.32, top=0.98, hspace=0.35)

    plt.show()

# plot the regenerated trace
if True:
    ini_state = [-np.pi / 2, 3 * np.pi / 4, -5, 3]
    true_time_grid = np.linspace(0, T, 20)

    # ground truth
    true_parameter = [5, 3, 3, 3, 3]
    _, true_opt_sol = oc.cocSolver(ini_state, T, true_parameter)
    state_traj = true_opt_sol(true_time_grid)[:, 0:n_state]

    # for 1 waypoints
    _, opt_sol = oc.cocSolver(ini_state, T, parameter_trace[-1])
    regenerated_state_traj = opt_sol(true_time_grid)[:, 0:n_state]
    print(parameter_trace[-1])

    fig = plt.figure(0, figsize=(10, 4))
    ax = fig.subplots(2, 3)

    # ax[0, 0].set_facecolor('#E6E6E6')
    line_true,=ax[0, 0].plot(true_time_grid, state_traj[:, 0], linewidth=4, color='black', linestyle='-')
    ax[0, 0].scatter(time_tau, waypoints[:, 0], marker="o", s=100, c='r', zorder=100)
    reproduced_line,=ax[0, 0].plot(true_time_grid, regenerated_state_traj[:, 0], linewidth=4, color='tab:orange', linestyle='-')
    ax[0, 0].set_ylabel(r'$q_1$')

    ax[0, 0].set_ylim([-5.2, 2.5])
    ax[0, 0].grid()
    ax[0, 0].legend([line_true, reproduced_line],
                    [r'$\theta^{true}}$', r'Learned $\theta$'],
                    ncol=1, loc='lower right',  columnspacing=.5, fontsize=13, handlelength=1)
    ax[0,0].plot(true_time_grid,np.pi/2 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )
    # ax[0, 0].text(1.15, np.pi/2-0.1, '$q^{g}_1$', fontsize=22)




    # ax[1, 0].set_facecolor('#E6E6E6')
    ax[1, 0].plot(true_time_grid, state_traj[:, 1], linewidth=4, color='black', linestyle='-')
    ax[1, 0].scatter(time_tau, waypoints[:, 1], marker="o", s=100, c='r', zorder=100)
    ax[1, 0].plot(true_time_grid, regenerated_state_traj[:, 1], linewidth=4, color='tab:orange', linestyle='-')
    ax[1, 0].set_xlabel(r'$\tau$')
    ax[1, 0].set_ylabel(r'$q_2$')
    ax[1,0].set_ylim([-0.5, 3.2])
    ax[1, 0].grid()
    ax[1, 0].plot(true_time_grid, 0 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )




    # ax[0, 1].set_facecolor('#E6E6E6')
    ax[0, 1].plot(loss_trace, linewidth=4, color='tab:brown', linestyle='-')
    ax[0, 1].set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
    ax[0, 1].set_ylim([-0.5, 10])
    ax[0, 1].grid()


    # ax[1, 1].set_facecolor('#E6E6E6')
    ax[1, 1].plot(parameter_error,  linewidth=4, color='tab:brown', linestyle='-')
    ax[1, 1].set_xlabel('Iteration')
    ax[1, 1].set_ylabel(r'$||\theta-\theta^{true}||^2$')
    ax[1, 1].grid()
    print(loss_trace[-1])








    ini_state = [-np.pi / 4, 0, 0, 0]

    T = 2
    true_time_grid = np.linspace(0, T, 20)
    # ground truth
    true_parameter = [5, 3, 3, 3, 3]
    _, true_opt_sol = oc.cocSolver(ini_state, T, true_parameter)
    state_traj = true_opt_sol(true_time_grid)[:, 0:n_state]
    # for 1 waypoints
    _, opt_sol = oc.cocSolver(ini_state, T, parameter_trace[-1])
    regenerated_state_traj = opt_sol(true_time_grid)[:, 0:n_state]



    # ax[0, 2].set_facecolor('#E6E6E6')
    true_line,=ax[0, 2].plot(true_time_grid, state_traj[:, 0], linewidth=6, color='black', linestyle='--')
    generated_line,=ax[0, 2].plot(true_time_grid, regenerated_state_traj[:, 0], linewidth=5, color='tab:blue', linestyle='-')

    ax[0, 2].set_ylabel(r'$q_2$')
    ax[0, 2].grid()
    ax[0, 2].legend([true_line, generated_line],
                    [r'$\theta^{true}$', r'learned $\theta$'],
                    ncol=1, loc='lower right',  columnspacing=1.5, fontsize=13,handlelength=1)
    ax[0, 2].plot(true_time_grid, np.pi / 2 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )


    # ax[1, 2].set_facecolor('#E6E6E6')
    ax[1, 2].plot(true_time_grid, state_traj[:, 1], linewidth=6, color='black', linestyle='--')
    ax[1, 2].plot(true_time_grid, regenerated_state_traj[:, 1], linewidth=4, color='tab:blue', linestyle='-')
    ax[1, 2].set_xlabel(r'$\tau$')
    ax[1, 2].set_ylabel(r'$q_2$')
    ax[1, 2].grid()
    ax[1,2].plot(true_time_grid,0 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )

    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.15, wspace=0.48, top=0.98, hspace=0.30)
    # plt.tight_layout()
    plt.show()

    true_posa = np.array([np.pi / 2, 0, 0, 0])
    print('final state:', regenerated_state_traj[-1], '|  error to the goal:',
          np.linalg.norm(regenerated_state_traj[-1] - true_posa))
    print('final state for true theta:', state_traj[-1], '|  error to the goal:',
          np.linalg.norm(state_traj[-1] - true_posa))