
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
load1 = np.load('./robotarm_true_1.npy', allow_pickle=True).item()
loss_trace1 = load1['loss_trace']
parameter_trace1 = np.array(load1['parameter_trace'])
true_parameter1 = np.array(load1['true_parameter'])
init_parameter1 = load1['initial_parameter']
parameter_error1 = np.linalg.norm(parameter_trace1 - true_parameter1, axis=1) ** 2

load2 = np.load('./robotarm_true_2.npy', allow_pickle=True).item()
loss_trace2 = load2['loss_trace']
parameter_trace2 = np.array(load2['parameter_trace'])
true_parameter2 = np.array(load2['true_parameter'])
init_parameter2 = load2['initial_parameter']
parameter_error2 = np.linalg.norm(parameter_trace2 - true_parameter2, axis=1) ** 2

load3 = np.load('./robotarm_true_3.npy', allow_pickle=True).item()
loss_trace3 = load3['loss_trace']
parameter_trace3 = np.array(load3['parameter_trace'])
true_parameter3 = np.array(load3['true_parameter'])
init_parameter3 = load3['initial_parameter']
parameter_error3 = np.linalg.norm(parameter_trace3 - true_parameter3, axis=1) ** 2

load4 = np.load('./robotarm_true_4.npy', allow_pickle=True).item()
loss_trace4 = load4['loss_trace']
parameter_trace4 = np.array(load4['parameter_trace'])
true_parameter4 = np.array(load4['true_parameter'])
init_parameter4 = load4['initial_parameter']
parameter_error4 = np.linalg.norm(parameter_trace4 - true_parameter4, axis=1) ** 2

load8 = np.load('./robotarm_true_8.npy', allow_pickle=True).item()
loss_trace8 = load8['loss_trace']
parameter_trace8 = np.array(load8['parameter_trace'])
true_parameter8 = np.array(load8['true_parameter'])
init_parameter8 = load8['initial_parameter']
parameter_error8 = np.linalg.norm(parameter_trace8 - true_parameter8, axis=1) ** 2
print(parameter_error8[0], parameter_error8[50], parameter_error8[100], parameter_error8[500], parameter_error8[999])

# plot different learning results_1 for different number of sparse demonstrations
if True:
    fig = plt.figure(0, figsize=(10, 4))
    ax = fig.subplots(1, 2)

    # ax[0].set_facecolor('#E6E6E6')
    loss1, = ax[0].plot(loss_trace1, linewidth=3, color='tab:purple')
    loss2, = ax[0].plot(loss_trace2, linewidth=3, color='tab:orange')
    loss3, = ax[0].plot(loss_trace3, linewidth=3,  color='tab:green')
    loss4, = ax[0].plot(loss_trace4, linewidth=3,color='tab:blue')
    loss8, = ax[0].plot(loss_trace8, linewidth=3,color='tab:red')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
    # ax[0].set_xlim([0,20])
    ax[0].set_yscale('log')

    ax[0].grid()

    # ax[1].set_facecolor('#E6E6E6')
    error1, = ax[1].plot(parameter_error1, linewidth=3, color='tab:purple')
    error2, = ax[1].plot(parameter_error2, linewidth=3, color='tab:orange')
    error3, = ax[1].plot(parameter_error3, linewidth=3,  color='tab:green')
    error4, = ax[1].plot(parameter_error4, linewidth=3,color='tab:blue')
    error8, = ax[1].plot(parameter_error8, linewidth=3,color='tab:red')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel(r'$||\theta-\theta^{true}||^2$')
    ax[1].set_yscale('log')
    ax[1].grid()

    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.16, wspace=0.4, top=0.84)
    fig.legend([loss1, loss2, loss3, loss4, loss8],
                 ['1 keyframe', '2 keyframes', '3 keyframes',
                  '4 keyframes', '8 keyframes', ],
               ncol=5,  bbox_to_anchor=(1.00,1.0), handlelength=0.7, columnspacing=0.5, handletextpad=0.4)
    plt.show()

# plot the ground truth trajectory generated from known cost function and time-warping functions
if True:
    true_opt_sol = load8['true_opt_sol']
    true_time_grid = np.linspace(0, 1, 20)
    n_state = load8['n_state']
    state_traj = true_opt_sol(true_time_grid)[:, 0:n_state]
    time_tau = load8['time_tau']
    waypoints = load8['waypoints']
    np.set_printoptions(precision=3)
    print(time_tau)
    print(waypoints)

    fig = plt.figure(0, figsize=(10, 4))
    ax = fig.subplots(1, 2)

    # ax[0].set_facecolor('#E6E6E6')
    ax[0].plot(true_time_grid, state_traj[:, 0], linewidth=5, color='black')

    ax[0].set_xlabel(r'$\tau$')
    ax[0].set_ylabel(r'$q_1$')
    ax[0].scatter(time_tau, waypoints[:, 0], marker="o", s=100, c='r', zorder=100)
    ax[0].grid()
    ax[0].plot(true_time_grid, np.pi / 2 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )
    ax[0].text(1.1, np.pi / 2 - 0.15, '$q^{g}_1$', fontsize=22)

    # ax[1].set_facecolor('#E6E6E6')
    ax[1].plot(true_time_grid, state_traj[:, 1], linewidth=5, color='black')
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'$q_2$')
    ax[1].scatter(time_tau, waypoints[:, 1], marker="o", s=100, c='r', zorder=100)
    ax[1].grid()
    ax[1].plot(true_time_grid, 0 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )
    ax[1].text(1.1, 0-0.008, '$q^{g}_2$', fontsize=22)

    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.15, wspace=0.4, top=0.84)
    # plt.subplots_adjust(left=0.1, right=0.95, bottom=0.2, wspace=0.3, top=0.85)
    plt.suptitle('Generation of Keyframes', fontsize=20)
    plt.show()

# plot the generalization of learned control cost function and compare with the ground truth.
if True:
    # new initial condition
    ini_state = [-np.pi/4, 0, 0, 0]
    # ini_state = [-np.pi / 2, 3 * np.pi / 4, -5, 3]

    T = 2
    true_time_grid = np.linspace(0, T, 20)
    n_state = load8['n_state']


    # ground truth
    true_parameter = [5, 3, 3, 3, 3]
    _, true_opt_sol = oc.cocSolver(ini_state, T, true_parameter)
    state_traj = true_opt_sol(true_time_grid)[:, 0:n_state]

    # for 1 waypoints
    _, true_opt_sol1 = oc.cocSolver(ini_state, T, parameter_trace1[-1])
    state_traj1 = true_opt_sol1(true_time_grid)[:, 0:n_state]

    # for 2 waypoints
    _, true_opt_sol2 = oc.cocSolver(ini_state, T, parameter_trace2[-1])
    state_traj2 = true_opt_sol2(true_time_grid)[:, 0:n_state]

    # for 3 waypoints
    _, true_opt_sol3 = oc.cocSolver(ini_state, T, parameter_trace3[-1])
    state_traj3 = true_opt_sol3(true_time_grid)[:, 0:n_state]

    # for 4 waypoints
    _, true_opt_sol4 = oc.cocSolver(ini_state, T, parameter_trace4[-1])
    state_traj4 = true_opt_sol4(true_time_grid)[:, 0:n_state]

    # for 8 waypoints
    _, true_opt_sol8 = oc.cocSolver(ini_state, T, parameter_trace8[-1])
    state_traj8 = true_opt_sol8(true_time_grid)[:, 0:n_state]


    fig = plt.figure(0, figsize=(10, 4))
    ax = fig.subplots(1, 2)

    # ax[0].set_facecolor('#E6E6E6')
    line_true,=ax[0].plot(true_time_grid, state_traj[:, 0], linewidth=6, color='black',linestyle='--')
    line_1,=ax[0].plot(true_time_grid, state_traj1[:, 0], linewidth=3, color='tab:purple')
    line_2,=ax[0].plot(true_time_grid, state_traj2[:, 0], linewidth=3, color='tab:orange')
    line_3,=ax[0].plot(true_time_grid, state_traj3[:, 0], linewidth=3,color='tab:green')
    line_4,=ax[0].plot(true_time_grid, state_traj4[:, 0], linewidth=4, color='tab:blue')
    line_8,=ax[0].plot(true_time_grid, state_traj8[:, 0], linewidth=3, color='tab:red')
    ax[0].plot(true_time_grid,np.pi/2 * np.ones_like(true_time_grid), color='gray', linestyle='--', linewidth=5, )
    ax[0].text(2.15, np.pi/2-0.1, '$q^{g}_1$', fontsize=22)
    ax[0].set_xlabel(r'$\tau$')
    ax[0].set_ylabel(r'$q_1$')
    ax[0].grid()

    # ax[1].set_facecolor('#E6E6E6')
    ax[1].plot(true_time_grid, state_traj[:, 1], linewidth=6, color='black',linestyle='--')
    ax[1].plot(true_time_grid, state_traj1[:, 1], linewidth=3, color='tab:purple')
    ax[1].plot(true_time_grid, state_traj2[:, 1], linewidth=3, color='tab:orange')
    ax[1].plot(true_time_grid, state_traj3[:, 1], linewidth=3,color='tab:green')
    ax[1].plot(true_time_grid, state_traj4[:, 1], linewidth=4, color='tab:blue')
    ax[1].plot(true_time_grid, state_traj8[:, 1], linewidth=3, color='tab:red')
    ax[1].plot(true_time_grid,0*np.ones_like(true_time_grid), color='gray',linestyle='--', linewidth=5,)
    ax[1].text(2.15, 0-0.005, '$q^{g}_2$', fontsize=22)
    ax[1].set_xlabel(r'$\tau$')
    ax[1].set_ylabel(r'$q_2$')
    ax[1].grid()

    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.15, wspace=0.4, top=0.84)
    fig.legend([line_1, line_2, line_3, line_4, line_8, line_true,],
                 ['1 keyframe', '2 keyframes', '3 keyframes',
                  '4 keyframes', '8 keyframes', ],
               ncol=5,  bbox_to_anchor=(1.00,1.0), handlelength=0.7, columnspacing=0.5, handletextpad=0.4)
    plt.show()

#   print the final pose in each learned case
    true_posa=np.array([np.pi/2, 0, 0, 0])
    print('final state for 1 keyframes:', state_traj1[-1], '|  error to the goal:', np.linalg.norm(state_traj1[-1]-true_posa))
    print('final state for 2 keyframes:', state_traj2[-1], '|  error to the goal:', np.linalg.norm(state_traj2[-1]-true_posa))
    print('final state for 3 keyframes:', state_traj3[-1], '|  error to the goal:', np.linalg.norm(state_traj3[-1]-true_posa))
    print('final state for 4 keyframes:', state_traj4[-1], '|  error to the goal:', np.linalg.norm(state_traj4[-1]-true_posa))
    print('final state for 8 keyframes:', state_traj8[-1], '|  error to the goal:', np.linalg.norm(state_traj8[-1]-true_posa))
    print('final state for true theta:', state_traj[-1], '|  error to the goal:', np.linalg.norm(state_traj[-1]-true_posa))
