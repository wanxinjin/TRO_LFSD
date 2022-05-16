import casadi
import numpy as np

from CPDP import CPDP
from CPDP import PDP
from JinEnv import JinEnv
from casadi import *
from scipy.integrate import solve_ivp
import scipy.io as sio
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# set the plotting parameters
params = {'axes.labelsize': 20,
          'axes.titlesize': 16,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16,
          'legend.fontsize': 16}
plt.rcParams.update(params)

# plot the results plain ipopt
if True:
    load1 = np.load('./compare_plain_ipopt_1.npy', allow_pickle=True).item()
    loss_trace1 = load1['loss_trace']
    waypoints = load1['waypoints']
    time_tau = load1['time_tau']
    parameter_trace1 = np.array(load1['parameter_trace'])
    true_parameter = np.array(load1['true_parameter'])
    parameter_error1 = np.linalg.norm(parameter_trace1 - true_parameter, axis=1) ** 2
    time_grid = load1['time_grid']
    opt_sol1 = load1['opt_sol']

    load2 = np.load('./compare_plain_ipopt_2.npy', allow_pickle=True).item()
    loss_trace2 = load2['loss_trace']
    parameter_trace2 = np.array(load2['parameter_trace'])
    parameter_error2 = np.linalg.norm(parameter_trace2 - true_parameter, axis=1) ** 2
    opt_sol2 = load2['opt_sol']

    load3 = np.load('./compare_plain_ipopt_3.npy', allow_pickle=True).item()
    loss_trace3 = load3['loss_trace']
    parameter_trace3 = np.array(load3['parameter_trace'])
    parameter_error3 = np.linalg.norm(parameter_trace3 - true_parameter, axis=1) ** 2
    opt_sol3 = load3['opt_sol']

    fig = plt.figure(1, figsize=(5, 10))
    ax = fig.subplots(4, 1)

    # ax[0].set_facecolor('#E6E6E6')
    loss_line1, = ax[0].plot(loss_trace1, linewidth=3, color='tab:green')
    loss_line2, = ax[0].plot(loss_trace2, linewidth=3, color='tab:red')
    loss_line3, = ax[0].plot(loss_trace3, linewidth=3, color='tab:blue')
    ax[0].legend([loss_line1, loss_line2, loss_line3, ], ['Trial 1', 'Trial 2', 'Trial 3', ], handlelength=1.5)
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')

    ax[0].grid()

    # ax[1].set_facecolor('#E6E6E6')
    error_line1, = ax[1].plot(parameter_error1, linewidth=3, color='tab:green')
    error_line2, = ax[1].plot(parameter_error2, linewidth=3, color='tab:red')
    error_line3, = ax[1].plot(parameter_error3, linewidth=3, color='tab:blue')
    ax[1].legend([error_line1, error_line2, error_line3], ['Trial 1', 'Trial 2', 'Trial 3'], loc='upper left', handlelength=1.5)
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel(r'$||\theta-\theta^{true}||^2$',)
    ax[1].set_yscale('log')
    ax[1].grid()


    # ax[2].set_facecolor('#E6E6E6')
    lineq1_1,=ax[2].plot(time_grid, opt_sol1['state_traj_opt'][:, 0], linewidth=3, color='tab:green')
    lineq1_2,=ax[2].plot(time_grid, opt_sol2['state_traj_opt'][:, 0], linewidth=3, color='tab:red')
    lineq1_3,=ax[2].plot(time_grid, opt_sol3['state_traj_opt'][:, 0], linewidth=3, color='tab:blue')
    ax[2].legend([lineq1_1, lineq1_2, lineq1_3], ['Trial 1', 'Trial 2', 'Trial 3'], loc='lower left', handlelength=1.5, ncol=2)
    ax[2].scatter(time_tau, waypoints[:, 0], marker="o", s=100, c='r', zorder=100)
    ax[2].set_xlabel(r'$\tau$')
    ax[2].set_ylabel(r'$q_1$', labelpad=10)
    ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax[2].grid()
    ax[2].plot(time_grid, np.pi/2 * np.ones_like(time_grid), color='gray', linestyle='--', linewidth=5, zorder=-100)


    # ax[3].set_facecolor('#E6E6E6')
    lineq2_1,=ax[3].plot(time_grid, opt_sol1['state_traj_opt'][:, 1], linewidth=3, color='tab:green')
    lineq2_2,=ax[3].plot(time_grid, opt_sol2['state_traj_opt'][:, 1], linewidth=3, color='tab:red')
    lineq2_3,=ax[3].plot(time_grid, opt_sol3['state_traj_opt'][:, 1], linewidth=3, color='tab:blue')
    ax[3].legend([lineq2_1, lineq2_2, lineq2_3], ['Trial 1', 'Trial 2', 'Trial 3'], loc='lower left', handlelength=1.5, ncol=2)

    ax[3].scatter(time_tau, waypoints[:, 1], marker="o", s=100, c='r', zorder=100)
    ax[3].set_xlabel(r'$\tau$')
    ax[3].set_ylabel(r'$q_2$', labelpad=10)
    ax[3].grid()
    ax[3].plot(time_grid, 0 * np.ones_like(time_grid), color='gray', linestyle='--', linewidth=5, zorder=-100)

    plt.subplots_adjust(left=0.20, right=0.98, bottom=0.07, top=0.99, hspace=0.60, )
    plt.show()

if True:
    load1 = np.load('./compare_bilevel_1.npy', allow_pickle=True).item()
    loss_trace1 = load1['loss_trace']
    waypoints = load1['waypoints']
    time_tau = load1['time_tau']
    parameter_trace1 = np.array(load1['parameter_trace'])
    true_parameter = np.array(load1['true_parameter'])
    parameter_error1 = np.linalg.norm(parameter_trace1 - true_parameter, axis=1) ** 2
    time_grid = np.linspace(0, 1, 20)
    opt_sol1 = load1['opt_sol']

    load2 = np.load('./compare_bilevel_2.npy', allow_pickle=True).item()
    loss_trace2 = load2['loss_trace']
    parameter_trace2 = np.array(load2['parameter_trace'])
    parameter_error2 = np.linalg.norm(parameter_trace2 - true_parameter, axis=1) ** 2
    opt_sol2 = load2['opt_sol']

    load3 = np.load('./compare_bilevel_3.npy', allow_pickle=True).item()
    loss_trace3 = load3['loss_trace']
    parameter_trace3 = np.array(load3['parameter_trace'])
    parameter_error3 = np.linalg.norm(parameter_trace2 - true_parameter, axis=1) ** 2
    opt_sol3 = load3['opt_sol']

    fig = plt.figure(2, figsize=(5, 10))
    ax = fig.subplots(4, 1)

    # ax[0].set_facecolor('#E6E6E6')
    loss_line1, = ax[0].plot(loss_trace1, linewidth=3, color='tab:green')
    loss_line2, = ax[0].plot(loss_trace2, linewidth=3, color='tab:red')
    loss_line3, = ax[0].plot(loss_trace3, linewidth=3, color='tab:blue')
    ax[0].legend([loss_line1, loss_line2, loss_line3, ], ['Trial 1', 'Trial 2', 'Trial 3'], handlelength=1.5)
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel(r'$L(\xi_\theta,\mathcal{D})$')
    ax[0].set_xlim([-10,200])
    ax[0].set_yscale('log')
    ax[0].grid()

    # ax[1].set_facecolor('#E6E6E6')
    error_line1, = ax[1].plot(parameter_error1, linewidth=3, color='tab:green')
    error_line2, = ax[1].plot(parameter_error2, linewidth=3, color='tab:red')
    error_line3, = ax[1].plot(parameter_error3, linewidth=3, color='tab:blue')
    ax[1].legend([error_line1, error_line2, error_line3, ], ['Trial 1', 'Trial 2', 'Trial 3'], handlelength=1.5)
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel(r'$||\theta-\theta^{true}||^2$')
    ax[1].set_yscale('log')
    ax[1].grid()

    # ax[2].set_facecolor('#E6E6E6')
    q1_line1,=ax[2].plot(time_grid, opt_sol1(time_grid)[:, 0], linewidth=3, color='tab:green')
    q1_line2,=ax[2].plot(time_grid, opt_sol2(time_grid)[:, 0], linewidth=3, color='tab:red')
    q1_line3,=ax[2].plot(time_grid, opt_sol3(time_grid)[:, 0], linewidth=3, color='tab:blue')
    ax[2].legend([q1_line1, q1_line2, q1_line3, ], ['Trial 1', 'Trial 2', 'Trial 3'], handlelength=1.5)
    ax[2].scatter(time_tau, waypoints[:, 0], marker="o", s=100, c='r', zorder=100)
    ax[2].set_xlabel(r'$\tau$')
    ax[2].set_ylabel(r'$q_1$', labelpad=0)
    ax[2].grid()
    ax[2].plot(time_grid, np.pi/2 * np.ones_like(time_grid), color='gray', linestyle='--', linewidth=5, zorder=-100)

    # ax[3].set_facecolor('#E6E6E6')
    q2_line1,=ax[3].plot(time_grid, opt_sol1(time_grid)[:, 1], linewidth=3, color='tab:green')
    q2_line2,=ax[3].plot(time_grid, opt_sol2(time_grid)[:, 1], linewidth=3, color='tab:red')
    q2_line3,=ax[3].plot(time_grid, opt_sol3(time_grid)[:, 1], linewidth=3, color='tab:blue')
    ax[3].legend([q2_line1, q2_line2, q2_line3, ], ['Trial 1', 'Trial 2', 'Trial 3'], handlelength=1.5)
    ax[3].scatter(time_tau, waypoints[:, 1], marker="o", s=100, c='r', zorder=100)
    ax[3].set_xlabel(r'$\tau$')
    ax[3].set_ylabel(r'$q_2$', labelpad=15)
    ax[3].grid()
    ax[3].plot(time_grid, 0 * np.ones_like(time_grid), color='gray', linestyle='--', linewidth=5, zorder=-100)

    plt.subplots_adjust(left=0.21, right=0.95, bottom=0.07, top=0.99, hspace=0.60, )
    plt.show()
