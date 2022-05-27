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


# --------------------------- give the sparse demonstration ----------------------------------------
ini_state = [-pi / 2, 3 * pi / 4, -5, 3]
T = 1
true_parameter = [5, 3, 3, 3, 3]
true_time_grid, true_opt_sol = oc.cocSolver(ini_state, T, true_parameter)


time_tau = true_time_grid[[1, 3, 4, 5, 7,  9, 12, 14]]
waypoints = np.zeros((time_tau.size, interface_pos_fn.numel_out()))
for k, t in enumerate(time_tau):
    waypoints[k, :] = interface_pos_fn(true_opt_sol(t)[0:oc.n_state]).full().flatten()

import matplotlib.pyplot as plt

# set the plotting parameters
params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'legend.fontsize': 18}
plt.rcParams.update(params)


time_grid = np.linspace(0, 1, 15)
state_traj = true_opt_sol(time_grid)[:, 0:oc.n_state]
np.set_printoptions(precision=3)
print(time_tau)
print(waypoints)

fig = plt.figure(0, figsize=(10, 4))
ax = fig.subplots(1, 2)

# ax[0].set_facecolor('#E6E6E6')
ax[0].plot(time_grid, state_traj[:, 0], linewidth=5, color='black')

ax[0].set_xlabel(r'$\tau$')
ax[0].set_ylabel(r'$q_1$')
ax[0].scatter(time_tau, waypoints[:, 0], marker="o", s=100, c='r', zorder=100)
ax[0].grid()
ax[0].plot(time_grid, np.pi / 2 * np.ones_like(time_grid), color='gray', linestyle='--', linewidth=5, )
ax[0].text(1.1, np.pi / 2 - 0.15, '$q^{g}_1$', fontsize=22)

# ax[1].set_facecolor('#E6E6E6')
ax[1].plot(time_grid, state_traj[:, 1], linewidth=5, color='black')
ax[1].set_xlabel(r'$\tau$')
ax[1].set_ylabel(r'$q_2$')
ax[1].scatter(time_tau, waypoints[:, 1], marker="o", s=100, c='r', zorder=100)
ax[1].grid()
ax[1].plot(time_grid, 0 * np.ones_like(time_grid), color='gray', linestyle='--', linewidth=5, )
ax[1].text(1.1, 0-0.008, '$q^{g}_2$', fontsize=22)

plt.subplots_adjust(left=0.08, right=0.95, bottom=0.15, wspace=0.4, top=0.84)
# plt.subplots_adjust(left=0.1, right=0.95, bottom=0.2, wspace=0.3, top=0.85)
plt.suptitle('Generation of keyframes', fontsize=20)
plt.show()
