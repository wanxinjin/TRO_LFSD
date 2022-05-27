import numpy as np

from CPDP import CPDP
from JinEnv import JinEnv
from casadi import *
from scipy.integrate import solve_ivp
import scipy.io as sio
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

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
# env.play_animation(l1=1, l2=1, dt=true_time_grid[1] - true_time_grid[0], state_traj=true_opt_sol(true_time_grid)[:, 0:oc.n_state])


time_tau = true_time_grid[[1, 3, 4, 5, 7,  9, 12, 14]]
waypoints = np.zeros((time_tau.size, interface_pos_fn.numel_out()))
for k, t in enumerate(time_tau):
    waypoints[k, :] = interface_pos_fn(true_opt_sol(t)[0:oc.n_state]).full().flatten()

# --------------------------- learn a spline function ----------------------------------------
waypoints_q1 = waypoints[:, 0]
waypoints_q2 = waypoints[:, 1]
spl_q1 = UnivariateSpline(time_tau, waypoints_q1, k=3)
spl_q2 = UnivariateSpline(time_tau, waypoints_q2, k=3)
plot_time = np.linspace(0, 1, 50)


# save


# save
if True:
    save_data = {'waypoints_q1': waypoints_q1,
                 'waypoints_q2': waypoints_q2,
                 'spl_q1': spl_q1,
                 'spl_q2': spl_q2,
                 'plot_time': plot_time,
                 'time_tau': time_tau,
                 'waypoints': waypoints,
                 'true_opt_sol': true_opt_sol,
                 'true_time_grid': true_time_grid,
                 'n_state': oc.n_state,
                 'n_control': oc.n_control,
                 'T': T}
    np.save('./results_data/keyframes/compare_keyframe.npy', save_data)



