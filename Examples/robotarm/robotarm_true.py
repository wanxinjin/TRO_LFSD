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
oc.setIntegrator(n_grid=10)

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


# time_tau = true_time_grid[[6, ]]
# time_tau = true_time_grid[[6, 9]]
# time_tau = true_time_grid[[ 3, 6,  9]]
# time_tau = true_time_grid[[1, 3, 6,  9]]
time_tau = true_time_grid[[1, 2, 3, 4, 5, 6, 8, 9]]

waypoints = np.zeros((time_tau.size, interface_pos_fn.numel_out()))
for k, t in enumerate(time_tau):
    waypoints[k, :] = interface_pos_fn(true_opt_sol(t)[0:oc.n_state]).full().flatten()

print(waypoints)

# --------------------------- the learning process ----------------------------------------
lr = 2e-1  # learning rate

max_trial = 10

for trial_num in range(max_trial):

    # initial_parameter = np.array([1, 2.5, 3.5, 2.5, 3.5])  # initial parameter
    initial_parameter = np.random.uniform(1, 4, size=(oc.n_auxvar,))
    loss_trace, parameter_trace = [], []
    current_parameter = initial_parameter
    parameter_trace += [current_parameter.tolist()]

    for j in range(int(1000)):
        parameter_error = norm_2((current_parameter - np.array(true_parameter))) ** 2
        time_grid, opt_sol = oc.cocSolver(ini_state, T, current_parameter)
        auxsys_sol = oc.auxSysSolver(time_grid, opt_sol, current_parameter)
        loss, diff_loss = getloss_pos_corrections(time_tau, waypoints, opt_sol, auxsys_sol)

        current_parameter -= lr * diff_loss
        # do the projection step
        current_parameter[0] = fmax(current_parameter[0], 0.00000001)

        loss_trace += [loss]
        parameter_trace += [current_parameter.tolist()]
        print('trial:', trial_num, 'iter:', j, 'loss:', loss_trace[-1].tolist(), 'error:', parameter_error)

    # save
    if True:
        save_data = {'loss_trace': loss_trace,
                     'parameter_trace': parameter_trace,
                     'true_parameter': true_parameter,
                     'initial_parameter': initial_parameter,
                     'time_tau': time_tau,
                     'waypoints': waypoints,
                     'true_opt_sol': true_opt_sol,
                     'true_time_grid': true_time_grid,
                     'n_state': oc.n_state,
                     'n_control': oc.n_control,
                     'T': T,
                     'lr': lr}
        np.save('./results/true_trial_' + str(trial_num) + 'num_waypoints_'
                + str(time_tau.size) + '.npy', save_data)
