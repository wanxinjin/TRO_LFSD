from CPDP import CPDP
from CPDP import CPDP_Ex
from JinEnv import JinEnv
from casadi import *
from scipy.integrate import solve_ivp
import scipy.io as sio

# ---------------------------------------load environment---------------------------------------
env = JinEnv.RobotArm()
env.initDyn(l1=1, m1=2, l2=1, m2=1, g=0)
env.initCost_WeightedDistance(wu=0.5)

# --------------------------- create PDP object ----------------------------------------
oc = CPDP_Ex.COCSys_Ex()
oc.setStateVariable(env.X)
oc.setControlVariable(env.U)
t = SX.sym('t')
oc.setTimeVariable(t)

# beta1 = SX.sym('beta1')
# oc.setAuxvarVariable(vcat([beta1,  env.cost_auxvar]))
# v = beta1
# tw_order = 1


# beta1 = SX.sym('beta1')
# beta2 = SX.sym('beta2')
# oc.setAuxvarVariable(vcat([beta1, beta2, env.cost_auxvar]))
# v = beta1 + 2 * beta2 * t
# tw_order = 2

# beta1 = SX.sym('beta1')
# beta2 = SX.sym('beta2')
# beta3 = SX.sym('beta3')
# oc.setAuxvarVariable(vcat([beta1, beta2, beta3, env.cost_auxvar]))
# v = beta1 + 2 * beta2 * t + 3 * beta3 * t ** 2
# tw_order = 3

beta1 = SX.sym('beta1')
beta2 = SX.sym('beta2')
beta3 = SX.sym('beta3')
beta4 = SX.sym('beta4')
oc.setAuxvarVariable(vcat([beta1, beta2, beta3, beta4, env.cost_auxvar]))
v = beta1 + 2 * beta2 * t + 3 * beta3 * t ** 2 + 4 * beta4 * t ** 3
tw_order = 4

dyn = v * env.f
oc.setDyn(dyn)
path_cost = v * env.path_cost
oc.setPathCost(path_cost)
oc.setFinalCost(env.final_cost)
oc.setIntegrator(n_grid=25)

# --------------------------- create way points ----------------------------------------

time_tau = np.array([0.2, 0.33333333, 0.46666667, 0.6, 0.8, 0.93333333])
waypoints = np.array([[-2., 2.5],
                      [-2., 2.],
                      [-1., 1.],
                      [-1., 1.],
                      [0., 1.],
                      [0., 1.]])
ini_state = [-pi / 2, 3 * pi / 4, -5, 3]
T = 1

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


# --------------------------- the learning process ----------------------------------------
max_trial = 10
for trial in range(5, max_trial):
    lr = 0.8e-1  # learning rate
    # initial_parameter = np.array([1., 6, 3, 3, 6.])  # initial parameter
    # initial_parameter = np.array([1., 0, 6, 3, 3, 6.])  # initial parameter
    # initial_parameter = np.array([1., 0, 0,  6, 3, 3, 6.])  # initial parameter
    # initial_parameter = np.array([1., 0, 0, 0, 6, 3, 3, 6.])  # initial parameter
    initial_parameter = np.random.uniform(2, 3, size=(oc.n_auxvar,))
    loss_trace, parameter_trace = [], []
    current_parameter = initial_parameter
    parameter_trace += [current_parameter.tolist()]
    for j in range(int(200)):
        time_grid, opt_sol = oc.cocSolver(ini_state, T, current_parameter)
        auxsys_sol = oc.auxSysSolver(time_grid, opt_sol, current_parameter)
        loss, diff_loss = getloss_pos_corrections(time_tau, waypoints, opt_sol, auxsys_sol)

        current_parameter -= lr * diff_loss
        # do the projection step
        current_parameter[0] = fmax(current_parameter[0], 0.00000001)

        loss_trace += [loss]
        parameter_trace += [current_parameter.tolist()]
        print('tw_order:', tw_order, 'trial:', trial, 'iter:', j, 'loss:', loss_trace[-1].tolist())

    # save
    time_grid, opt_sol = oc.cocSolver(ini_state, T, current_parameter)
    if True:
        save_data = {'loss_trace': loss_trace,
                     'parameter_trace': parameter_trace,
                     'initial_parameter': initial_parameter,
                     'time_tau': time_tau,
                     'waypoints': waypoints,
                     'n_state': oc.n_state,
                     'n_control': oc.n_control,
                     'time_grid': time_grid,
                     'opt_sol': opt_sol,
                     'tw_order': tw_order,
                     'T': T,
                     'lr': lr}
        np.save('./results_data/time/tworder_' + str(tw_order) + '_trial_' + str(trial) + '.npy', save_data)
