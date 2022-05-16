import numpy as np

from CPDP import CPDP
from JinEnv import JinEnv
from casadi import *
from scipy.integrate import solve_ivp
import scipy.io as sio
import time

# ---------------------------------------load environment---------------------------------------
env = JinEnv.RobotArm()
env.initDyn(l1=1, m1=2, l2=1, m2=1, g=0)
env.initCost_WeightedDistance(wu=0.5)

# ------------------------------ define the neural network objective function------------------------
n_layer1 = 4
n_layer2 = 8
net_para = []

A = env.X
M1 = SX.sym('M1', n_layer2, n_layer1)
b1 = SX.sym('b1', n_layer2)
A = (mtimes(M1, A) + b1)
net_para += [M1.reshape((-1, 1))]
net_para += [b1.reshape((-1, 1))]
net_out = dot(A, A)
net_para = vcat(net_para)

path_cost = net_out + 0.5 * (dot(env.U, env.U))
final_cost = net_out
cost_auxvar = net_para
print(cost_auxvar.numel())

# --------------------------- create PDP object ----------------------------------------
oc = CPDP.COCSys()
beta = SX.sym('beta')
dyn = beta * env.f
oc.setAuxvarVariable(vertcat(beta, cost_auxvar))
oc.setStateVariable(env.X)
oc.setControlVariable(env.U)
oc.setDyn(dyn)
path_cost = beta * path_cost
oc.setPathCost(path_cost)
oc.setFinalCost(final_cost)
oc.setIntegrator(n_grid=10)

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
T = 1.6

time_tau = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1])*T
waypoints = np.array([[-2.58688485, 2.21633114],
                      [-1.77631602, 1.51828429],
                      [-0.88806066, 0.8525125],
                      [-0.17179371, 0.45352602],
                      [0.38514534, 0.24244121],
                      [0.80564785, 0.13466983],
                      [1.327348, 0.05996715],
                      [1.47776075, 0.04868195],
                      [np.pi / 2, 0.]])

ini_state = [-pi / 2, 3 * pi / 4, -5, 3]


# --------------------------- the learning process ----------------------------------------

# numerical gradient delta


lr = 0.01e-1  # learning rate
np.random.seed(1)
initial_parameter = np.random.randn(oc.n_auxvar)
initial_parameter[0] = 1
loss_trace, parameter_trace = [], []
current_parameter = initial_parameter
parameter_trace += [current_parameter.tolist()]
for j in range(int(10)):
    time0 = time.time()
    time_grid, opt_sol = oc.cocSolver(ini_state, T, current_parameter)

    # using the analytical gradient
    # time starts
    time1 = time.time()
    auxsys_sol = oc.auxSysSolver(time_grid, opt_sol, current_parameter)
    loss, diff_loss = getloss_pos_corrections(time_tau, waypoints, opt_sol, auxsys_sol)

    # using numerical gradient
    time2 = time.time()
    diff_loss_numeric = np.zeros_like(current_parameter)
    for k in range(current_parameter.size):
        delta = 1e-8
        temp_parameter = current_parameter
        temp_parameter[k] = temp_parameter[k] + delta
        _, temp_opt_sol = oc.cocSolver(ini_state, T, temp_parameter)
        temp_loss, _ = getloss_pos_corrections(time_tau, waypoints, temp_opt_sol, auxsys_sol)
        diff_loss_numeric[k] = (temp_loss - loss) / delta
    time3 = time.time()


    # current_parameter -= lr * diff_loss_numeric
    current_parameter -= lr * diff_loss
    # do the projection step
    current_parameter[0] = fmax(current_parameter[0], 0.00000001)

    loss_trace += [loss]
    parameter_trace += [current_parameter.tolist()]
    print(j, 'loss:', loss_trace[-1].tolist(), 'numerical time:', time3-time2, 'analytical time:', time2-time1)

if False:
    save_data = {'loss_trace': loss_trace,
                 'parameter_trace': parameter_trace,
                 'initial_parameter': initial_parameter,
                 'time_tau': time_tau,
                 'waypoints': waypoints,
                 'n_state': oc.n_state,
                 'n_control': oc.n_control,
                 'time_grid': time_grid,
                 'opt_sol': opt_sol,
                 'opt_sol_generalize': opt_sol_generalize,
                 'T_generalize': T_generalize,
                 'ini_state_generalize': ini_state_generalize,
                 'lr': lr}
    np.save('../robotarm_results/compare_gradient.npy', save_data)
