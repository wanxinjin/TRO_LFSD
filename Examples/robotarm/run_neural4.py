from CPDP import CPDP
from JinEnv import JinEnv
from casadi import *
from scipy.integrate import solve_ivp
import scipy.io as sio

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
# M2 = SX.sym('M2', 1, n_layer2)
# b2 = SX.sym('b2', 1)
# A = (mtimes(M2, A) + b2)
# net_para += [M2.reshape((-1, 1))]
# net_para += [b2.reshape((-1, 1))]

net_out = dot(A, A)
net_para = vcat(net_para)

path_cost = net_out + 0.05 * (dot(env.U, env.U))
final_cost = net_out
cost_auxvar = net_para

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
oc.setIntegrator(n_grid=15)

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
time_tau = np.array([0.067, 0.2, 0.267, 0.333])
waypoints = np.array([[-2.497, 2.301],
                      [-1.71, 1.353],
                      [-1.142, 0.924],
                      [-0.629, 0.606]])


# time_tau = np.array([0.1, 0.2, 0.4, 1])
# waypoints = np.array([[-2.58688485, 2.21633114],
#                       [-1.77631602, 1.51828429],
#                       [-0.17179371, 0.45352602],
#                       [np.pi/2, 0.]])


# time_tau = np.array([0.067, 0.2, 0.267, 0.333, 0.467, 0.6, 0.8, 0.933, 1])
# waypoints = np.array([[-2.497, 2.301],
#                       [-1.71, 1.353],
#                       [-1.142, 0.924],
#                       [-0.629, 0.606],
#                       [0.201, 0.25],
#                       [0.791, 0.108],
#                       [1.319, 0.049],
#                       [1.512, 0.043],
#                       [np.pi / 2, 0.]])





ini_state = [-pi / 2, 3 * pi / 4, -5, 3]
T = 1

# --------------------------- the learning process ----------------------------------------
max_trial = 10

for trial in range(6,max_trial):

    lr = 10e-3  # learning rate

    # np.random.seed(trial)
    initial_parameter = 0.5 * np.random.randn(oc.n_auxvar)
    initial_parameter[0] = 1.  # makesure the time-warping parameter is positive at the beginning.

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
        print('num_waypoints:', len(time_tau), 'trial', trial, 'iter:', j, 'loss:', loss_trace[-1].tolist())

        if j > 40:
            lr = 10e-3

    # save
    time_grid, opt_sol = oc.cocSolver(ini_state, T, current_parameter)
    regenerated_state_traj = opt_sol(time_grid)

    print(current_parameter)

    import matplotlib.pyplot as plt

    # generalization
    ini_state_generalize = [-np.pi / 4, 0, 0, 0]
    T_generalize = 2
    time_grid_generalize = np.linspace(0, T_generalize, 20)
    _, opt_sol_generalize = oc.cocSolver(ini_state_generalize, T_generalize, current_parameter)
    state_traj_generalize = opt_sol_generalize(time_grid_generalize)

    # plt.figure(3)
    # plt.plot(time_grid_generalize, state_traj_generalize[:, 0])
    # plt.figure(4)
    # plt.plot(time_grid_generalize, state_traj_generalize[:, 1])
    # plt.show()

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
                     'opt_sol_generalize': opt_sol_generalize,
                     'T_generalize': T_generalize,
                     'ini_state_generalize': ini_state_generalize,
                     'lr': lr}
        np.save('./results_data/neural/num_waypoints_' + str(len(time_tau)) + '_3_trial_' + str(trial) + '.npy',
                save_data)
