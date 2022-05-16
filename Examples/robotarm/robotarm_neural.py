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

path_cost = net_out + 0.5 * (dot(env.U, env.U))
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
# time_tau = np.array([0.1, 0.2, 0.4, 0.5])
# waypoints = np.array([[-2.58688485, 2.21633114],
#                       [-1.77631602, 1.51828429],
#                       [-0.17179371, 0.45352602],
#                       [0.38514534, 0.24244121]])


# time_tau = np.array([0.1, 0.2, 0.4, 1])
# waypoints = np.array([[-2.58688485, 2.21633114],
#                       [-1.77631602, 1.51828429],
#                       [-0.17179371, 0.45352602],
#                       [np.pi/2, 0.]])


time_tau = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1])
waypoints = np.array([[-2.58688485, 2.21633114],
                      [-1.77631602, 1.51828429],
                      [-0.88806066, 0.8525125],
                      [-0.17179371, 0.45352602],
                      [0.38514534, 0.24244121],
                      [0.80564785, 0.13466983],
                      [1.327348, 0.05996715],
                      [1.47776075, 0.04868195],
                      [np.pi/2, 0.]])




# time_tau = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1])
# waypoints = np.array([[-2.58688485, 2.21633114],
#                       [-1.77631602, 1.51828429],
#                       [-0.88806066, 0.8525125],
#                       [-0.17179371, 0.45352602],
#                       [0.38514534, 0.24244121],
#                       [0.80564785, 0.13466983],
#                       [1.327348, 0.05996715],
#                       [1.47776075, 0.04868195],
#                       [np.pi/2, 0.]])

ini_state = [-pi / 2, 3 * pi / 4, -5, 3]
T = 1

# --------------------------- the learning process ----------------------------------------
lr = 1e-1  # learning rate
np.random.seed(1)
initial_parameter = np.random.randn(oc.n_auxvar)

# initial_parameter = np.array([1.32535187, 0.44042419, 0.46872797, 0.99926929, -0.97369331, 2.53778729,
#                               -1.64129294, 0.87061714, -0.1611848, -0.24374225, -1.61547253, 1.85626649,
#                               0.01666912, 1.07478339, -0.85445819, 1.41351739, 0.62719281, 0.85007355,
#                               -0.04860342, -0.59221922, 1.08443597, -1.79540858, -0.57062616, -0.53009778,
#                               -1.07416811, 0.69884421])

initial_parameter[0] = 1
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
    print('iter:', j, 'loss:', loss_trace[-1].tolist())

# save
time_grid, opt_sol = oc.cocSolver(ini_state, T, current_parameter)
regenerated_state_traj = opt_sol(time_grid)

# print(current_parameter)
#
import matplotlib.pyplot as plt

# generalization
ini_state_generalize = [-np.pi / 4, 0, 0, 0]
T_generalize = 2
time_grid_generalize = np.linspace(0, T_generalize, 20)
_, opt_sol_generalize = oc.cocSolver(ini_state_generalize, T_generalize, current_parameter)
state_traj_generalize = opt_sol_generalize(time_grid_generalize)

plt.figure(3)
plt.plot(time_grid_generalize, state_traj_generalize[:, 0])
plt.figure(4)
plt.plot(time_grid_generalize, state_traj_generalize[:, 1])
plt.show()

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
    np.save('../robotarm_results/robotarm_neural_9.npy', save_data)
