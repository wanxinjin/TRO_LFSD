import numpy as np

from CPDP import CPDP
import drone_env
import CPDP.optim as opt
from casadi import *
from scipy.integrate import solve_ivp
import scipy.io as sio
from sim_animation import *

# ---------------------------------------load environment---------------------------------------
env = drone_env.Quadrotor()
env.initDyn(Jx=.5, Jy=.5, Jz=1, mass=1, l=1, c=0.02)
env.initCost_Polynomial(w_thrust=0.1, goal_r_I=np.array([1.5, 2., 0]))

# --------------------------- create PDP object and OC solver ----------------------------------------
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

# --------------------------- set initial condition and horizon ------------------------------
# set initial condition
ini_r_I = [-1.0, -2., 0.25]
ini_v_I = [0, 0, 0.]
ini_q = drone_env.toQuaternion(0, [0, 0, 1])
ini_w = [0.0, 0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
T = 1

# ---------------------- define the loss function and interface function ------------------
interface_pos_fn = Function('interface', [oc.state], [oc.state[0:3]])
diff_interface_pos_fn = Function('diff_interface', [oc.state], [jacobian(oc.state[0:3], oc.state)])
def getloss_pos_corrections(time_grid, target_waypoints, opt_sol, auxsys_sol):
    loss = 0
    diff_loss = numpy.zeros(oc.n_auxvar)
    for k, t in enumerate(time_grid):
        # solve loss
        target_waypoint = target_waypoints[k, :]
        target_position = target_waypoint[0:3]
        current_position = interface_pos_fn(opt_sol(t)[0:oc.n_state]).full().flatten()

        loss += numpy.linalg.norm(target_position - current_position) ** 2
        # solve gradient by chain rule
        dl_dpos = current_position - target_position
        dpos_dx = diff_interface_pos_fn(opt_sol(t)[0:oc.n_state]).full()
        dxpos_dp = auxsys_sol(t)[0:oc.n_state * oc.n_auxvar].reshape((oc.n_state, oc.n_auxvar))

        dl_dp = np.matmul(numpy.matmul(dl_dpos, dpos_dx), dxpos_dp)
        diff_loss += dl_dp
    return loss, diff_loss



waypoints = np.array([
    [-.0, -1.5, 0.5],
    [.5, -1.0, 1.],
    [0.5, -0.25, 1.],
    [-.5, 0.0, 1.],
    [0, 1, 1.],
    [1, 1.5, 0.5],
])

taus = np.linspace(0.1, 1, waypoints.shape[0], endpoint=False)


# --------------------------- start the learning process --------------------------------
lr = 1e-1
loss_trace, parameter_trace = [], []
current_parameter = np.array([7, 1, 0, 1, 0, 1, 0, 0, 0, 0])
# current_parameter = np.array([ 3.5109, 8.6468, -1.264 ,  2.7641, -0.6662, 1.0328, -0.8512,  4.0165])
parameter_trace += [current_parameter.tolist()]
optimizier = opt.Adam()
optimizier.learning_rate = 1e-2
for j in range(int(100)):
    time_grid, opt_sol = oc.cocSolver(ini_state, T, current_parameter)
    auxsys_sol = oc.auxSysSolver(time_grid, opt_sol, current_parameter)
    loss, diff_loss = getloss_pos_corrections(taus, waypoints, opt_sol, auxsys_sol)
    current_parameter = optimizier.step(current_parameter, diff_loss)
    # current_parameter -= lr * diff_loss
    loss_trace += [loss]
    parameter_trace += [current_parameter.tolist()]
    np.set_printoptions(precision=4)
    print('iter:', j, 'loss:', np.array(loss_trace[-1].tolist()), 'para:', current_parameter.flatten())
    # print('iter:', j, 'loss:', loss_trace[-1].tolist())

# save the true
# Below is to obtain the final uav trajectory based on the learned objective function
_, opt_sol = oc.cocSolver(ini_state, T, current_parameter)
time_steps = np.linspace(0, T, num=100)  # generate the time inquiry grid with N is the point number
opt_traj = opt_sol(time_steps)
opt_state_traj = opt_traj[:, :oc.n_state]  # state trajectory ----- N*[r,v,q,w]
opt_control_traj = opt_traj[:, oc.n_state:oc.n_state + oc.n_control]  # control trajectory ---- N*[t1,t2,t3,t4]
play_animation(wing_len=0.25, state_traj=opt_state_traj, waypoints=waypoints)

if True:
    save_data = {'loss_trace': loss_trace,
                 'parameter_trace': parameter_trace,
                 'time_tau': taus,
                 'waypoints': waypoints,
                 'opt_sol': opt_sol,
                 'time_grid': time_steps,
                 'n_state': oc.n_state,
                 'n_control': oc.n_control,
                 'T': T,
                 'lr': lr}
    np.save('learned_results.npy', save_data)
