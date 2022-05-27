from CPDP import CPDP
from JinEnv import JinEnv
from casadi import *
from scipy.integrate import solve_ivp
import scipy.io as sio
import CPDP.optim as opt

# ---------------------------------------load environment---------------------------------------
env = JinEnv.Quadrotor()
env.initDyn(Jx=.5, Jy=.5, Jz=1, mass=1, l=1, c=0.02)
env.initCost_Polynomial(w_thrust=0.1, goal_r_I=np.array([8, 8, 0]))

# --------------------------- create PDP object and OC solver ----------------------------------------
oc = CPDP.COCSys()
beta = SX.sym('beta')
dyn = beta * env.f
oc.setStateVariable(env.X)
oc.setControlVariable(env.U)
oc.setDyn(dyn)

# set the feature-based features
pos_uav = env.X[0:3]

# feature 1
pos_obstacle1 = np.array([-1, -4])
feature_1 = -dot(pos_uav[0:2] - pos_obstacle1, pos_uav[0:2] - pos_obstacle1)
weight_1 = SX.sym('weight_1')

# feature 2
pos_obstacle2 = np.array([-1, -7.5, ])
feature_2 = -dot(pos_uav[0:2] - pos_obstacle2, pos_uav[0:2] - pos_obstacle2)
weight_2 = SX.sym('weight_2')

# feature 3
pos_obstacle3 = np.array([0.5, 0, ])
feature_3 = -dot(pos_uav[0:2] - pos_obstacle3, pos_uav[0:2] - pos_obstacle3)
weight_3 = SX.sym('weight_3')

# feature 4
pos_obstacle4 = np.array([0.5, 4, ])
feature_4 = -dot(pos_uav[0:2] - pos_obstacle4, pos_uav[0:2] - pos_obstacle4)
weight_4 = SX.sym('weight_4')

# feature 3
feature_5 = env.cost_thrust

path_features = weight_1 * feature_1 + weight_2 * feature_2 + weight_3 * feature_3 + weight_4 * feature_4 + 0.1 * feature_5
oc.setAuxvarVariable(vertcat(beta, weight_1, weight_2, weight_3, weight_4))
path_cost = beta * path_features
oc.setPathCost(path_cost)
oc.setFinalCost(env.final_cost)
oc.setIntegrator(n_grid=15)
print(oc.auxvar)

# --------------------------- set initial condition and horizon ------------------------------
ini_r_I = [-8, -8, 5.]
ini_v_I = [15, 5, -10]
ini_q = JinEnv.toQuaternion(0, [0, 0, 1])
ini_w = [0.0, 0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
T = 1

# ---------------------- define the loss function and interface function ------------------
interface_pos_fn = Function('interface', [oc.state], [oc.state[0:3]])
diff_interface_pos_fn = Function('diff_interface', [oc.state], [jacobian(oc.state[0:3], oc.state)])


def getloss_pos_corrections(time_grid, target_waypoints, opt_sol, auxsys_sol):
    loss = 0.
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


taus = np.array([0.1, 0.2, 0.4, 0.6, 0.8])
waypoints = np.array([[-4, -6., 3],
                      [1, -6., 3],
                      [1, -1., 4],
                      [-1.0, 1., 5],
                      [2.0, 3.0, 4.0]])

# --------------------------- start the learning process --------------------------------
max_trial=5
for k in range(max_trial):
    optimizier = opt.Adam()
    optimizier.learning_rate = 10e-2
    loss_trace, parameter_trace = [], []
    current_parameter = np.random.uniform(0.1,1,oc.n_auxvar)
    #
    # current_parameter = np.array([2, 0.1, 0.1, 0.1, 0.1 ])
    # current_parameter = np.array([ 1.83657309, -1.07227283, -1.09366231,  0.17243003])
    parameter_trace += [current_parameter.tolist()]
    for j in range(int(200)):
        time_grid, opt_sol = oc.cocSolver(ini_state, T, current_parameter)
        auxsys_sol = oc.auxSysSolver(time_grid, opt_sol, current_parameter)
        loss, diff_loss = getloss_pos_corrections(taus, waypoints, opt_sol, auxsys_sol)
        current_parameter = optimizier.step(current_parameter, diff_loss)
        loss_trace += [loss]
        parameter_trace += [current_parameter.tolist()]
        print('case feature, subcase 1:', 'trial:', k, 'iter:', j, 'loss:', loss_trace[-1].tolist())
        if loss<30: optimizier.learning_rate=1e-2
        #


    # save the true
    # Below is to obtain the final uav trajectory based on the learned objective function
    _, opt_sol = oc.cocSolver(ini_state, T, current_parameter)
    time_steps = np.linspace(0, T, num=100)  # generate the time inquiry grid with N is the point number
    opt_traj = opt_sol(time_steps)
    opt_state_traj = opt_traj[:, :oc.n_state]  # state trajectory ----- N*[r,v,q,w]
    opt_control_traj = opt_traj[:, oc.n_state:oc.n_state + oc.n_control]  # control trajectory ---- N*[t1,t2,t3,t4]
    # env.play_animation(wing_len=1, state_traj=opt_state_traj, waypoints=waypoints)

    if True:
        save_data = {'loss_trace': loss_trace,
                     'parameter_trace': parameter_trace,
                     'time_tau': taus,
                     'waypoints': waypoints,
                     'opt_sol': opt_sol,
                     'time_grid': time_steps,
                     'n_state': oc.n_state,
                     'n_control': oc.n_control,
                     'T': T}
        np.save('./results/case_feature/subcase1_trial_'+str(k)+'.npy', save_data)
