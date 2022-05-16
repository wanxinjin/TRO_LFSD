from CPDP import CPDP
from JinEnv import JinEnv
from casadi import *

# Plot the learned trajectory
load = np.load('../quadrotor_case4.npy', allow_pickle=True).item()
time_tau = load['time_tau']
waypoints = load['waypoints']
parameter_trace = np.array(load['parameter_trace'])
loss_trace = load['loss_trace']

# ----------load environment---------------------------------------
env = JinEnv.Quadrotor()
env.initDyn(Jx=.5, Jy=.5, Jz=1, mass=1, l=1, c=0.02)
env.initCost_Polynomial(w_thrust=0.1, goal_r_I=np.array([8, 8, 0]))

# --------------------------- create optimal control object ----------------------------------------
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
oc.setIntegrator(n_grid=18)
# --------------------------- set initial condition and horizon ------------------------------
# set initial condition
ini_r_I = [-8, -8, 5.]
ini_v_I = [15, 5, -10]
ini_q = JinEnv.toQuaternion(0, [0, 0, 1])
ini_w = [0.0, 0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
T = 1

_, opt_sol = oc.cocSolver(ini_state, T, parameter_trace[-1])
time_steps = np.linspace(0, T, num=200)  # generate the time inquiry grid with N is the point number
opt_traj = opt_sol(time_steps)
opt_state_traj = opt_traj[:, :oc.n_state]  # state trajectory ----- N*[r,v,q,w]
opt_control_traj = opt_traj[:, oc.n_state:oc.n_state + oc.n_control]  # control trajectory ---- N*[t1,t2,t3,t4]
env.play_animation2(wing_len=1, state_traj=opt_state_traj, waypoints=waypoints,
                    save_option=1)



