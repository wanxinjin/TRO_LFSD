import numpy as np

from CPDP import CPDP
import drone_env
import CPDP.optim as opt
from casadi import *
from scipy.integrate import solve_ivp
import scipy.io as sio
from sim_animation import *
import matplotlib.pyplot as plt

load = np.load('learned_results.npy', allow_pickle=True).item()
time_tau = load['time_tau']
waypoints = load['waypoints']
parameter_trace = np.array(load['parameter_trace'])
loss_trace = load['loss_trace']
print('iteration count:', parameter_trace.shape[0])


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
ini_r_I = [1.0, -2., 0.25]
ini_v_I = [0, 0, 0.]
ini_q = drone_env.toQuaternion(0, [0, 0, 1])
ini_w = [0.0, 0.0, 0.0]
ini_state = ini_r_I + ini_v_I + ini_q + ini_w
T = 1

# save the true
# Below is to obtain the final uav trajectory based on the learned objective function
_, opt_sol = oc.cocSolver(ini_state, T, parameter_trace[-1])
time_steps = np.linspace(0, T, num=100)  # generate the time inquiry grid with N is the point number
opt_traj = opt_sol(time_steps)
opt_state_traj = opt_traj[:, :oc.n_state]  # state trajectory ----- N*[r,v,q,w]
opt_control_traj = opt_traj[:, oc.n_state:oc.n_state + oc.n_control]  # control trajectory ---- N*[t1,t2,t3,t4]
play_animation(wing_len=0.25, state_traj=opt_state_traj, waypoints=waypoints)

# uav_traj = opt_state_traj[:, 0:3]
# np.save('uav_traj_80.npy', uav_traj)
