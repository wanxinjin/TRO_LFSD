import casadi
import numpy as np

from CPDP import CPDP
from CPDP import PDP
from JinEnv import JinEnv
from casadi import *
from scipy.integrate import solve_ivp
import scipy.io as sio
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import time

# ---------------------------------------load environment---------------------------------------
env = JinEnv.RobotArm()
env.initDyn(l1=1, m1=2, l2=1, m2=1, g=0)
env.initCost_WeightedDistance(wu=0.5)

# --------------------------- create optimal control object --------------------------
coc = CPDP.COCSys()
beta = SX.sym('beta')
cdyn = beta * env.f
coc.setAuxvarVariable(vertcat(beta, env.cost_auxvar))
coc.setStateVariable(env.X)
coc.setControlVariable(env.U)
coc.setDyn(cdyn)
path_cost = beta * env.path_cost
coc.setPathCost(path_cost)
coc.setFinalCost(env.final_cost)
n_grid = 50
coc.setIntegrator(n_grid=n_grid)
ini_state = [-pi / 2, 3 * pi / 4, -5, 3]
T = 1

# --------------------------- compute the solution for the continueous system --------------------------
true_parameter = [5., 3, 3, 3, 3]
true_time_grid, true_opt_sol = coc.cocSolver(ini_state, T, true_parameter)

# --------------------------- discretize the continuous system --------------------------
DT = T / n_grid
ddyn = env.X + DT * cdyn
doc = PDP.OCSys()
doc.setAuxvarVariable(vertcat(beta, env.cost_auxvar))
doc.setStateVariable(env.X)
doc.setControlVariable(env.U)
doc.setDyn(ddyn)
doc.setPathCost(path_cost * DT)
doc.setFinalCost(env.final_cost)
doc.diffPMP()
dlqr_solver = PDP.LQR()
horizon = n_grid
sol = doc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=true_parameter)

# --------------------------- generate the sparse waypoints-----------------
time_tau = np.array([0.1, 0.3, 0.4, 0.6, 0.8, 0.9, ])
time_tau_grid = (time_tau / DT).astype(int)
waypoints = sol['state_traj_opt'][time_tau_grid][:, 0:2]

# plt.plot(sol['state_traj_opt'][:, 0])
# plt.plot(true_opt_sol(true_time_grid)[:, 0])
# plt.show()

# --------------------------- establish the plain constrained optimization--------------------------
opt = casadi.Opti()
# define the variable
traj_x = opt.variable(horizon + 1, doc.n_state)
traj_u = opt.variable(horizon, doc.n_control)
traj_costate = opt.variable(horizon, doc.n_state)
para = opt.variable(5)
# define the objective
obj = norm_2(vec(waypoints) - vec(traj_x[time_tau_grid, :][:, 0:2])) ** 2
opt.minimize(obj)
# define the constraints
constr = [traj_x[0, :].T == np.array(ini_state)]
for t in range(horizon):
    constr += [traj_x[t + 1, :].T == doc.dyn_fn(traj_x[t, :], traj_u[t, :], para)]
for t in range(horizon - 1):
    constr += [traj_costate[t, :].T == doc.dHx_fn(traj_x[t + 1, :], traj_u[t + 1, :], traj_costate[t + 1, :], para)]
for t in range(horizon):
    constr += [0 == doc.dHu_fn(traj_x[t, :], traj_u[t, :], traj_costate[t, :], para)]

constr += [traj_costate[-1, :].T == doc.dhx_fn(traj_x[-1, :], para)]

# define the optimization
opt.subject_to(constr)

# set the initial guess and solve the opt
# initial_parameter=np.array([1, 2, 3, 2, 3.])
# initial_parameter=np.array([2, 2.5, 3.5, 2.5, 3.5])
initial_parameter=np.array([0.5, 2.8, 3.2, 2.8, 3.2])

opt.set_initial(para, initial_parameter)
# opt.set_initial(traj_x, sol['state_traj_opt'])
# opt.set_initial(traj_u,sol['control_traj_opt'])
# opt.set_initial(traj_costate, sol['costate_traj_opt']+0.1)


p_opts = {"expand": True, }
s_opts = {"max_iter": 150,}
opt.solver("ipopt", p_opts, s_opts)

result_trace = []

opt.callback(lambda i: result_trace.append([opt.value(obj), opt.value(para)]))
# opt.callback(lambda i: parameter_trace.append(opt.value(para)))

try:
    sol = opt.solve()
except:
    print('exit')

loss_trace = []
parameter_trace = []
for item in result_trace:
    loss_trace += [item[0]]
    parameter_trace += [item[1]]

plt.plot(loss_trace)
plt.show()

sol = doc.ocSolver(ini_state=ini_state, horizon=horizon, auxvar_value=parameter_trace[-1])


# save
if True:
    save_data = {'loss_trace': loss_trace,
                 'parameter_trace': parameter_trace,
                 'true_parameter': true_parameter,
                 'initial_parameter': initial_parameter,
                 'time_tau': time_tau,
                 'waypoints': waypoints,
                 'opt_sol': sol,
                 'time_grid': np.linspace(0, 1, horizon + 1)},
    np.save('./results_data/plain_opt/compare_plain_ipopt_3.npy', save_data)
