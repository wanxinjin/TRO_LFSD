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
time_tau = np.array([0.1,  0.3, 0.4,  0.6, 0.8, 0.9,])
time_tau_grid = (time_tau / DT).astype(int)
waypoints = sol['state_traj_opt'][time_tau_grid][:, 0:2]

# plt.plot(sol['state_traj_opt'][:, 0])
# plt.plot(true_opt_sol(true_time_grid)[:, 0])
# plt.show()

# --------------------------- establish the plain constrained optimization--------------------------


# define the variable
traj_x = SX.sym('traj_x', horizon + 1, doc.n_state)
traj_u = SX.sym('traj_u', horizon, doc.n_control)
traj_costate = SX.sym('traj_costate', horizon, doc.n_state)
para = SX.sym('para', 5)

# define the objective
obj = norm_2(vec(waypoints) - vec(traj_x[time_tau_grid, :][:, 0:2])) ** 2

# define the total variable
x = vcat([para, vec(traj_x), vec(traj_u), vec(traj_costate)])

# define the constraints
constr = [traj_x[0, :].T - np.array(ini_state) == 0]
for t in range(horizon):
    constr += [traj_x[t + 1, :].T - doc.dyn_fn(traj_x[t, :], traj_u[t, :], para)]
for t in range(horizon - 1):
    constr += [traj_costate[t, :].T - doc.dHx_fn(traj_x[t + 1, :], traj_u[t + 1, :], traj_costate[t + 1, :], para)]
for t in range(horizon):
    constr += [doc.dHu_fn(traj_x[t, :], traj_u[t, :], traj_costate[t, :], para)]

constr += [traj_costate[-1, :].T - doc.dhx_fn(traj_x[-1, :], para)]
g = vcat(constr)

# construct functions
f_fn = Function('f_fn', [x], [obj])
g_fn = Function('g_fn', [x], [g])

# define the optimization
# opt = casadi.Opti()
# y = opt.variable(*(x.shape))
# opt.minimize(f_fn(y))
# opt.subject_to(g_fn(y) == 0)

# # set the initial guess and solve the opt
# opt.set_initial(y, vcat(
#     [np.array([6, 3, 3, 3, 3]), vec(sol['state_traj_opt']), vec(sol['control_traj_opt']), vec(sol['costate_traj_opt'])]))
#
# p_opts = {"expand": True}
# s_opts = {"max_iter": 100}
# opt.solver("ipopt", p_opts, s_opts)
#
# sol = opt.solve()
#
#
# print(sol.value(f_fn(y)))
# print(sol.value(y[0:5]))


# set up the gradient
y = SX.sym('y', x.numel())
lam = SX.sym('lam', g.numel())
# lagrangian
L = f_fn(y) + dot(lam, g_fn(y))
dL1 = jacobian(L, y)
dL1_fn = Function('dL1_fn', [y, lam], [dL1])
dL2 = jacobian(L, lam)
dL2_fn = Function('dL2_fn', [y, lam], [dL2])

# update primal dual gradient
initial_parameter=np.array([0.5, 2, 3, 2, 3.])
current_y = np.zeros(y.numel())
current_y[0:5] = initial_parameter
current_lam = np.zeros(lam.numel())
lr = 0.1e-1
loss_trace = []
parameter_trace = []
for k in range(1000):
    parameter_trace += [current_y[0:5]]
    current_y += -lr * dL1_fn(current_y, current_lam).full().flatten()
    current_lam +=lr * dL2_fn(current_y, current_lam).full().flatten()
    loss_trace += [f_fn(current_y).full().item()]

    # print
    print('iter:', k, 'loss:', f_fn(current_y).full().item(), )

# save
time_grid, opt_sol = coc.cocSolver(ini_state, T, parameter_trace[-1])
plt.plot(time_grid, opt_sol(time_grid)[:, 0])
plt.scatter(time_tau, waypoints[:, 0], marker="o", s=100, c='r', zorder=100)

if True:
    save_data = {'loss_trace': loss_trace,
                 'parameter_trace': parameter_trace,
                 'true_parameter': true_parameter,
                 'initial_parameter': initial_parameter,
                 'time_tau': time_tau,
                 'waypoints': waypoints,
                 'opt_sol': opt_sol,
                 'time_grid': time_grid},
    np.save('../robotarm_results/compare_plain_primaldual_1.npy', save_data)
