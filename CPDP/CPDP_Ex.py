# Copyright @ Wanxin Jin

from casadi import *
import numpy
from scipy.integrate import solve_ivp
import scipy.interpolate as ip
import time

class COCSys_Ex:

    def __init__(self, project_name="myOc"):
        self.sys_name = project_name

    def setAuxvarVariable(self, auxvar=SX.sym('auxvar', 1)):
        self.auxvar = auxvar
        self.n_auxvar = self.auxvar.numel()

    def setStateVariable(self, state, state_lb=[], state_ub=[]):
        self.state = state
        self.n_state = self.state.numel()
        if len(state_lb) == self.n_state:
            self.state_lb = state_lb
        else:
            self.state_lb = self.n_state * [-1e20]

        if len(state_ub) == self.n_state:
            self.state_ub = state_ub
        else:
            self.state_ub = self.n_state * [1e20]

    def setControlVariable(self, control, control_lb=[], control_ub=[]):
        self.control = control
        self.n_control = self.control.numel()

        if len(control_lb) == self.n_control:
            self.control_lb = control_lb
        else:
            self.control_lb = self.n_control * [-1e20]

        if len(control_ub) == self.n_control:
            self.control_ub = control_ub
        else:
            self.control_ub = self.n_control * [1e20]

    def setTimeVariable(self,t=SX.sym('time',1)):
        self.time=t

    def setDyn(self, ode):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()
        if not hasattr(self,'time'):
            self.setTimeVariable()

        self.dyn = ode
        self.dyn_fn = casadi.Function('dyn', [self.time, self.state, self.control, self.auxvar], [self.dyn])
        self.dfx_fn = casadi.Function('dfx', [self.time, self.state, self.control, self.auxvar],
                                      [jacobian(self.dyn, self.state)])
        self.dfu_fn = casadi.Function('dfu', [self.time, self.state, self.control, self.auxvar],
                                      [jacobian(self.dyn, self.control)])

    def setPathCost(self, path_cost):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()
        if not hasattr(self,'time'):
            self.setTimeVariable()

        self.path_cost = path_cost
        self.path_cost_fn = casadi.Function('path_cost', [self.time,self.state, self.control, self.auxvar], [self.path_cost])
        self.dcx_fn = casadi.Function('dcx', [self.time,self.state, self.control, self.auxvar],
                                      [jacobian(self.path_cost, self.state)])
        self.dcu_fn = casadi.Function('dcu', [self.time,self.state, self.control, self.auxvar],
                                      [jacobian(self.path_cost, self.control)])

    def setFinalCost(self, final_cost):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()
        if not hasattr(self,'time'):
            self.setTimeVariable()

        self.final_cost = final_cost
        self.final_cost_fn = casadi.Function('final_cost', [self.time,self.state, self.auxvar], [self.final_cost])
        self.dhx_fn = casadi.Function('dhx', [self.time,self.state, self.auxvar],
                                      [jacobian(self.final_cost, self.state)])

    def setIntegrator(self, n_grid=10, steps_per_grid=4):
        self.n_grid = n_grid
        self.steps_per_grid = steps_per_grid

    def cocSolver(self, ini_state, horizon, auxvar_value=1, interplation_level=1, print_level=0):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'control'), "Define the control variable first!"
        assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        assert hasattr(self, 'path_cost'), "Define the running cost/reward function first!"
        assert hasattr(self, 'final_cost'), "Define the final cost/reward function first!"

        # Check if the integration parameter is set
        if not hasattr(self, 'n_grid'):
            self.setIntegrator()

        if type(ini_state) is list:
            ini_state=numpy.array(ini_state).flatten()

        # Substitute with the given auxvarValues
        auxvar_value = DM(auxvar_value)
        f = substitute(self.dyn, self.auxvar, auxvar_value)
        c = substitute(self.path_cost, self.auxvar, auxvar_value)

        DT = horizon / self.n_grid / self.steps_per_grid
        fc = Function('fc', [self.time, self.state, self.control], [f,c])
        t=self.time
        X0 = self.state
        U = self.control
        X = X0
        Q = 0
        for j in range(self.steps_per_grid):
            k1, k1_q = fc(t, X, U)
            k2, k2_q = fc(t, X + DT / 2 * k1, U)
            k3, k3_q = fc(t, X + DT / 2 * k2, U)
            k4, k4_q = fc(t, X + DT * k3, U)
            X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        grid_fc = Function('F', [t, X0, U], [X, Q], ['t','x0', 'p'], ['xf', 'qf'])

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = MX.sym('X0', self.n_state)
        w += [Xk]
        w0 += self.n_state * [0]
        lbw += self.state_lb
        ubw += self.state_ub

        # add the condition for the initial condition
        g += [ini_state - Xk]
        lbg += self.n_state * [0]
        ubg += self.n_state * [0]

        # Formulate the NLP
        time_grid=numpy.linspace(0,horizon,self.n_grid+1)
        for k in range(self.n_grid):
            # New NLP variable for the control
            Uk = MX.sym('U_' + str(k),self.n_control)
            w += [Uk]
            lbw += self.control_lb
            ubw += self.control_ub
            w0 += [0.5 * (x + y) for x, y in zip(self.control_lb, self.control_ub)]

            # Integrate till the end of the interval
            tk=time_grid[k]
            Fk = grid_fc(t=tk, x0=Xk, p=Uk)
            Xk_end = Fk['xf']
            J = J + Fk['qf']

            # New NLP variable for state at end of interval
            Xk = MX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.state_lb
            ubw += self.state_ub
            w0 += [0.5 * (x + y) for x, y in zip(self.state_lb, self.state_ub)]

            # Add equality constraint
            g += [Xk_end - Xk]
            lbg += self.n_state * [0]
            ubg += self.n_state * [0]

        # Adding the final cost
        J = J + self.final_cost_fn(time_grid[-1], Xk, auxvar_value)

        # Create an NLP solver
        opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        # Obtain the control, state, and costate
        state_control_grid_opt = numpy.concatenate((w_opt, self.n_control * [0]))
        state_control_grid_opt = numpy.reshape(state_control_grid_opt, (-1, self.n_state + self.n_control))
        state_grid_opt = state_control_grid_opt[:, 0:self.n_state]
        control_grid_opt = state_control_grid_opt[:, self.n_state:]
        control_grid_opt[-1, :] = control_grid_opt[-2, :]
        costate_grid_opt = numpy.reshape(sol['lam_g'].full().flatten(), (-1, self.n_state))

        # output
        opt_sol = self.interpolation(time_grid, numpy.concatenate((state_grid_opt, control_grid_opt, costate_grid_opt), axis=1), interplation_level)

        return time_grid, opt_sol

    def diffPMP(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'control'), "Define the control variable first!"
        assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        assert hasattr(self, 'path_cost'), "Define the running cost/reward function first!"
        assert hasattr(self, 'final_cost'), "Define the final cost/reward function first!"

        # System first order differential of dynamics
        self.dfx = jacobian(self.dyn, self.state)
        self.dfx_fn = casadi.Function('dfx', [self.time, self.state, self.control, self.auxvar], [self.dfx])
        self.dfu = jacobian(self.dyn, self.control)
        self.dfu_fn = casadi.Function('dfu', [self.time, self.state, self.control, self.auxvar], [self.dfu])
        self.dfe = jacobian(self.dyn, self.auxvar)
        self.dfe_fn = casadi.Function('dfe', [self.time, self.state, self.control, self.auxvar], [self.dfe])

        # Hamiltonian
        self.costate = casadi.SX.sym('lambda', self.state.size())
        self.path_Hamil = self.path_cost + (self.dyn.T) @ self.costate
        self.final_hamil = self.final_cost

        # first order differential of path Hamiltonian
        self.dHx = jacobian(self.path_Hamil, self.state).T
        self.dHx_fn = casadi.Function('dHx', [self.time, self.state, self.control, self.costate, self.auxvar], [self.dHx])
        self.dHu = jacobian(self.path_Hamil, self.control).T
        self.dHu_fn = casadi.Function('dHu', [self.time, self.state, self.control, self.costate, self.auxvar], [self.dHu])
        # second order differential of path Hamiltonian
        self.ddHxx = jacobian(self.dHx, self.state)
        self.ddHxx_fn = casadi.Function('ddHxx', [self.time,self.state, self.control, self.costate, self.auxvar], [self.ddHxx])
        self.ddHxu = jacobian(self.dHx, self.control)
        self.ddHxu_fn = casadi.Function('ddHxu', [self.time,self.state, self.control, self.costate, self.auxvar], [self.ddHxu])
        self.ddHxe = jacobian(self.dHx, self.auxvar)
        self.ddHxe_fn = casadi.Function('ddHxe', [self.time,self.state, self.control, self.costate, self.auxvar], [self.ddHxe])

        self.ddHux = jacobian(self.dHu, self.state)
        self.ddHux_fn = casadi.Function('ddHux', [self.time, self.state, self.control, self.costate, self.auxvar], [self.ddHux])
        self.ddHuu = jacobian(self.dHu, self.control)
        self.ddHuu_fn = casadi.Function('ddHuu', [self.time, self.state, self.control, self.costate, self.auxvar], [self.ddHuu])
        self.ddHue = jacobian(self.dHu, self.auxvar)
        self.ddHue_fn = casadi.Function('ddHue', [self.time, self.state, self.control, self.costate, self.auxvar], [self.ddHue])

        # first order differential of final Hamiltonian
        self.dhx = jacobian(self.final_hamil, self.state).T
        self.dhx_fn = casadi.Function('dhx', [self.time, self.state, self.auxvar], [self.dhx])
        # second order differential of path Hamiltonian
        self.ddhxx = jacobian(self.dhx, self.state)
        self.ddhxx_fn = casadi.Function('ddhxx', [self.time, self.state, self.auxvar], [self.ddhxx])
        self.ddhxe = jacobian(self.dhx, self.auxvar)
        self.ddhxe_fn = casadi.Function('ddhxe', [self.time, self.state, self.auxvar], [self.ddhxe])

    def raccatiODE(self):
        statement = [hasattr(self, 'dfx_fn'), hasattr(self, 'dfu_fn'), hasattr(self, 'dfe_fn'),
                     hasattr(self, 'ddHxx_fn'), hasattr(self, 'ddHxu_fn'), hasattr(self, 'ddHxe_fn'),
                     hasattr(self, 'ddHux_fn'),
                     hasattr(self, 'ddHuu_fn'), hasattr(self, 'ddHue_fn'), hasattr(self, 'ddhxx_fn'),
                     hasattr(self, 'ddhxe_fn'), ]
        if not all(statement):
            self.diffPMP()

        invHuu=casadi.inv(self.ddHuu)
        GinvHuu=casadi.mtimes(self.dfu,invHuu)
        HxuinvHuu = casadi.mtimes(self.ddHxu, invHuu)
        A = self.dfx - casadi.mtimes(GinvHuu, self.ddHxu.T)
        R = casadi.mtimes(GinvHuu, self.dfu.T)
        Q = self.ddHxx - casadi.mtimes(HxuinvHuu, self.ddHxu.T)
        r = self.dfe - casadi.mtimes(GinvHuu, self.ddHue)
        q = self.ddHxe - casadi.mtimes(HxuinvHuu, self.ddHue)

        self.P=SX.sym('P', self.n_state, self.n_state)
        self.W = SX.sym('p', self.n_state, self.n_auxvar)
        P_dot = -(Q + casadi.mtimes(A.T, self.P) + casadi.mtimes(self.P, A) - casadi.mtimes(casadi.mtimes(self.P, R),self.P))
        W_dot = casadi.mtimes(casadi.mtimes(self.P, R), self.W) - casadi.mtimes(A.T, self.W) - casadi.mtimes(self.P, r) - q

        self.raccati_fn=Function('Pdot', [self.time, self.state, self.control, self.costate, self.auxvar, self.P, self.W], [P_dot, W_dot])

    def interpolation(self,x,y, method=1):
        if method==1:
            ipFun=ip.interp1d(x,y,axis=0)
            return ipFun
        if method==2:
            ipFun=ip.interp1d(x,y,axis=0,kind='cubic')
            return ipFun

    def auxSysODE(self):
        statement = [hasattr(self, 'dfx_fn'), hasattr(self, 'dfu_fn'), hasattr(self, 'dfe_fn'),
                     hasattr(self, 'ddHxx_fn'), hasattr(self, 'ddHxu_fn'), hasattr(self, 'ddHxe_fn'),
                     hasattr(self, 'ddHux_fn'),
                     hasattr(self, 'ddHuu_fn'), hasattr(self, 'ddHue_fn'), hasattr(self, 'ddhxx_fn'),
                     hasattr(self, 'ddhxe_fn'), ]
        if not all(statement):
            self.diffPMP()

        if not hasattr(self,'P'):
            self.raccatiODE()

        self.auxsys_state=SX.sym('auxsys_x', self.n_state, self.n_auxvar)
        invHuu = casadi.inv(self.ddHuu)
        auxsys_control=-mtimes(invHuu, mtimes((self.ddHux+self.dfu.T@self.P), self.auxsys_state) + mtimes(self.dfu.T, self.W) + self.ddHue)
        self.auxsys_controller_fn=Function('auxsys_controller', [self.time, self.state, self.control, self.costate, self.auxvar, self.P, self.W, self.auxsys_state], [auxsys_control])
        auxsys_state_dot= mtimes(self.dfx, self.auxsys_state) + mtimes(self.dfu, auxsys_control) + self.dfe
        self.auxsys_state_dot_fn=Function('auxsys_x_dot', [self.time, self.state, self.control, self.costate, self.auxvar, self.P, self.W, self.auxsys_state], [auxsys_state_dot])

    def auxSysSolver(self, time_grid, opt_sol, auxvar_value=1):
        statement = [hasattr(self, 'dfx_fn'), hasattr(self, 'dfu_fn'), hasattr(self, 'dfe_fn'),
                     hasattr(self, 'ddHxx_fn'), hasattr(self, 'ddHxu_fn'), hasattr(self, 'ddHxe_fn'),
                     hasattr(self, 'ddHux_fn'),
                     hasattr(self, 'ddHuu_fn'), hasattr(self, 'ddHue_fn'), hasattr(self, 'ddhxx_fn'),
                     hasattr(self, 'ddhxe_fn'), ]
        if not all(statement):
            self.diffPMP()

        if not hasattr(self, 'P'):
            self.raccatiODE()

        if not hasattr(self,'auxsys_controller_fn'):
            self.auxSysODE()

        # Generate the Raccati matrices P and W (both vectorized)
        def vec_PW_ode(t, vec_PW):
            P = numpy.reshape(vec_PW[0:self.n_state * self.n_state], (self.n_state, self.n_state))
            W = numpy.reshape(vec_PW[self.n_state * self.n_state:], (self.n_state, -1))
            xulam=opt_sol(t)
            x= xulam[0:self.n_state]
            u= xulam[self.n_state:self.n_state + self.n_control]
            lam= xulam[self.n_state + self.n_control:]
            P_dot, W_dot=self.raccati_fn(t, x, u, lam, auxvar_value, P, W)
            return numpy.concatenate((P_dot.full().flatten(), W_dot.full().flatten()))

        xulam = opt_sol(numpy.asscalar(time_grid[-1]))
        x = xulam[0:self.n_state]
        vec_PW_grid = numpy.zeros((self.n_grid + 1, self.n_state * self.n_state + self.n_state * self.n_auxvar))
        vec_PW_grid[-1, :] = numpy.concatenate((self.ddhxx_fn(time_grid[-1],x, auxvar_value).full().flatten(),
                                              self.ddhxe_fn(time_grid[-1],x, auxvar_value).full().flatten()))

        for k in range(self.n_grid, 0, -1):
            t_span = [time_grid[k], time_grid[k - 1]]
            intSol = solve_ivp(vec_PW_ode, t_span, vec_PW_grid[k, :], t_eval=[t_span[1]])
            vec_PW_grid[k - 1, :] = intSol.y.flatten()

        vec_PW_sol = self.interpolation(time_grid, vec_PW_grid)

        # integrating the auxiliary system
        def vec_auxsys_state_ode(t, vec_auxsys_state):
            auxsys_state=vec_auxsys_state.reshape((self.n_state, self.n_auxvar))
            xulam = opt_sol(t)
            x = xulam[0:self.n_state]
            u = xulam[self.n_state:self.n_state + self.n_control]
            lam = xulam[self.n_state + self.n_control:]
            vec_PW=vec_PW_sol(t)
            P = numpy.reshape(vec_PW[0:self.n_state * self.n_state], (self.n_state, self.n_state))
            W = numpy.reshape(vec_PW[self.n_state * self.n_state:], (self.n_state, -1))
            return self.auxsys_state_dot_fn(t, x, u, lam, auxvar_value, P, W, auxsys_state).full().flatten()

        vec_auxsys_state = numpy.zeros((self.n_grid + 1, self.n_state * self.n_auxvar))
        vec_auxsys_control = numpy.zeros((self.n_grid + 1, self.n_control * self.n_auxvar))

        vec_auxsys_state[0, :] = numpy.zeros(self.n_state * self.n_auxvar)
        xulam = opt_sol(0)
        x = xulam[0:self.n_state]
        u = xulam[self.n_state:self.n_state + self.n_control]
        lam = xulam[self.n_state + self.n_control:]
        vec_PW = vec_PW_sol(0)
        P = numpy.reshape(vec_PW[0:self.n_state * self.n_state], (self.n_state, self.n_state))
        W = numpy.reshape(vec_PW[self.n_state * self.n_state:], (self.n_state, -1))
        auxsys_control=self.auxsys_controller_fn(0, x, u, lam, auxvar_value, P, W, vec_auxsys_state[0, :].reshape((self.n_state, self.n_auxvar)))
        vec_auxsys_control[0,:]= auxsys_control.full().flatten()

        for k in range(0, self.n_grid):
            t_span = [time_grid[k], time_grid[k + 1]]
            intSol = solve_ivp(vec_auxsys_state_ode, t_span, vec_auxsys_state[k, :], t_eval=[time_grid[k + 1]])
            vec_auxsys_state[k+1, :] = intSol.y.flatten()
            xulam = opt_sol(numpy.asscalar(time_grid[k + 1]))
            x = xulam[0:self.n_state]
            u = xulam[self.n_state:self.n_state + self.n_control]
            lam = xulam[self.n_state + self.n_control:]
            vec_PW = vec_PW_sol(numpy.asscalar(time_grid[k + 1]))
            P = numpy.reshape(vec_PW[0:self.n_state * self.n_state], (self.n_state, self.n_state))
            W = numpy.reshape(vec_PW[self.n_state * self.n_state:], (self.n_state, -1))
            auxsys_control=self.auxsys_controller_fn(time_grid[k+1], x, u, lam, auxvar_value, P, W, vec_auxsys_state[k + 1, :].reshape((self.n_state, self.n_auxvar)))
            vec_auxsys_control[k + 1, :]=auxsys_control.full().flatten()


        return self.interpolation(time_grid, numpy.concatenate((vec_auxsys_state, vec_auxsys_control), axis=1))

class motionPlanner:
    def __init__(self, projectName="planner"):
        self.sysName = projectName

    def setStatevar(self, state, stateLB=[], stateUB=[]):
        self.state = state
        self.nstate = self.state.numel()
        if len(stateLB) == self.nstate:
            self.stateLB = stateLB
        else:
            self.stateLB = self.nstate * [-1e20]

        if len(stateUB) == self.nstate:
            self.stateUB = stateUB
        else:
            self.stateUB = self.nstate * [1e20]

    def setControlvar(self, control, controlLB=[], controlUB=[]):
        self.control = control
        self.ncontrol = self.control.numel()

        if len(controlLB) == self.ncontrol:
            self.controlLB = controlLB
        else:
            self.controlLB = self.ncontrol * [-1e20]

        if len(controlUB) == self.ncontrol:
            self.controlUB = controlUB
        else:
            self.controlUB = self.ncontrol * [1e20]

    def setDyn(self, ode):
        self.dyn = ode
        self.dynFun = casadi.Function('dynFun', [self.state, self.control], [self.dyn])

    def setPathObj(self, pathObj):
        self.pathObjFun = casadi.Function('pathObjFun', [self.state, self.control], [pathObj])
        self.pathObj = pathObj

    def setFinalObj(self, finalObj):
        self.finalObjFun = casadi.Function('finalObjFun', [self.state], [finalObj])
        self.finalObj = finalObj

    def setPivot(self, horizon, nPoly=2, mode=False):
        # use the polynomial to represent the control inputs
        self.nPoly = nPoly  # polynomial order
        self.t = SX.sym('t')
        if mode:
            # method 1
            Ktemp = []
            self.controlEqu = 0.0
            for i in range(self.nPoly + 1):
                Ki = SX.sym('K_' + str(i), self.ncontrol)
                Ktemp += [Ki]
                self.controlEqu = self.controlEqu + Ki * (self.t ** i)
            self.pivot = vcat(Ktemp)  # this is control pivots which are coefficients of polynomial
            self.npivot = self.pivot.numel()
            self.pivotFun = casadi.Function('controlFun', [self.t, self.pivot], [self.controlEqu])
        else:
            # Method 2
            pTime = numpy.linspace(0, horizon - 1, self.nPoly + 1)
            self.controlEqu = numpy.zeros(self.ncontrol)
            pControl = []
            for i in range(self.nPoly + 1):
                Ui = SX.sym('U_' + str(i), self.ncontrol)
                pControl += [Ui]
                bi = 1
                for j in range(self.nPoly + 1):
                    if j != i:
                        bi = bi * (self.t - pTime[j]) / (pTime[i] - pTime[j])
                self.controlEqu = self.controlEqu + bi * Ui
            self.pivot = vcat(pControl)
            self.npivot = self.pivot.numel()
            self.pivotFun = casadi.Function('controlFun', [self.t, self.pivot], [self.controlEqu])

    def generatePlannerDiff(self):
        # System first order differential of dynamics
        self.dfx = jacobian(self.dyn, self.state)
        self.dfxFun = casadi.Function('dfx', [self.state, self.control], [self.dfx])
        self.dfu = jacobian(self.dyn, self.control)
        self.dfuFun = casadi.Function('dfu', [self.state, self.control], [self.dfu])

        # differential control function
        self.dpivot = jacobian(self.controlEqu, self.pivot)
        self.dpivotFun = casadi.Function('dpivot', [self.t, self.pivot], [self.dpivot])

        # differential objective function
        self.dcxFun = casadi.Function('dcx', [self.state, self.control], [jacobian(self.pathObj, self.state)])
        self.dcuFun = casadi.Function('dcx', [self.state, self.control], [jacobian(self.pathObj, self.control)])
        self.dhxFun = casadi.Function('dhx', [self.state], [jacobian(self.finalObj, self.state)])

    def motionPlanning(self, iniState, horizon, initialPivot=[], eta=1e-3, maxIter=1.0000e4, epsilon=1.0000e-06):

        # set initial pivot
        if not initialPivot:
            # self.initialPivot=-numpy.ones(self.npivot)
            self.initialPivot = 1.5 * numpy.random.randn(self.npivot)
        else:
            self.initialPivot = initialPivot
        self.eta = eta
        self.maxIter = maxIter
        self.epsilon = epsilon

        # do the planning
        iterIndex = 0
        lossTrace = []  # for storage
        pivotTrace = []  # for storage
        pivot = self.initialPivot

        while True:
            # orginial system trajectory
            state = [numpy.array(iniState)]
            control = []

            # backprop system trajectory
            diffState = [numpy.zeros((self.nstate, self.npivot))]
            diffControl = []

            # derivative of loss with respect to piovt
            loss = 0.0
            dlpiovt = numpy.zeros(self.npivot)
            # path
            for t in range(horizon):
                # orginial system
                currControl = self.pivotFun(t, pivot).full().flatten()
                currState = state[t]
                control += [currControl]
                nextState = self.dynFun(currState, currControl).full().flatten()  # next state
                state += [nextState]

                # backprop system
                Ft = self.dfxFun(currState, currControl).full()
                Gt = self.dfuFun(currState, currControl).full()
                currdupivot = self.dpivotFun(t, pivot).full()
                currdxpivot = diffState[t]
                diffControl += [currdupivot]
                nextdxpivot = numpy.dot(Ft, currdxpivot) + numpy.dot(Gt, currdupivot)
                diffState += [nextdxpivot]

                # loss function
                loss = loss + self.pathObjFun(currState, currControl)
                currdlx = self.dcxFun(currState, currControl).full()
                currdlu = self.dcuFun(currState, currControl).full()
                currdlpivot = numpy.dot(currdlx, currdxpivot) + numpy.dot(currdlu, currdupivot)
                dlpiovt = dlpiovt + currdlpivot.flatten()
            # final
            loss = loss + self.finalObjFun(state[-1]).full()
            currdlx = self.dhxFun(state[-1]).full()
            dlpiovt = dlpiovt + (numpy.dot(currdlx, diffState[-1])).flatten()

            self.eta = self.eta * 0.9999
            # store and update
            lossTrace += [loss]
            pivotTrace += [pivot]
            pivot = pivot - self.eta * dlpiovt
            print(loss)

            iterIndex = iterIndex + 1
            if iterIndex > maxIter or numpy.linalg.norm(dlpiovt) < epsilon:
                break

        stateOpt = [state[i].tolist() for i in range(len(state))]
        controlOpt = [(control[i]) for i in range(len(control))]
        lossTraceOut = [lossTrace[i].full() for i in range(len(lossTrace))]

        print(lossTraceOut)
        print(pivot)
        optSol = {'stateOpt': stateOpt,
                  'controlOpt': controlOpt,
                  'loss': lossTraceOut[-1],
                  'lossTrace': lossTraceOut,
                  'pivot': pivot,
                  'pivotTrace': pivotTrace,
                  'iniPivot': self.initialPivot}

        return optSol

class systemID:
    def __init__(self, projectName='model'):
        self.projectName = projectName

    def setAuxvar(self, auxvar=SX.sym('auxvar', 1)):
        self.auxvar = auxvar
        self.nauxvar = self.auxvar.numel()

    def setStatevar(self, state, stateLB=[], stateUB=[]):
        self.state = state
        self.nstate = self.state.numel()
        if len(stateLB) == self.nstate:
            self.stateLB = stateLB
        else:
            self.stateLB = self.nstate * [-1e20]

        if len(stateUB) == self.nstate:
            self.stateUB = stateUB
        else:
            self.stateUB = self.nstate * [1e20]

    def setControlvar(self, control, controlLB=[], controlUB=[]):
        self.control = control
        self.ncontrol = self.control.numel()

        if len(controlLB) == self.ncontrol:
            self.controlLB = controlLB
        else:
            self.controlLB = self.ncontrol * [-1e20]

        if len(controlUB) == self.ncontrol:
            self.controlUB = controlUB
        else:
            self.controlUB = self.ncontrol * [1e20]

    def setDyn(self, ode):
        self.dynFun = casadi.Function('dynFun', [self.state, self.control, self.auxvar], [ode])
        self.dyn = ode

    def setLoss(self):
        self.obState = SX.sym('Xo', self.nstate)
        self.loss = dot(self.state - self.obState, self.state - self.obState)
        self.lossFun = casadi.Function('lossFun', [self.state, self.obState], [self.loss])

    def generateIDDiff(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'control'), "Define the control variable first!"
        assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        assert hasattr(self, 'loss'), "Define the system dynamics first!"

        # first order differential of dynamics
        self.dfx = jacobian(self.dyn, self.state)
        self.dfxFun = casadi.Function('dfx', [self.state, self.control, self.auxvar], [self.dfx])
        self.dfu = jacobian(self.dyn, self.control)
        self.dfuFun = casadi.Function('dfu', [self.state, self.control, self.auxvar], [self.dfu])
        self.dfe = jacobian(self.dyn, self.auxvar)
        self.dfeFun = casadi.Function('dfe', [self.state, self.control, self.auxvar], [self.dfe])
        # first order differential loss function
        self.dlx = jacobian(self.loss, self.state)
        self.dlxFun = casadi.Function('dlx', [self.state, self.obState], [self.dlx])

    def getInputSignal(self, random=True, typeSignal='linear'):
        if random==True:
            a=numpy.random.rand(1).item()
            if a<0.3:
                typeSignal='linear'
            elif 0.4<a<0.6:
                typeSignal='quadratic'
            else:
                typeSignal='others'

        if typeSignal == 'linear':
            def controlFun(t):
                return self.ncontrol*[t]

            return controlFun
        elif typeSignal == 'quadratic':
            def controlFun(t):
                return self.ncontrol*[t ** 2]

            return controlFun
        else:
            def controlFun(t):
                return self.ncontrol*[t ** 3]

        return controlFun

    def interpolation(self,x,y, method=1):
        if method==1:
            ipFun=ip.interp1d(x,y,axis=0)
            return ipFun

    def integrateDyn(self, auxvar, inistate,horizon, inputFun):
        # construct the dynamical ode used for integration
        def dynODE(t,x,inputFun,auxvar):
            u=inputFun(t)
            xdot=self.dynFun(x,u,auxvar).full().flatten()
            return xdot

        # do the integration
        t_span=[0.0,horizon]
        sol=solve_ivp(lambda t,x:dynODE(t,x,inputFun,auxvar),t_span,inistate,t_eval=numpy.linspace(0,horizon,endpoint = True))
        tGrid=sol.t
        stateGrid=sol.y.T
        # store
        stateFun=self.interpolation(tGrid,stateGrid)
        return  stateFun

    def sysID(self, batchInput, batchObservedData, initialAuxvar=[], eta=1e-3, maxIter=1.0000e4, epsilon=1.0000e-08,
              alpha=0.99):
        # set initial guess
        if len(initialAuxvar) > 0:
            self.initialAuxvar = initialAuxvar
        else:
            self.initialAuxvar = numpy.random.rand(self.nauxvar)
        self.eta = eta
        self.maxIter = maxIter
        self.epsilon = epsilon

        # preprocessing the data
        batch = len(batchObservedData)
        batchIniState = [batchObservedData[i][0, :] for i in range(batch)]

        # do the planning
        iterIndex = 0
        lossTrace = [1e20]  # for storage
        auxvarTrace = []
        auxvar = self.initialAuxvar

        while True:
            loss = 0.0
            dlauxvar = numpy.zeros(self.nauxvar)

            for i in range(batch):
                # take the data
                iniState = batchIniState[i]
                input = batchInput[i]
                observedData = batchObservedData[i]

                # orginial system
                state = numpy.empty((input.shape[0] + 1, self.nstate))
                state[0, :] = iniState

                # backprop system
                diffstate = [numpy.zeros((self.nstate, self.nauxvar))]
                # path
                for t in range(input.shape[0]):
                    currx = state[t, :]
                    curru = input[t, :]
                    nextx = self.dynFun(currx, curru, auxvar).full().flatten()
                    state[t + 1, :] = nextx
                    Ft = self.dfxFun(currx, curru, auxvar).full()
                    Et = self.dfeFun(currx, curru, auxvar).full()
                    currdx = diffstate[t]
                    nextdx = numpy.dot(Ft, currdx) + Et
                    diffstate += [nextdx]

                    loss += self.lossFun(currx, observedData[t, :]).full()
                    currdlx = self.dlxFun(currx, observedData[t, :]).full()
                    dlauxvar += numpy.dot(currdlx, currdx).flatten()
                # final
                loss += self.lossFun(state[-1], observedData[-1, :]).full()
                currdlx = self.dlxFun(state[-1], observedData[-1, :]).full()
                dlauxvar += numpy.dot(currdlx, diffstate[-1]).flatten()

            # update
            iterIndex += 1
            lossTrace += [numpy.asscalar(loss)]
            auxvarTrace += [auxvar]
            self.eta = self.eta * alpha
            auxvar = auxvar - self.eta * dlauxvar
            print(numpy.asscalar(loss))

            if iterIndex > self.maxIter or numpy.fabs(loss - lossTrace[-2]) < self.epsilon:
                break

        sol = {'auxvar': auxvar,
               'lossTrace': lossTrace[1:],
               'auxvarTrace': auxvarTrace,
               'iniAuxvar': self.initialAuxvar}

        return sol

# other function facility

