'''
# This module is a simulation environment, which provides different-level (from easy to hard)
# simulation benchmark environments and animation facilities for the user to test their learning algorithm.
# This environment is versatile to use, e.g. the user can arbitrarily:
# set the parameters for the dynamics and objective function,
# obtain the analytical dynamics models, as well as the differentiations.
# define and modify the control cost function
# animate the motion of the system.

# Do NOT distribute without written permission from Wanxin Jin
# Do NOT use it for any commercial purpose

# Contact email: wanxinjin@gmail.com
# Last update: May. 15, 2020

#

'''

from casadi import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch
import math
import time


# quadrotor (UAV) environment
class Quadrotor:
    def __init__(self, project_name='my UAV'):
        self.project_name = 'my uav'

        # define the state of the quadrotor
        rx, ry, rz = SX.sym('rx'), SX.sym('ry'), SX.sym('rz')
        self.r_I = vertcat(rx, ry, rz)
        vx, vy, vz = SX.sym('vx'), SX.sym('vy'), SX.sym('vz')
        self.v_I = vertcat(vx, vy, vz)
        # quaternions attitude of B w.r.t. I
        q0, q1, q2, q3 = SX.sym('q0'), SX.sym('q1'), SX.sym('q2'), SX.sym('q3')
        self.q = vertcat(q0, q1, q2, q3)
        wx, wy, wz = SX.sym('wx'), SX.sym('wy'), SX.sym('wz')
        self.w_B = vertcat(wx, wy, wz)
        # define the quadrotor input
        f1, f2, f3, f4 = SX.sym('f1'), SX.sym('f2'), SX.sym('f3'), SX.sym('f4')
        self.T_B = vertcat(f1, f2, f3, f4)

    def initDyn(self, Jx=None, Jy=None, Jz=None, mass=None, l=None, c=None):
        # global parameter
        g = 10

        # parameters settings
        parameter = []
        if Jx is None:
            self.Jx = SX.sym('Jx')
            parameter += [self.Jx]
        else:
            self.Jx = Jx

        if Jy is None:
            self.Jy = SX.sym('Jy')
            parameter += [self.Jy]
        else:
            self.Jy = Jy

        if Jz is None:
            self.Jz = SX.sym('Jz')
            parameter += [self.Jz]
        else:
            self.Jz = Jz

        if mass is None:
            self.mass = SX.sym('mass')
            parameter += [self.mass]
        else:
            self.mass = mass

        if l is None:
            self.l = SX.sym('l')
            parameter += [self.l]
        else:
            self.l = l

        if c is None:
            self.c = SX.sym('c')
            parameter += [self.c]
        else:
            self.c = c

        self.dyn_auxvar = vcat(parameter)

        # Angular moment of inertia
        self.J_B = diag(vertcat(self.Jx, self.Jy, self.Jz))
        # Gravity
        self.g_I = vertcat(0, 0, -g)
        # Mass of rocket, assume is little changed during the landing process
        self.m = self.mass

        # total thrust in body frame
        thrust = self.T_B[0] + self.T_B[1] + self.T_B[2] + self.T_B[3]
        self.thrust_B = vertcat(0, 0, thrust)
        # total moment M in body frame
        Mx = -self.T_B[1] * self.l / 2 + self.T_B[3] * self.l / 2
        My = -self.T_B[0] * self.l / 2 + self.T_B[2] * self.l / 2
        Mz = (self.T_B[0] - self.T_B[1] + self.T_B[2] - self.T_B[3]) * self.c
        self.M_B = vertcat(Mx, My, Mz)

        # cosine directional matrix
        C_B_I = self.dir_cosine(self.q)  # inertial to body
        C_I_B = transpose(C_B_I)  # body to inertial

        # Newton's law
        dr_I = self.v_I
        dv_I = 1 / self.m * mtimes(C_I_B, self.thrust_B) + self.g_I
        # Euler's law
        dq = 1 / 2 * mtimes(self.omega(self.w_B), self.q)
        dw = mtimes(inv(self.J_B), self.M_B - mtimes(mtimes(self.skew(self.w_B), self.J_B), self.w_B))

        self.X = vertcat(self.r_I, self.v_I, self.q, self.w_B)
        self.U = self.T_B
        self.f = vertcat(dr_I, dv_I, dq, dw)

    def initCost(self, wr=None, wv=None, wq=None, ww=None, wthrust=0.1):

        parameter = []
        if wr is None:
            self.wr = SX.sym('wr')
            parameter += [self.wr]
        else:
            self.wr = wr

        if wv is None:
            self.wv = SX.sym('wv')
            parameter += [self.wv]
        else:
            self.wv = wv

        if wq is None:
            self.wq = SX.sym('wq')
            parameter += [self.wq]
        else:
            self.wq = wq

        if ww is None:
            self.ww = SX.sym('ww')
            parameter += [self.ww]
        else:
            self.ww = ww

        self.cost_auxvar = vcat(parameter)

        # goal position in the world frame
        goal_r_I = np.array([0, 0, 0])
        self.cost_r_I = dot(self.r_I - goal_r_I, self.r_I - goal_r_I)

        # goal velocity
        goal_v_I = np.array([0, 0, 0])
        self.cost_v_I = dot(self.v_I - goal_v_I, self.v_I - goal_v_I)

        # final attitude error
        goal_q = toQuaternion(0, [0, 0, 1])
        goal_R_B_I = self.dir_cosine(goal_q)
        R_B_I = self.dir_cosine(self.q)
        self.cost_q = trace(np.identity(3) - mtimes(transpose(goal_R_B_I), R_B_I))

        # auglar velocity cost
        goal_w_B = np.array([0, 0, 0])
        self.cost_w_B = dot(self.w_B - goal_w_B, self.w_B - goal_w_B)

        # the thrust cost
        self.cost_thrust = dot(self.T_B, self.T_B)

        self.path_cost = self.wr * self.cost_r_I + \
                         self.wv * self.cost_v_I + \
                         self.ww * self.cost_w_B + \
                         self.wq * self.cost_q + \
                         wthrust * self.cost_thrust
        self.final_cost = self.wr * self.cost_r_I + \
                          self.wv * self.cost_v_I + \
                          self.ww * self.cost_w_B + \
                          self.wq * self.cost_q

    def initCost2(self, wthrust=0.1):

        parameter = []

        self.wrx = SX.sym('wrx')
        parameter += [self.wrx]
        self.wry = SX.sym('wry')
        parameter += [self.wry]
        self.wrz = SX.sym('wrz')
        parameter += [self.wrz]

        self.wvx = SX.sym('wvx')
        parameter += [self.wvx]
        self.wvy = SX.sym('wvy')
        parameter += [self.wvy]
        self.wvz = SX.sym('wvz')
        parameter += [self.wvz]

        self.wwx = SX.sym('wwx')
        parameter += [self.wwx]
        self.wwy = SX.sym('wwy')
        parameter += [self.wwy]
        self.wwz = SX.sym('wwz')
        parameter += [self.wwz]

        self.wq = SX.sym('wq')
        parameter += [self.wq]

        self.cost_auxvar = vcat(parameter)

        # goal position in the world frame
        goal_r_I = np.array([0, 0, 5])
        self.cost_r_I_x = (self.r_I[0] - goal_r_I[0]) ** 2
        self.cost_r_I_y = (self.r_I[1] - goal_r_I[1]) ** 2
        self.cost_r_I_z = (self.r_I[2] - goal_r_I[2]) ** 2

        # goal velocity
        goal_v_I = np.array([0, 0, 0])
        self.cost_v_I_x = (self.v_I[0] - goal_v_I[0]) ** 2
        self.cost_v_I_y = (self.v_I[1] - goal_v_I[1]) ** 2
        self.cost_v_I_z = (self.v_I[2] - goal_v_I[2]) ** 2

        # final attitude error
        goal_q = toQuaternion(0, [0, 0, 1])
        goal_R_B_I = self.dir_cosine(goal_q)
        R_B_I = self.dir_cosine(self.q)
        self.cost_q = trace(np.identity(3) - mtimes(transpose(goal_R_B_I), R_B_I))

        # auglar velocity cost
        goal_w_B = np.array([0, 0, 0])
        self.cost_w_B_x = (self.w_B[0] - goal_w_B[0]) ** 2
        self.cost_w_B_y = (self.w_B[1] - goal_w_B[1]) ** 2
        self.cost_w_B_z = (self.w_B[2] - goal_w_B[2]) ** 2

        # the thrust cost
        self.cost_thrust = dot(self.T_B, self.T_B)

        self.path_cost = self.wrx * self.cost_r_I_x + self.wry * self.cost_r_I_y + self.wrz * self.cost_r_I_z + \
                         self.wvx * self.cost_v_I_x + self.wvy * self.cost_v_I_y + self.wvz * self.cost_v_I_z + \
                         self.wwx * self.cost_w_B_x + self.wwy * self.cost_w_B_y + self.wwz * self.cost_w_B_z + \
                         self.wq * self.cost_q + \
                         wthrust * self.cost_thrust
        self.final_cost = self.wrx * self.cost_r_I_x + self.wry * self.cost_r_I_y + self.wrz * self.cost_r_I_z + \
                          self.wvx * self.cost_v_I_x + self.wvy * self.cost_v_I_y + self.wvz * self.cost_v_I_z + \
                          self.wwx * self.cost_w_B_x + self.wwy * self.cost_w_B_y + self.wwz * self.cost_w_B_z + \
                          self.wq * self.cost_q

    def initCost_Polynomial(self, w_thrust=0.1, goal_r_I=np.array([8, 8, 0])):

        parameter = []

        # goal aspect
        self.cost_goal_r = dot(self.r_I - goal_r_I, self.r_I - goal_r_I)

        # velocity aspect
        goal_v_I = np.array([0, 0, 0])
        self.cost_goal_v = dot(self.v_I - goal_v_I, self.v_I - goal_v_I)

        # orientation aspect
        goal_q = toQuaternion(0, [0, 0, 1])
        goal_R_B_I = self.dir_cosine(goal_q)
        R_B_I = self.dir_cosine(self.q)
        self.cost_goal_q = trace(np.identity(3) - mtimes(transpose(goal_R_B_I), R_B_I))

        # angular aspect
        goal_w_B = np.array([0, 0, 0])
        self.cost_goal_w = dot(self.w_B - goal_w_B, self.w_B - goal_w_B)

        # thrust aspect
        self.cost_thrust = dot(self.T_B, self.T_B)

        # features for x
        self.w_xsq = SX.sym('w_xsq')
        self.feature_xsq = 0.5 * self.r_I[0] * self.r_I[0]
        parameter += [self.w_xsq]
        self.w_x = SX.sym('w_x')
        self.feature_x = self.r_I[0]
        parameter += [self.w_x]

        # features for y
        self.w_ysq = SX.sym('w_ysq')
        self.feature_ysq = 0.5 * self.r_I[1] * self.r_I[1]
        parameter += [self.w_ysq]
        self.w_y = SX.sym('w_y')
        self.feature_y = self.r_I[1]
        parameter += [self.w_y]

        # features for z
        self.w_zsq = SX.sym('w_zsq')
        self.feature_zsq = 0.5 * self.r_I[2] * self.r_I[2]
        parameter += [self.w_zsq]
        self.w_z = SX.sym('w_z')
        self.feature_z = self.r_I[2]
        parameter += [self.w_z]

        # # feature for xy
        self.w_xy = SX.sym('w_xy')
        self.feature_xy = self.r_I[0] * self.r_I[1]
        parameter += [self.w_xy]

        # # feature for xz
        self.w_xz = SX.sym('w_xz')
        self.feature_xz = self.r_I[0] * self.r_I[2]
        parameter += [self.w_xz]

        # # feature for xz
        self.w_yz = SX.sym('w_yz')
        self.feature_yz = self.r_I[1] * self.r_I[2]
        parameter += [self.w_yz]

        self.path_cost = self.w_xsq * self.feature_xsq + self.w_x * self.feature_x + \
                         self.w_ysq * self.feature_ysq + self.w_y * self.feature_y + \
                         self.w_zsq * self.feature_zsq + self.w_z * self.feature_z + \
                         self.w_xy * self.feature_xy + \
                         self.w_xz * self.feature_xz + \
                         self.w_yz * self.feature_yz + \
                         w_thrust * self.cost_thrust

        self.final_cost = 10 * self.cost_goal_r + \
                          1 * self.cost_goal_v + \
                          1 * self.cost_goal_q + \
                          1 * self.cost_goal_w
        self.cost_auxvar = vcat(parameter)

    def get_quadrotor_position(self, wing_len, state_traj):

        # thrust_position in body frame
        r1 = vertcat(wing_len / 2, 0, 0)
        r2 = vertcat(0, -wing_len / 2, 0)
        r3 = vertcat(-wing_len / 2, 0, 0)
        r4 = vertcat(0, wing_len / 2, 0)

        # horizon
        horizon = np.size(state_traj, 0)
        position = np.zeros((horizon, 15))
        for t in range(horizon):
            # position of COM
            rc = state_traj[t, 0:3]
            # altitude of quaternion
            q = state_traj[t, 6:10]
            q = q / (np.linalg.norm(q) + 0.00001)

            # direction cosine matrix from body to inertial
            CIB = np.transpose(self.dir_cosine(q).full())

            # position of each rotor in inertial frame
            r1_pos = rc + mtimes(CIB, r1).full().flatten()
            r2_pos = rc + mtimes(CIB, r2).full().flatten()
            r3_pos = rc + mtimes(CIB, r3).full().flatten()
            r4_pos = rc + mtimes(CIB, r4).full().flatten()

            # store
            position[t, 0:3] = rc
            position[t, 3:6] = r1_pos
            position[t, 6:9] = r2_pos
            position[t, 9:12] = r3_pos
            position[t, 12:15] = r4_pos

        return position

    def play_animation(self, wing_len, state_traj, state_traj_ref=None, dt=0.1, save_option=0, title='UAV Maneuvering',
                       horizon=1, waypoints=None):

        # plot
        # params = {'axes.labelsize': 25,
        #           'axes.titlesize': 25,
        #           'xtick.labelsize': 20,
        #           'ytick.labelsize': 20,
        #           'legend.fontsize': 16}
        # plt.rcParams.update(params)

        fig = plt.figure(0, figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (m)', fontsize=15, labelpad=15)
        ax.set_ylabel('Y (m)', fontsize=15, labelpad=15)
        ax.set_zlabel('Z (m)', fontsize=15, labelpad=15)
        ax.set_zlim(0, 12)
        ax.set_ylim(-9, 9)
        ax.set_xlim(-9, 9)
        ax.set_title('UAV manuvering', pad=15, fontsize=20)
        time_template = 'time = %.1fs'
        time_text = ax.text2D(0.8, 0.9, "time", transform=ax.transAxes, fontsize=15)

        # fig = plt.figure(figsize=(7,6))
        # ax = fig.add_subplot(1, 1, 1, projection='3d', )
        # ax.set_xlabel('X', fontsize=40, labelpad=10)
        # ax.set_ylabel('Y', fontsize=40, labelpad=10)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # ax.set_ylim(-9, 9)
        # ax.set_xlim(-9, 9)
        # ax.set_xticks(np.arange(-8, 9, 4))
        # ax.set_yticks(np.arange(-8, 9, 8))
        # ax.tick_params(labelbottom=False, labelright=False, labelleft=False)
        # ax.view_init(elev=88, azim=-90)
        # ax.set_title('Top view', fontsize=40, pad=-25)
        # ax.set_position([-0.17, -0.12, 1.30, 1.15])
        # time_template = 'time = %.1fs'
        # time_text = ax.text2D(0.55, 0.20, "time", transform=ax.transAxes, fontsize=0)

        # fig = plt.figure(figsize=(7,6))
        # ax = fig.add_subplot(1, 1, 1, projection='3d', )
        # ax.set_xlabel('X', fontsize=40, labelpad=10)
        # ax.set_zlabel('Z', fontsize=40, labelpad=10)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # ax.set_ylim(-9, 9)
        # ax.set_xlim(-9, 9)
        # ax.set_zlim(0, 8)
        # ax.set_zticks(np.arange(0, 8, 2))
        # ax.set_xticks(np.arange(-8, 9, 4))
        # # ax.set_yticks(np.arange(-8, 9, 8))
        # ax.tick_params(labelbottom=False, labelright=False, labelleft=False)
        # # ax.view_init(elev=0, azim=-90)
        # ax.set_title('Front view', fontsize=40, pad=-25)
        # ax.set_position([-0.17, -0.12, 1.30, 1.15])
        # time_template = 'time = %.1fs'
        # time_text = ax.text2D(0.55, 0.20, "time", transform=ax.transAxes, fontsize=0)

        # draw the quadrotor
        # bar2_back = ax.bar3d([-1], [-3], [0], dx=[0.5], dy=[0.5], dz=[4.5], color='#D95319')
        # bar2_top = ax.bar3d([-1], [-3], [4], dx=[0.5], dy=[-3.5], dz=[0.5], color='#D95319')
        # bar2_front = ax.bar3d([-1], [-6.5], [4.5], dx=[0.5], dy=[0.5], dz=[-4.5], color='#D95319')
        #
        # bar1_front = ax.bar3d([2.5], [2], [1.5], dx=[0.5], dy=[0.5], dz=[4.5], color='#D95319')
        # bar1_bottom = ax.bar3d([2.5], [2], [1.5], dx=[0.5], dy=[4.5], dz=[0.5], color='#D95319')
        # bar1_top = ax.bar3d([2.5], [2.5], [5.5], dx=[0.5], dy=[4.5], dz=[0.5], color='#D95319')
        # bar1_back = ax.bar3d([2.5], [6.5], [6.0], dx=[0.5], dy=[0.5], dz=[-4.5], color='#D95319')

        if waypoints is not None:
            ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], s=80, zorder=10000, color='red', alpha=1,
                       marker='^')

        # data
        position = self.get_quadrotor_position(wing_len, state_traj)
        sim_horizon = np.size(position, 0)
        time_interval = float(horizon / sim_horizon)

        if state_traj_ref is None:
            position_ref = self.get_quadrotor_position(0, numpy.zeros_like(position))
        else:
            position_ref = self.get_quadrotor_position(wing_len, state_traj_ref)

        # animation
        line_traj, = ax.plot(position[:1, 0], position[:1, 1], position[:1, 2])
        c_x, c_y, c_z = position[0, 0:3]
        r1_x, r1_y, r1_z = position[0, 3:6]
        r2_x, r2_y, r2_z = position[0, 6:9]
        r3_x, r3_y, r3_z = position[0, 9:12]
        r4_x, r4_y, r4_z = position[0, 12:15]
        line_arm1, = ax.plot([c_x, r1_x], [c_y, r1_y], [c_z, r1_z], linewidth=4, color='blue', marker='o', markersize=4,
                             markerfacecolor='black')
        line_arm2, = ax.plot([c_x, r2_x], [c_y, r2_y], [c_z, r2_z], linewidth=4, color='red', marker='o', markersize=4,
                             markerfacecolor='black')
        line_arm3, = ax.plot([c_x, r3_x], [c_y, r3_y], [c_z, r3_z], linewidth=4, color='blue', marker='o', markersize=4,
                             markerfacecolor='black', )
        line_arm4, = ax.plot([c_x, r4_x], [c_y, r4_y], [c_z, r4_z], linewidth=4, color='red', marker='o', markersize=4,
                             markerfacecolor='black', )

        line_traj_ref, = ax.plot(position_ref[:1, 0], position_ref[:1, 1], position_ref[:1, 2], color='gray', alpha=0.5)
        c_x_ref, c_y_ref, c_z_ref = position_ref[0, 0:3]
        r1_x_ref, r1_y_ref, r1_z_ref = position_ref[0, 3:6]
        r2_x_ref, r2_y_ref, r2_z_ref = position_ref[0, 6:9]
        r3_x_ref, r3_y_ref, r3_z_ref = position_ref[0, 9:12]
        r4_x_ref, r4_y_ref, r4_z_ref = position_ref[0, 12:15]
        line_arm1_ref, = ax.plot([c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref], [c_z_ref, r1_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)
        line_arm2_ref, = ax.plot([c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref], [c_z_ref, r2_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)
        line_arm3_ref, = ax.plot([c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref], [c_z_ref, r3_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)
        line_arm4_ref, = ax.plot([c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref], [c_z_ref, r4_z_ref], linewidth=2,
                                 color='gray', marker='o', markersize=3, alpha=0.7)

        # customize
        if state_traj_ref is not None:
            plt.legend([line_traj, line_traj_ref], ['learned', 'OC solver'], ncol=1, loc='best',
                       bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))

        def update_traj(num):

            # customize
            time_text.set_text(time_template % (num * time_interval))

            # trajectory
            line_traj.set_data(position[:num, 0], position[:num, 1])
            line_traj.set_3d_properties(position[:num, 2])

            # uav
            c_x, c_y, c_z = position[num, 0:3]
            r1_x, r1_y, r1_z = position[num, 3:6]
            r2_x, r2_y, r2_z = position[num, 6:9]
            r3_x, r3_y, r3_z = position[num, 9:12]
            r4_x, r4_y, r4_z = position[num, 12:15]

            line_arm1.set_data(np.array([[c_x, r1_x], [c_y, r1_y]]))
            line_arm1.set_3d_properties(np.array([c_z, r1_z]))

            line_arm2.set_data(np.array([[c_x, r2_x], [c_y, r2_y]]))
            line_arm2.set_3d_properties(np.array([c_z, r2_z]))

            line_arm3.set_data(np.array([[c_x, r3_x], [c_y, r3_y]]))
            line_arm3.set_3d_properties(np.array([c_z, r3_z]))

            line_arm4.set_data(np.array([[c_x, r4_x], [c_y, r4_y]]))
            line_arm4.set_3d_properties(np.array([c_z, r4_z]))

            # trajectory ref
            num = sim_horizon - 1
            line_traj_ref.set_data(position_ref[:num, 0], position_ref[:num, 1])
            line_traj_ref.set_3d_properties(position_ref[:num, 2])

            # uav ref
            c_x_ref, c_y_ref, c_z_ref = position_ref[num, 0:3]
            r1_x_ref, r1_y_ref, r1_z_ref = position_ref[num, 3:6]
            r2_x_ref, r2_y_ref, r2_z_ref = position_ref[num, 6:9]
            r3_x_ref, r3_y_ref, r3_z_ref = position_ref[num, 9:12]
            r4_x_ref, r4_y_ref, r4_z_ref = position_ref[num, 12:15]

            line_arm1_ref.set_data(np.array([[c_x_ref, r1_x_ref], [c_y_ref, r1_y_ref]]))
            line_arm1_ref.set_3d_properties(np.array([c_z_ref, r1_z_ref]))

            line_arm2_ref.set_data(np.array([[c_x_ref, r2_x_ref], [c_y_ref, r2_y_ref]]))
            line_arm2_ref.set_3d_properties(np.array([c_z_ref, r2_z_ref]))

            line_arm3_ref.set_data(np.array([[c_x_ref, r3_x_ref], [c_y_ref, r3_y_ref]]))
            line_arm3_ref.set_3d_properties(np.array([c_z_ref, r3_z_ref]))

            line_arm4_ref.set_data(np.array([[c_x_ref, r4_x_ref], [c_y_ref, r4_y_ref]]))
            line_arm4_ref.set_3d_properties(np.array([c_z_ref, r4_z_ref]))

            return line_traj, line_arm1, line_arm2, line_arm3, line_arm4, \
                   line_traj_ref, line_arm1_ref, line_arm2_ref, line_arm3_ref, line_arm4_ref, time_text

        ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=80, blit=True, cache_frame_data=False)

        if save_option != 0:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
            ani.save('uav_5_learning_3d.mp4', writer=writer, dpi=300)
            print('save_success')

        plt.show()

    def dir_cosine(self, q):
        C_B_I = vertcat(
            horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
            horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
            horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
        )
        return C_B_I

    def skew(self, v):
        v_cross = vertcat(
            horzcat(0, -v[2], v[1]),
            horzcat(v[2], 0, -v[0]),
            horzcat(-v[1], v[0], 0)
        )
        return v_cross

    def omega(self, w):
        omeg = vertcat(
            horzcat(0, -w[0], -w[1], -w[2]),
            horzcat(w[0], 0, w[2], -w[1]),
            horzcat(w[1], -w[2], 0, w[0]),
            horzcat(w[2], w[1], -w[0], 0)
        )
        return omeg

    def quaternion_mul(self, p, q):
        return vertcat(p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                       p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                       p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                       p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
                       )


# converter to quaternion from (angle, direction)
def toQuaternion(angle, dir):
    if type(dir) == list:
        dir = numpy.array(dir)
    dir = dir / numpy.linalg.norm(dir)
    quat = numpy.zeros(4)
    quat[0] = math.cos(angle / 2)
    quat[1:] = math.sin(angle / 2) * dir
    return quat.tolist()


# normalized verctor
def normalizeVec(vec):
    if type(vec) == list:
        vec = np.array(vec)
    vec = vec / np.linalg.norm(vec)
    return vec


def quaternion_conj(q):
    conj_q = q
    conj_q[1] = -q[1]
    conj_q[2] = -q[2]
    conj_q[3] = -q[3]
    return conj_q
