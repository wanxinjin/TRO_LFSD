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


def get_quadrotor_position(wing_len, state_traj):
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
        CIB = np.transpose(dir_cosine(q).full())

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


def play_animation( wing_len, state_traj, state_traj_ref=None, dt=0.1, save_option=0, title='UAV Maneuvering',
                   horizon=1, waypoints=None):
    # plot
    # params = {'axes.labelsize': 25,
    #           'axes.titlesize': 25,
    #           'xtick.labelsize': 20,
    #           'ytick.labelsize': 20,
    #           'legend.fontsize': 16}
    # plt.rcParams.update(params)

    fig =plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect(aspect = (3,4,3))
    ax.set_xlabel('X (m)', fontsize=15, labelpad=15)
    ax.set_ylabel('Y (m)', fontsize=15, labelpad=15)
    ax.set_zlabel('Z (m)', fontsize=15, labelpad=15)
    ax.set_zlim(0, 3)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2, 2)
    ax.set_title('UAV manuvering', pad=15, fontsize=20)
    ax.view_init(elev=81, azim=-90)

    time_template = 'time = %.1fs'
    time_text = ax.text2D(0.8, 0.9, "time", transform=ax.transAxes, fontsize=15)



    if waypoints is not None:
        ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], s=80, zorder=10000, color='red', alpha=1,
                   marker='^')

    # data
    position = get_quadrotor_position(wing_len, state_traj)
    sim_horizon = np.size(position, 0)
    time_interval = float(horizon / sim_horizon)

    if state_traj_ref is None:
        position_ref = get_quadrotor_position(0, numpy.zeros_like(position))
    else:
        position_ref = get_quadrotor_position(wing_len, state_traj_ref)

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


def dir_cosine(q):
    C_B_I = vertcat(
        horzcat(1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])),
        horzcat(2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])),
        horzcat(2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))
    )
    return C_B_I

def skew(v):
    v_cross = vertcat(
        horzcat(0, -v[2], v[1]),
        horzcat(v[2], 0, -v[0]),
        horzcat(-v[1], v[0], 0)
    )
    return v_cross

def omega( w):
    omeg = vertcat(
        horzcat(0, -w[0], -w[1], -w[2]),
        horzcat(w[0], 0, w[2], -w[1]),
        horzcat(w[1], -w[2], 0, w[0]),
        horzcat(w[2], w[1], -w[0], 0)
    )
    return omeg

def quaternion_mul( p, q):
    return vertcat(p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                   p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                   p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                   p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]
                   )