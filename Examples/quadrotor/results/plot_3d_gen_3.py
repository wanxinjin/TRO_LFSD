import numpy as np

from CPDP import CPDP
from JinEnv import JinEnv
from casadi import *

import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


# plot the generalization test true
if True:

    # Plot the learned trajectory
    trials = [0,1,2,4,6,7]
    loss_trace_list=[]
    learned_param_list=[]
    time_tau=None
    waypoints=None
    for trial in trials:
        load = np.load('./case_5/quadrotor_trial_'+str(trial)+'_2.npy', allow_pickle=True).item()
        loss_trace_list.append(load['loss_trace'])
        learned_param_list.append(load['parameter_trace'][-1])
        time_tau = load['time_tau']
        waypoints = load['waypoints']

    learned_param_mean=np.mean(np.array(learned_param_list), axis=0)
    print(learned_param_mean)
    learned_param_std=np.std(np.array(learned_param_list), axis=0)


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
    oc.setIntegrator(n_grid=30)
    # --------------------------- set initial condition and horizon ------------------------------
    # set initial condition
    ini_r_I = [-8, 3, 1.]
    ini_v_I = [10, -20, 0]
    ini_q = JinEnv.toQuaternion(1, [0.1, 0.1, 0.3])
    ini_w = [0.0, 0.0, 0.0]
    ini_state = ini_r_I + ini_v_I + ini_q + ini_w
    T = 1

    _, opt_sol = oc.cocSolver(ini_state, T, learned_param_mean)
    time_grid = np.linspace(0, T, num=200)  # generate the time inquiry grid with N is the point number
    position = env.get_quadrotor_position(wing_len=1.8, state_traj=opt_sol(time_grid)[:, 0:oc.n_state])

    # compute the average nearest distance of the trajectory to the given waypoints
    nearest_distance = []
    for waypoint in waypoints:
        distance = 10000
        # sear the nearest trajectory point
        for pos_t in position:
            xyz = pos_t[0:3]
            dist_t = norm_2(waypoint - xyz)
            if dist_t < distance:
                distance = dist_t
        nearest_distance += [distance.full().item()]
    print('average nearest_distance:',np.array(nearest_distance).mean())


    # compare with policy-imitation
    if True:
        waypoints_rx = waypoints[:, 0]
        waypoints_ry = waypoints[:, 1]
        waypoints_rz = waypoints[:, 2]
        spl_rx = UnivariateSpline(time_tau, waypoints_rx)
        spl_ry = UnivariateSpline(time_tau, waypoints_ry)
        spl_rz = UnivariateSpline(time_tau, waypoints_rz)
        plot_time = np.linspace(0, 1, 100)
        waypoints_rx = spl_rx(plot_time)
        waypoints_ry = spl_ry(plot_time)
        waypoints_rz = spl_rz(plot_time)
        waypoints_interpolate = np.vstack((waypoints_rx, waypoints_ry, waypoints_rz)).T

        # check which waypoint is nearest to the new starting point
        nearest_index = np.argmin(np.linalg.norm(waypoints - np.array(ini_r_I), axis=1))
        waypoints_news=np.vstack((np.array(ini_r_I),waypoints[nearest_index:,:]))
        time_tau_new=np.hstack((0, time_tau[nearest_index:]))
        spl_rx_new = UnivariateSpline(time_tau_new, waypoints_news[:,0], k=2)
        spl_ry_new = UnivariateSpline(time_tau_new, waypoints_news[:,1], k=2)
        spl_rz_new = UnivariateSpline(time_tau_new, waypoints_news[:,2], k=2)


        generalize_rx = spl_rx_new(plot_time)
        generalize_ry = spl_ry_new(plot_time)
        generalize_rz = spl_rz_new(plot_time)

        imitation_position = np.vstack((generalize_rx, generalize_ry, generalize_rz)).T

        # compute the average nearest distance of the trajectory to the given waypoints
        nearest_distance = []
        for waypoint in waypoints:
            distance = 10000
            # sear the nearest trajectory point
            for pos_t in imitation_position:
                xyz = pos_t[0:3]
                dist_t = norm_2(waypoint - xyz)
                if dist_t < distance:
                    distance = dist_t
            nearest_distance += [distance.full().item()]
        print('average_nearest_distance for policy imitation:', np.array(nearest_distance).mean())

    # compte a





    # plot
    import matplotlib.pyplot as plt

    params = {'axes.labelsize': 25,
              'axes.titlesize': 25,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'legend.fontsize': 16}
    plt.rcParams.update(params)

    fig = plt.figure(figsize=(8.5, 5.0))

    ax = fig.add_subplot(1, 3, 2, projection='3d', )
    ax.set_xlabel('X', fontsize=15, labelpad=-0)
    ax.set_ylabel('Y', fontsize=15, labelpad=-5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_ylim(-9, 9)
    ax.set_xlim(-9, 9)
    ax.set_xticks(np.arange(-8, 9, 4))
    ax.set_yticks(np.arange(-8, 9, 4))
    ax.tick_params(labelbottom=False, labelright=False, labelleft=False)
    ax.view_init(elev=88, azim=-90)
    bar2 = ax.bar3d([-1], [-4], [1], dx=[0.5], dy=[0.5], dz=[3.5], color='#D95319')
    bar22 = ax.bar3d([-1], [-4], [4], dx=[0.5], dy=[-3.5], dz=[0.5], color='#D95319')
    bar23 = ax.bar3d([-1], [-7.5], [4], dx=[0.5], dy=[0.5], dz=[-3], color='#D95319', zorder=-100)
    bar24 = ax.bar3d([-1], [-4], [1], dx=[0.5], dy=[-3.5], dz=[0.5], color='#D95319', zorder=-100)

    bar1 = ax.bar3d([0.5], [0], [3.5], dx=[0.5], dy=[0.5], dz=[3], color='#D95319')
    bar11 = ax.bar3d([0.5], [0], [3.5], dx=[0.5], dy=[4.5], dz=[0.5], color='#D95319')
    bar12 = ax.bar3d([0.5], [0], [6.5], dx=[0.5], dy=[4.5], dz=[0.5], color='#D95319')
    bar13 = ax.bar3d([0.5], [4], [7.0], dx=[0.5], dy=[0.5], dz=[-3.5], color='#D95319')
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], s=56, zorder=10000, color='red', alpha=1, marker='^')
    ax.plot(position[:, 0], position[:, 1], position[:, 2], zorder=-100, color='blue',linewidth=2,)
    ax.plot(imitation_position[:, 0], imitation_position[:, 1], imitation_position[:, 2], zorder=-100, color='c',linewidth=2,)

    time_step = [0, -1]
    for t in time_step:
        c_x, c_y, c_z = position[t, 0:3]
        r1_x, r1_y, r1_z = position[t, 3:6]
        r2_x, r2_y, r2_z = position[t, 6:9]
        r3_x, r3_y, r3_z = position[t, 9:12]
        r4_x, r4_y, r4_z = position[t, 12:15]
        line_arm1, = ax.plot([c_x, r1_x], [c_y, r1_y], [c_z, r1_z], linewidth=3, color='blue', marker='o', markersize=4,
                             markerfacecolor='black', zorder=100)
        line_arm2, = ax.plot([c_x, r2_x], [c_y, r2_y], [c_z, r2_z], linewidth=3, color='red', marker='o', markersize=4,
                             markerfacecolor='black', zorder=100)
        line_arm3, = ax.plot([c_x, r3_x], [c_y, r3_y], [c_z, r3_z], linewidth=3, color='blue', marker='o', markersize=4,
                             markerfacecolor='black', zorder=100)
        line_arm4, = ax.plot([c_x, r4_x], [c_y, r4_y], [c_z, r4_z], linewidth=3, color='red', marker='o', markersize=4,
                             markerfacecolor='black', zorder=100)

    ax.set_title('Top view', fontsize=20, pad=-15)
    ax.set_position([0.65, 0.46, 0.35, 0.50])

    ax = fig.add_subplot(1, 3, 3, projection='3d', )
    ax.set_xlabel('X', fontsize=15, labelpad=0)
    ax.set_zlabel('Z', fontsize=15, labelpad=-5)
    ax.set_zlim(0, 8)
    ax.set_zticks(np.arange(0, 10, 2))
    # ax.set_ylim(-8, 9)
    ax.set_yticks([])
    ax.set_xlim(-9, 9)
    ax.set_xticks(np.arange(-8, 9, 4))
    ax.tick_params(labelbottom=False, labelright=False, labelleft=False)
    # ax.set_yticks(np.arange(-8, 9, 4))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.view_init(elev=0, azim=-90)
    bar2 = ax.bar3d([-1], [-4], [1], dx=[0.5], dy=[0.5], dz=[3.5], color='#D95319')
    bar22 = ax.bar3d([-1], [-4], [4], dx=[0.5], dy=[-3.5], dz=[0.5], color='#D95319')
    bar23 = ax.bar3d([-1], [-7.5], [4], dx=[0.5], dy=[0.5], dz=[-3], color='#D95319', zorder=-100)
    bar24 = ax.bar3d([-1], [-4], [1], dx=[0.5], dy=[-3.5], dz=[0.5], color='#D95319', zorder=-100)

    bar1 = ax.bar3d([0.5], [0], [3.5], dx=[0.5], dy=[0.5], dz=[3], color='#D95319')
    bar11 = ax.bar3d([0.5], [0], [3.5], dx=[0.5], dy=[4.5], dz=[0.5], color='#D95319')
    bar12 = ax.bar3d([0.5], [0], [6.5], dx=[0.5], dy=[4.5], dz=[0.5], color='#D95319')
    bar13 = ax.bar3d([0.5], [4], [7.0], dx=[0.5], dy=[0.5], dz=[-3.5], color='#D95319')
    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], s=56, zorder=-10000, color='red', marker='^', alpha=1)
    seg1,seg2,seg3,seg4=40,45,141,151
    ax.plot(position[:seg1, 0], position[:seg1, 1], position[:seg1, 2], zorder=10, color='blue',linewidth=2,)
    ax.plot(position[seg2:seg3, 0], position[seg2:seg3, 1], position[seg2:seg3, 2], zorder=10, color='blue',linewidth=2,)
    ax.plot(position[seg4:, 0], position[seg4:, 1], position[seg4:, 2], zorder=10, color='blue',linewidth=2,)
    # ax.plot(position[:, 0], position[:, 1], position[:, 2], zorder=100, color='blue')
    ax.plot(imitation_position[:, 0], imitation_position[:, 1], imitation_position[:, 2], zorder=-100, color='c',linewidth=2,)

    time_step = [0, -1]
    for t in time_step:
        c_x, c_y, c_z = position[t, 0:3]
        r1_x, r1_y, r1_z = position[t, 3:6]
        r2_x, r2_y, r2_z = position[t, 6:9]
        r3_x, r3_y, r3_z = position[t, 9:12]
        r4_x, r4_y, r4_z = position[t, 12:15]
        line_arm1, = ax.plot([c_x, r1_x], [c_y, r1_y], [c_z, r1_z], linewidth=3, color='blue', marker='o', markersize=4,
                             markerfacecolor='black', zorder=100)
        line_arm2, = ax.plot([c_x, r2_x], [c_y, r2_y], [c_z, r2_z], linewidth=3, color='red', marker='o', markersize=4,
                             markerfacecolor='black', zorder=100)
        line_arm3, = ax.plot([c_x, r3_x], [c_y, r3_y], [c_z, r3_z], linewidth=3, color='blue', marker='o', markersize=4,
                             markerfacecolor='black', zorder=100)
        line_arm4, = ax.plot([c_x, r4_x], [c_y, r4_y], [c_z, r4_z], linewidth=3, color='red', marker='o', markersize=4,
                             markerfacecolor='black', zorder=100)
    ax.set_title('Front view', fontsize=20, pad=-30)
    ax.set_position([0.65, 0.0, 0.35, 0.48])

    ax = fig.add_subplot(1, 3, 1, projection='3d', )
    ax.set_xlabel('X (m)', fontsize=20, labelpad=18)
    ax.set_ylabel('Y (m)', fontsize=20, labelpad=18)
    ax.set_zlabel('Z (m)', fontsize=20, labelpad=12)
    ax.set_zlim(0, 8)
    ax.set_ylim(-9, 9)
    ax.set_xlim(-9, 9)
    ax.set_xticks(np.arange(-8, 9, 4))
    ax.set_yticks(np.arange(-8, 9, 8))
    ax.set_zticks(np.arange(0, 10, 2))
    ax.view_init(elev=24, azim=-65)
    bar2 = ax.bar3d([-1], [-4], [1], dx=[0.5], dy=[0.5], dz=[3.5], color='#D95319')
    bar22 = ax.bar3d([-1], [-4], [4], dx=[0.5], dy=[-3.5], dz=[0.5], color='#D95319')
    bar23 = ax.bar3d([-1], [-7.5], [4], dx=[0.5], dy=[0.5], dz=[-3], color='#D95319', zorder=-100)
    bar24 = ax.bar3d([-1], [-4], [1], dx=[0.5], dy=[-3.5], dz=[0.5], color='#D95319', zorder=-100)

    bar1 = ax.bar3d([0.5], [0], [3.5], dx=[0.5], dy=[0.5], dz=[3], color='#D95319')
    bar11 = ax.bar3d([0.5], [0], [3.5], dx=[0.5], dy=[4.5], dz=[0.5], color='#D95319')
    bar12 = ax.bar3d([0.5], [0], [6.5], dx=[0.5], dy=[4.5], dz=[0.5], color='#D95319')
    bar13 = ax.bar3d([0.5], [4], [7.0], dx=[0.5], dy=[0.5], dz=[-3.5], color='#D95319')

    ax.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], s=56, zorder=0, color='red', alpha=1, marker='^')
    seg1,seg2,seg3,seg4=29,37,137,144
    ax.plot(position[:seg1, 0], position[:seg1, 1], position[:seg1, 2], zorder=1000, color='blue',linewidth=3, label='the proposed')
    ax.plot(position[seg2:seg3, 0], position[seg2:seg3, 1], position[seg2:seg3, 2], zorder=100, color='blue',linewidth=3,)
    ax.plot(position[seg4:, 0], position[seg4:, 1], position[seg4:, 2], zorder=1000, color='blue',linewidth=3,)
    # ax.plot(position[:, 0], position[:, 1], position[:, 2], zorder=100, color='blue')
    imitation_seg1, imitation_seg2=67, 68
    ax.plot(imitation_position[0:imitation_seg1, 0], imitation_position[:imitation_seg1, 1], imitation_position[:imitation_seg1, 2], zorder=-100, color='c',linewidth=3,label='kinematic learning')
    ax.plot(imitation_position[imitation_seg2:, 0], imitation_position[imitation_seg2:, 1], imitation_position[imitation_seg2:, 2], zorder=1000, color='c',linewidth=3,)
    ax.legend(loc='best', bbox_to_anchor=(0.1, 0.4, 0.5, 0.5))

    time_step = [0, -1]
    for t in time_step:
        c_x, c_y, c_z = position[t, 0:3]
        r1_x, r1_y, r1_z = position[t, 3:6]
        r2_x, r2_y, r2_z = position[t, 6:9]
        r3_x, r3_y, r3_z = position[t, 9:12]
        r4_x, r4_y, r4_z = position[t, 12:15]
        line_arm1, = ax.plot([c_x, r1_x], [c_y, r1_y], [c_z, r1_z], linewidth=3, color='blue', marker='o', markersize=5,
                             markerfacecolor='black', zorder=100)
        line_arm2, = ax.plot([c_x, r2_x], [c_y, r2_y], [c_z, r2_z], linewidth=3, color='red', marker='o', markersize=5,
                             markerfacecolor='black', zorder=100)
        line_arm3, = ax.plot([c_x, r3_x], [c_y, r3_y], [c_z, r3_z], linewidth=3, color='blue', marker='o', markersize=5,
                             markerfacecolor='black', zorder=100)
        line_arm4, = ax.plot([c_x, r4_x], [c_y, r4_y], [c_z, r4_z], linewidth=3, color='red', marker='o', markersize=5,
                             markerfacecolor='black', zorder=100)

    ax.set_position([-0.04, 0.01, 0.72, 1.1])

    plt.show()

