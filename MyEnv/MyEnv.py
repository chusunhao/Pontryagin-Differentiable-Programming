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


# Rocket environment
class Satellite:
    def __init__(self, project_name='spacecraft formation flying'):
        self.project_name = project_name

        # define the rock states
        rx, ry, rz = SX.sym('rx'), SX.sym('ry'), SX.sym('rz')
        self.r_I = vertcat(rx, ry, rz)
        vx, vy, vz = SX.sym('vx'), SX.sym('vy'), SX.sym('vz')
        self.v_I = vertcat(vx, vy, vz)

        # define the rock input
        ux, uy, uz = SX.sym('ux'), SX.sym('uy'), SX.sym('uz')
        self.u = vertcat(ux, uy, uz)

    def initDyn(self, n=None):

        # parameters settings
        parameter = []
        if n is None:
            self.n = SX.sym('n')
            parameter += [self.n]
        else:
            self.n = n

        self.dyn_auxvar = vcat(parameter)

        A1 = np.array([[3 * n ** 2, 0, 0], [0, 0, 0], [0, 0, -n ** 2]])
        A2 = np.array([[0, 2 * n, 0], [-2 * n, 0, 0], [0, 0, 0]])
        A = np.block([[np.zeros((3, 3)), np.eye(3)], [A1, A2]])
        B = np.block([[np.zeros((3, 3))], [np.eye(3)]])

        dr_I = self.v_I
        dv_I = A1 @ self.r_I + A2 @ self.v_I + self.u

        self.X = vertcat(self.r_I, self.v_I)
        self.U = self.u
        self.f = vertcat(dr_I, dv_I)

    def initCost(self, Q=None, R=None):

        parameter = []

        if Q is None:
            self.Q = SX.sym('Q', 6, 6)
            parameter += [self.Q]
        else:
            self.Q = Q

        if R is None:
            self.R = SX.sym('R', 3, 3)
            parameter += [self.R]
        else:
            self.R = R

        self.cost_auxvar = vcat(parameter)

        # goal position and velocity (states) in the LVLH frame
        goal_X = np.array([1.22464680e-13, -2.00000000e+03, 2.44929360e-13, -1.03815925e+00,
                           -2.54275680e-16, -2.07631849e+00])
        self.cost_X = (self.X - goal_X).T @ self.Q @ (self.X - goal_X)

        # the thrust cost
        self.cost_U = self.U.T  @ self.R @ self.U

        self.path_cost = self.cost_X + self.cost_U
        self.final_cost = self.cost_X + self.cost_U

    # # def play_animation(self, rocket_len, state_traj, control_traj, state_traj_ref=None, control_traj_ref=None,
    # #                    save_option=0, dt=0.1,
    # #                    title='Rocket Powered Landing'):
    # #     fig = plt.figure()
    # #     ax = fig.add_subplot(111, projection='3d')
    # #     ax.set_xlabel('East (m)')
    # #     ax.set_ylabel('North (m)')
    # #     ax.set_zlabel('Upward (m)')
    # #     ax.set_zlim(0, 10)
    # #     ax.set_ylim(-8, 8)
    # #     ax.set_xlim(-8, 8)
    # #     ax.set_title(title, pad=20, fontsize=15)
    # #
    # #     # target landing point
    # #     p = Circle((0, 0), 3, color='g', alpha=0.3)
    # #     ax.add_patch(p)
    # #     art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
    # #
    # #     # data
    # #     position = self.get_rocket_body_position(rocket_len, state_traj, control_traj)
    # #     sim_horizon = np.size(position, 0)
    # #     for t in range(np.size(position, 0)):
    # #         x = position[t, 0]
    # #         if x < 0:
    # #             sim_horizon = t
    # #             break
    # #     # animation
    # #     line_traj, = ax.plot(position[:1, 1], position[:1, 2], position[:1, 0])
    # #     xg, yg, zg, xh, yh, zh, xf, yf, zf = position[0, 3:]
    # #     line_rocket, = ax.plot([yg, yh], [zg, zh], [xg, xh], linewidth=5, color='black')
    # #     line_force, = ax.plot([yg, yf], [zg, zf], [xg, xf], linewidth=2, color='red')
    # #
    # #     # reference data
    # #     if state_traj_ref is None or control_traj_ref is None:
    # #         position_ref=numpy.zeros_like(position)
    # #         sim_horizon_ref=sim_horizon
    # #     else:
    # #         position_ref = self.get_rocket_body_position(rocket_len, state_traj_ref, control_traj_ref)
    # #         sim_horizon_ref = np.size((position_ref,0))
    # #         for t in range(np.size(position_ref, 0)):
    # #             x = position_ref[t, 0]
    # #             if x < 0:
    # #                 sim_horizon_ref = t
    # #                 break
    # #     # animation
    # #     line_traj_ref, = ax.plot(position_ref[:1, 1], position_ref[:1, 2], position_ref[:1, 0], linewidth=2, color='gray', alpha=0.5)
    # #     xg_ref, yg_ref, zg_ref, xh_ref, yh_ref, zh_ref, xf_ref, yf_ref, zf_ref = position_ref[0, 3:]
    # #     line_rocket_ref, = ax.plot([yg_ref, yh_ref], [zg_ref, zh_ref], [xg_ref, xh_ref], linewidth=5, color='gray', alpha=0.5)
    # #     line_force_ref, = ax.plot([yg_ref, yf_ref], [zg_ref, zf_ref], [xg_ref, xf_ref], linewidth=2, color='red', alpha=0.2)
    # #
    # #     # time label
    # #     time_template = 'time = %.1fs'
    # #     time_text = ax.text2D(0.66, 0.55, "time", transform=ax.transAxes)
    # #     # time_text = ax.text2D(0.66, 0.65, "time", transform=ax.transAxes)
    # #     # time_text = ax.text2D(0.50, 0.65, "time", transform=ax.transAxes)
    # #
    # #     # customize
    # #     if state_traj_ref is not None or control_traj_ref is not None:
    # #         plt.legend([line_traj, line_traj_ref], ['learned', 'truth'], ncol=1, loc='best',
    # #                    bbox_to_anchor=(0.35, 0.25, 0.5, 0.5))
    #
    #     def update_traj(num):
    #         # customize
    #         time_text.set_text(time_template % (num * dt))
    #
    #         # trajectory
    #         if num> sim_horizon:
    #             t=sim_horizon
    #         else:
    #             t=num
    #         line_traj.set_data(position[:t, 1], position[:t, 2])
    #         line_traj.set_3d_properties(position[:t, 0])
    #
    #         # satellite
    #         xg, yg, zg, xh, yh, zh, xf, yf, zf = position[t, 3:]
    #         line_rocket.set_data([yg, yh], [zg, zh])
    #         line_rocket.set_3d_properties([xg, xh])
    #         line_force.set_data([yg, yf], [zg, zf])
    #         line_force.set_3d_properties([xg, xf])
    #
    #         # reference
    #         if num> sim_horizon_ref:
    #             t_ref=sim_horizon_ref
    #         else:
    #             t_ref=num
    #         line_traj_ref.set_data(position_ref[:t_ref, 1], position_ref[:t_ref, 2])
    #         line_traj_ref.set_3d_properties(position_ref[:t_ref, 0])
    #
    #         # satellite
    #         xg_ref, yg_ref, zg_ref, xh_ref, yh_ref, zh_ref, xf_ref, yf_ref, zf_ref = position_ref[num, 3:]
    #         line_rocket_ref.set_data([yg_ref, yh_ref], [zg_ref, zh_ref])
    #         line_rocket_ref.set_3d_properties([xg_ref, xh_ref])
    #         line_force_ref.set_data([yg_ref, yf_ref], [zg_ref, zf_ref])
    #         line_force_ref.set_3d_properties([xg_ref, xf_ref])
    #
    #
    #         return line_traj, line_rocket, line_force, line_traj_ref, line_rocket_ref, line_force_ref,  time_text
    #
    #     ani = animation.FuncAnimation(fig, update_traj, max(sim_horizon,sim_horizon_ref), interval=100, blit=True)
    #
    #     if save_option != 0:
    #         Writer = animation.writers['ffmpeg']
    #         writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
    #         ani.save(title + '.mp4', writer=writer, dpi=300)
    #         print('save_success')
    #
    #     plt.show()

    def get_rocket_body_position(self, rocket_len, state_traj, control_traj):

        # thrust_position in body frame
        r_T_B = vertcat(-rocket_len / 2, 0, 0)

        # horizon
        horizon = np.size(control_traj, 0)
        # for normalization in the plot
        norm_f = np.linalg.norm(control_traj, axis=1);
        max_f = np.amax(norm_f)
        position = np.zeros((horizon, 12))
        for t in range(horizon):
            # position of COM
            rc = state_traj[t, 0:3]
            # altitude of quaternion
            q = state_traj[t, 6:10]
            # thrust force
            f = control_traj[t, 0:3]

            # direction cosine matrix from body to inertial
            CIB = np.transpose(self.dir_cosine(q).full())

            # position of gimbal point (satellite tail)
            rg = rc + mtimes(CIB, r_T_B).full().flatten()
            # position of satellite tip
            rh = rc - mtimes(CIB, r_T_B).full().flatten()

            # direction of force
            df = np.dot(CIB, f) / max_f
            rf = rg - df

            # store
            position[t, 0:3] = rc
            position[t, 3:6] = rg
            position[t, 6:9] = rh
            position[t, 9:12] = rf

        return position


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
