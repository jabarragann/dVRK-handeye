"""
Copied from  surgical robotics challenge repository
https://github.com/surgical-robotics-ai/surgical_robotics_challenge/blob/d5624606d1a168c518b1bb9fe09cbf6bc274927e/scripts/surgical_robotics_challenge/kinematics/psmFK.py

Validation of the FK model against real data is done in the scripts 

"""

import numpy as np
from enum import Enum

PI = np.pi
PI_2 = np.pi / 2


class JointType(Enum):
    REVOLUTE = 0
    PRISMATIC = 1


class Convention(Enum):
    STANDARD = 0
    MODIFIED = 1


class DH:
    def __init__(self, alpha, a, theta, d, offset, joint_type, convention):
        self.alpha = alpha
        self.a = a
        self.theta = theta
        self.d = d
        self.offset = offset
        self.joint_type = joint_type
        self.convention = convention

    def mat_from_dh(self, alpha, a, theta, d, offset, joint_type, convention):
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        th = 0.0
        if joint_type == JointType.REVOLUTE:
            th = theta + offset
        elif joint_type == JointType.PRISMATIC:
            d = d + offset + theta
        else:
            assert (
                joint_type == JointType.REVOLUTE and joint_type == JointType.PRISMATIC
            )
            return

        ct = np.cos(th)
        st = np.sin(th)

        if convention == Convention.STANDARD:
            mat = np.mat(
                [
                    [ct, -st * ca, st * sa, a * ct],
                    [st, ct * ca, -ct * sa, a * st],
                    [0, sa, ca, d],
                    [0, 0, 0, 1],
                ]
            )
        elif convention == Convention.MODIFIED:
            mat = np.mat(
                [
                    [ct, -st, 0, a],
                    [st * ca, ct * ca, -sa, -d * sa],
                    [st * sa, ct * sa, ca, d * ca],
                    [0, 0, 0, 1],
                ]
            )
        else:
            raise "ERROR, DH CONVENTION NOT UNDERSTOOD"

        return mat

    def get_trans(self):
        return self.mat_from_dh(
            self.alpha,
            self.a,
            self.theta,
            self.d,
            self.offset,
            self.joint_type,
            self.convention,
        )


def enforce_limits(j_raw, lower_lims, upper_lims):
    num_joints = len(j_raw)
    j_limited = [0.0] * num_joints

    for idx in range(num_joints):
        min_lim = lower_lims[idx]
        max_lim = upper_lims[idx]
        j_limited[idx] = max(min_lim, min(j_raw[idx], max_lim))

    return j_limited


# THIS IS THE FK FOR THE PSM MOUNTED WITH THE LARGE NEEDLE DRIVER TOOL. THIS IS THE
# SAME KINEMATIC CONFIGURATION FOUND IN THE DVRK MANUAL. NOTE, JUST LIKE A FAULT IN THE
# MTM's DH PARAMETERS IN THE MANUAL, THERE IS A FAULT IN THE PSM's DH AS WELL. BASED ON
# THE FRAME ATTACHMENT IN THE DVRK MANUAL THE CORRECT DH CAN FOUND IN THIS FILE

# ALSO, NOTICE THAT AT HOME CONFIGURATION THE TIP OF THE PSM HAS THE FOLLOWING
# ROTATION OFFSET W.R.T THE BASE. THIS IS IMPORTANT FOR IK PURPOSES.
# R_7_0 = [ 0,  1,  0 ]
#       = [ 1,  0,  0 ]
#       = [ 0,  0, -1 ]
# Basically, x_7 is along y_0, y_7 is along x_0 and z_7 is along -z_0.

# You need to provide a list of joint positions. If the list is less that the number of joint
# i.e. the robot has 6 joints, but only provide 3 joints. The FK till the 3+1 link will be provided


class PSMKinematicData:
    def __init__(self):
        self.num_links = 7

        self.L_rcc = 0.4318  # From dVRK documentation x 10
        self.L_tool = 0.4162  # From dVRK documentation x 10
        self.L_pitch2yaw = 0.0091  # Fixed length from the palm joint to the pinch joint
        self.L_yaw2ctrlpnt = 0.0  # Fixed length from the pinch joint to the pinch tip
        self.L_tool2rcm_offset = 0.0229  # Distance between tool tip and the Remote Center of Motion at Home Pose

        # PSM DH Params
        # alpha | a | theta | d | offset | type
        # fmt: off
        self.kinematics = [
            DH(PI_2, 0, 0, 0, PI_2, JointType.REVOLUTE, Convention.MODIFIED),
            DH(-PI_2, 0, 0, 0, -PI_2, JointType.REVOLUTE, Convention.MODIFIED),
            DH(PI_2, 0, 0, 0, -self.L_rcc, JointType.PRISMATIC, Convention.MODIFIED),
            DH(0, 0, 0, self.L_tool, 0, JointType.REVOLUTE, Convention.MODIFIED),
            DH(-PI_2, 0, 0, 0, -PI_2, JointType.REVOLUTE, Convention.MODIFIED),
            DH( -PI_2, self.L_pitch2yaw, 0, 0, -PI_2, JointType.REVOLUTE, Convention.MODIFIED,),
            DH( -PI_2, 0, 0, self.L_yaw2ctrlpnt, PI_2, JointType.REVOLUTE, Convention.MODIFIED,),
        ]
        # fmt: on

        self.lower_limits = [
            np.deg2rad(-91.96),
            np.deg2rad(-60),
            -0.0,
            np.deg2rad(-175),
            np.deg2rad(-90),
            np.deg2rad(-85),
        ]

        self.upper_limits = [
            np.deg2rad(91.96),
            np.deg2rad(60),
            0.240,
            np.deg2rad(175),
            np.deg2rad(90),
            np.deg2rad(85),
        ]

    def get_link_params(self, link_num):
        if link_num < 0 or link_num > self.num_links:
            # Error
            print("ERROR, ONLY ", self.num_links, " JOINT DEFINED")
            return []
        else:
            return self.kinematics[link_num]


kinematics_data = PSMKinematicData()


def compute_FK(joint_pos, up_to_link):
    if up_to_link > kinematics_data.num_links:
        raise "ERROR! COMPUTE FK UP_TO_LINK GREATER THAN DOF"
    j = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(joint_pos)):
        j[i] = joint_pos[i]

    T_N_0 = np.identity(4)

    for i in range(up_to_link):
        link_dh = kinematics_data.get_link_params(i)
        link_dh.theta = j[i]
        T_N_0 = T_N_0 * link_dh.get_trans()

    return T_N_0


def round_mat(mat, rows, cols, precision=4):
    for i in range(0, rows):
        for j in range(0, cols):
            mat[i, j] = round(mat[i, j], precision)
    return mat


if __name__ == "__main__":

    joints_pose = [0.0, 0, 0.2, 0, 0, 0]
    T_7_0 = compute_FK(joints_pose, 7)
    T_6_0 = compute_FK(joints_pose, 6)
    T_5_0 = compute_FK(joints_pose, 5)

    print("\n AFTER ROUNDING \n")
    print("Matrix T_7_0")
    print(round_mat(T_7_0, 4, 4, 3))

    print("Matrix T_6_0")
    print(round_mat(T_6_0, 4, 4, 3))

    print("Matrix T_5_0")
    print(round_mat(T_5_0, 4, 4, 3))
