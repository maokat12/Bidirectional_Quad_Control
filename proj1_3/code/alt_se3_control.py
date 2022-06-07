import numpy as np
from scipy.spatial.transform import Rotation as Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params, k_p, k_d, k_r, k_w):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE

        #k_p = 20
        #k_d = 900
        k_p = k_p
        k_d = k_d #bro what the fuck
        k_w = k_w
        k_r = k_r
        self.K_p = np.array([[k_p, 0, 0],
                             [0, k_p, 0],
                             [0, 0, k_p]])

        self.K_d = np.array([[k_d, 0, 0],
                             [0, k_d, 0],
                             [0, 0, k_d]])

        self.K_w = np.array([[60, 0, 0],
                             [0, 60, 0],
                             [0, 0, 60]])

        self.K_r = np.array([[2430, 0, 0],
                             [0, 2430, 0],
                             [0, 0, 400]])

    def vee_map(self, R):
        #R -3 3x3  skew symmetric matrix
        c = R[1][0]
        b = R[0][2]
        a = R[2][1]

        #print('input R: ', R)
        #print('vee output: ', [a, b, c])

        return np.array([a, b, c])

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        #exit()

        # STUDENT CODE HERE
        #print('time: ', t)
        #print('current location: ', state["x"])
        #print('desired location: ', flat_output["x"])
        #print('desired velocity: ', flat_output["x_dot"])
        r_dott_des = flat_output["x_ddot"] - np.matmul(self.K_d, (state["v"]-flat_output["x_dot"])) - np.matmul(self.K_p, (state["x"] - flat_output["x"])) #desired acceleration
        #print('r dotdot des: ', r_dott_des)
        F_des = self.mass*r_dott_des + np.array([0, 0, self.mass*self.g]) #desired force
        #print('F des: ', F_des)

        #calculate rotaiton matrix from input quaternion
        q0 = state["q"][3] #w
        q1 = state["q"][0] #i
        q2 = state["q"][1] #j
        q3 = state["q"][2] #k

        #https://automaticaddison.com/wp-content/uploads/2020/09/quaternion-to-rotation-matrix.jpg
        R = np.array([[2*(q0**2+q1**2)-1, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
                      [2*(q1*q2+q0*q3), 2*(q0**2+q2**2)-1, 2*(q2*q3-q0*q1)],
                      [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 2*(q0**2+q3**2)-1]])

        R = Rotation.from_quat(state["q"]).as_matrix()

        #print('R: ', R)

        a_yaw = np.array([np.cos(flat_output["yaw"]),
                          np.sin(flat_output["yaw"]),
                          0])

        #print('a yaw: ', a_yaw)

        b3_des = F_des / np.linalg.norm(F_des)
        b2_des = np.cross(b3_des, a_yaw) / np.linalg.norm(np.cross(b3_des, a_yaw))
        b1_des = np.cross(b2_des, b3_des)

        #print('b3 des: ', b3_des)
        #print('b2 des: ', b2_des)
        #print('b1 des: ', b1_des)

        R_des = np.array([b1_des, b2_des, b3_des]).T

        #print('R des: ', R_des)

        e_R = 0.5*self.vee_map(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        #print('error R: ', e_R)
        e_W = state["w"] - 0 #setting desired angular velocity to 0
        #print('error W:', e_W)

        #desired inputs
        b3 = R@np.array([[0],[0],[1]])
        #print('b3', b3)
        u1 = np.matmul(b3.T, F_des)
        u2 = np.matmul(self.inertia, (-np.matmul(self.K_r, e_R) - np.matmul(self.K_w, e_W)))

        #print('u1: ', u1)
        #print('u2: ', u2)

        #calculate corresponding forces
        u = np.array([u1[0], u2[0], u2[1], u2[2]]) #input matrix

        #print('u: ', u)
        #follows the mellinger paper
        K = np.array([[self.k_thrust, self.k_thrust, self.k_thrust, self.k_thrust],
                      [0, self.k_thrust*self.arm_length, 0, -self.k_thrust*self.arm_length],
                      [-self.k_thrust*self.arm_length, 0, self.k_thrust*self.arm_length, 0],
                      [self.k_drag, -self.k_drag, self.k_drag, -self.k_drag]])

        #gamma = self.k_drag/self.k_thrust
        #K = np.array([[1, 1, 1, 1],
        #              [0, self.arm_length, 0, -self.arm_length],
        #              [-self.arm_length, 0, self.arm_length, 0],
        #              [gamma, -gamma, gamma, -gamma]])

        #print('K: ', K)

        cmd_motor_speeds = np.matmul(np.linalg.inv(K),u) #motor speeds^2
        #make sure motor speeds are physically feasible:
        for i in range(len(cmd_motor_speeds)):
            if cmd_motor_speeds[i] > self.rotor_speed_max**2:
                cmd_motor_speeds[i] = self.rotor_speed_max**2
            elif cmd_motor_speeds[i] < self.rotor_speed_min:
                cmd_motor_speeds[i] = self.rotor_speed_min
        cmd_thrust = self.k_thrust*cmd_motor_speeds #motor thrusts
        cmd_moment = self.k_drag*cmd_motor_speeds #motor moments
        cmd_motor_speeds = np.sqrt(cmd_motor_speeds)

        #print('cmd motor speeds: ', cmd_motor_speeds)
        #print('cmd thrust: ', cmd_thrust)
        #print('cmd moment: ', cmd_moment)


        #calculate desired quaternion
        #https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
        a = (1+R_des[0][0]+R_des[1][1]+R_des[2][2])/4
        if abs(a) < 0.0001:
            a = 0
        mag_q0 = np.sqrt(a)
        b = (1+R_des[0][0]-R_des[1][1]-R_des[2][2])/4
        if abs(b) < 0.0001:
            b = 0
        mag_q1 = np.sqrt(b)
        c = (1-R_des[0][0]+R_des[1][1]-R_des[2][2])/4
        if abs(c) < 0.0001:
            c = 0
        mag_q2 = np.sqrt(c)
        d = (1-R_des[0][0]-R_des[1][1]+R_des[2][2])/4
        if abs(d) < 0.0001:
            d = 0
        mag_q3 = np.sqrt(d)

        #print('mag q0: ', mag_q0)
        #print('mag q1: ', mag_q1)
        #print('mag q2: ', mag_q2)
        #print('mag q3: ', mag_q3)

        q0 = None
        q1 = None
        q2 = None
        q3 = None

        if max([mag_q0, mag_q1, mag_q2, mag_q3]) == mag_q0:
            q0 = mag_q0
            q1 = (R[2][1] - R[1][2])/(4*q0)
            q2 = (R[0][2] - R[2][0])/(4*q0)
            q3 = (R[1][0] - R[0][1])/(4*q0)

        elif max([mag_q0, mag_q1, mag_q2, mag_q3]) == mag_q1:
            q1 = mag_q1
            q0 = (R[2][1] - R[1][2])/(4*q1)
            q2 = (R[0][1] + R[1][0])/(4*q1)
            q3 = (R[0][2] + R[2][0])/(4*q1)

        elif max([mag_q0, mag_q1, mag_q2, mag_q3]) == mag_q2:
            q2 = mag_q2
            q0 = (R[0][2] - R[2][0])/(4*q2)
            q1 = (R[0][1] + R[1][0])/(4*q2)
            q3 = (R[1][2] + R[2][1])/(4*q2)
        else:
            q3 = mag_q3
            q0 = (R[1][0] - R[0][1])/(4*q3)
            q1 = (R[0][2] + R[2][0])/(4*q3)
            q2 = (R[1][2] + R[2][1])/(4*q3)

        cmd_q = [q1, q2, q3, q0]

        #print('cmd q: ', cmd_q)

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}

        #exit()

        return control_input
