import numpy as np
from scipy.spatial.transform import Rotation as Rotation

class HopfFibrationControl(object):
    """

    """
    def __init__(self, quad_params):
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
        self.rotor_speed_min = quad_params['rotor_speed_bi_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2
        self.quad_sign = 1 #1-upright, -1-invert | orientation of quad in world coordinates

        # STUDENT CODE HERE

        #k_p = 20
        #k_d = 900


        #k_w = 1
        #k_r = 1

        #THESE WORK: 7, 10, 24, 125
        #THESE WORK AND SETTLE WITHIN 2 SECONDS: 9, 4, 24, 450 @ 1 m/s (fails > 1.5
        #THESE WORK UP TO 2.25 ms: 5, 4, 24, 450
        #THE SET I SUBMITTED: 4, 3, 22, 445
        #THIS SET ALSO WORKS AT 2.5m/s - 7, 5, 25, 430
        #oscillations: 2, 10, 10, 35
        #2.0 settling time: 7, 5, 18, 450 - pos_yaw
        #2.5 settling time: 25, 22, 55, 420
        #traj timeeout: 7, 10, 24, 105
        #also kinda works but fails the cube test: 7, 10, 24, 220

        #the ones that griffon helped me in office hours find but still didn't work 7, 10, 19, 450

        #wiggly but close: 10, 30, 34, 110

        #this kind of works - 3, 350, 4, 10

        #harmonic oscillation lmao; 15, 20, 19, 420

        k_p =8.5 #5
        k_d = 5#3.5  # bro what the fuck
        k = 2.2*self.g
        kp = 2.3*self.g

        self.K_p = np.array([[4, 0, 0],
                             [0, 4, 0],
                             [0, 0, 5.0]])

        self.K_d = np.array([[4, 0, 0],
                             [0, 4, 0],
                             [0, 0, 4.0]])

        self.K_w = np.array([[k, 0, 0],
                             [0, k, 0],
                             [0, 0, k]])

        self.K_r = np.array([[k, 0, 0],
                             [0, k, 0],
                             [0, 0, k]])

        k_p =6 #5
        k_d = 3.5#3.5 f
        #kw - 23, 23 16
        #kr - 430 430 131

        self.K_p = np.array([[k_p, 0, 0],
                             [0, k_p, 0],
                             [0, 0, k_p]])

        self.K_d = np.array([[k_d, 0, 0],
                             [0, k_d, 0],
                             [0, 0, k_d]])

        self.K_w = np.array([[72, 0, 0],
                             [0, 72, 0],
                             [0, 0, 72]])

        self.K_r = np.array([[3700, 0, 0],
                             [0, 3700, 0],
                             [0, 0, 3700]])
        self.i = 0

    def vee_map(self, R):
        c = R[1][0]
        b = R[0][2]
        a = R[2][1]
        return np.array([a, b, c])

    def quaternion_multiply(self, Q0, Q1):
        """
        Multiplies two quaternions.
        Quaternion format: w + xi + yj + zk

        Input
        :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31)
        :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32)

        Output
        :return: A 4 element array containing the final quaternion (q03,q13,q23,q33)

        """
        # Extract the values from Q0
        w0 = Q0[0]
        x0 = Q0[1]
        y0 = Q0[2]
        z0 = Q0[3]

        # Extract the values from Q1
        w1 = Q1[0]
        x1 = Q1[1]
        y1 = Q1[2]
        z1 = Q1[3]

        # Computer the product of the two quaternions, term by term
        Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

        # Create a 4 element array containing the final quaternion
        final_quaternion = np.array([Q0Q1_w, Q0Q1_x, Q0Q1_y, Q0Q1_z])

        # Return a 4 element array containing the final quaternion (q02,q12,q22,q32)
        return final_quaternion

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
        '''
        print('time: ', t)
        print('current location: ', state["x"])
        print('desired location: ', flat_output["x"])
        print('desired velocity: ', flat_output["x_dot"])
        '''
        r_dott_des = flat_output["x_ddot"] - np.matmul(self.K_d, (state["v"]-flat_output["x_dot"])) - np.matmul(self.K_p, (state["x"] - flat_output["x"])) #desired acceleration
        #print('r dotdot des: ', r_dott_des)
        F_des = (self.mass*r_dott_des + np.array([0, 0, self.mass*self.g]))*self.quad_sign#desired force
        #print('F des: ', F_des)

        #calculate rotaiton matrix from input quaternion
        R = Rotation.from_quat(state["q"]).as_matrix()
        #print('R: ', R)

        #unit vector of desired force/b3
        [a, b, c] = self.quad_sign*F_des / np.linalg.norm(F_des)

        yaw = flat_output["yaw"]
        #print('c', c)

        q_yaw = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]) #w + xi + yj + zk
        q_abc = (1/np.sqrt(2*(1+c))) * np.array([1+c,-b,a,0]) #w + xi + yj + zk

        #calculate yaw/q_abc based on desired quat map
        if c <= 0 and self.quad_sign == 1:  # quad cross from upright to inverse
            q_abc_bar = 1/np.sqrt(2*(1-c)) * np.array([-b, 1-c, 0, a])
            q_abc_bar_inv = q_abc_bar * np.array([1, -1, -1, -1])
            q_yaw = self.quaternion_multiply(q_abc_bar_inv, self.quaternion_multiply(q_abc, q_yaw))
            q_abc = q_abc_bar
            #yaw = np.arctan2(a, b) + yaw
            self.quad_sign = -1
            print('flip to inverse!')
        elif c >= 0 and self.quad_sign == -1:  # quad cross from inverse to upright
            q_abc = 1 / np.sqrt(2 * (1 + c)) * np.array([1 + c, -b, a, 0])
            #yaw = np.arctan2(a, b) + yaw  ##TODO - double check math on this conversion
            #yaw kept as is
            self.quad_sign = 1
            print('flip to upright!')
        elif c <= 0 and self.quad_sign == -1:
            q_abc_bar = 1/np.sqrt(2*(1-c)) * np.array([-b, 1-c, 0, a])
            q_abc_bar_inv = q_abc_bar * np.array([1, -1, -1, -1])
            q_yaw = self.quaternion_multiply(q_abc_bar_inv, self.quaternion_multiply(q_abc, q_yaw))
            q_abc = q_abc_bar
            #yaw = np.arctan2(a, b) + yaw
            #print(3)
        elif c >= 0 and self.quad_sign == 1:
            q_abc = 1 / np.sqrt(2 * (1 + c)) * np.array([1+c, -b, a, 0])
            q_yaw = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]) #w + xi + yj + zk
            #print(4)
            # yaw is kept as is

        #q_yaw = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)]) #w + xi + yj + zk
        q_des = self.quaternion_multiply(q_abc, q_yaw) #w + xi + yj + zk
        q_des = np.hstack((q_des[1:4], q_des[0])) #xi + yj + zk + w
        R_des = Rotation.from_quat(q_des).as_matrix() #from quat takes xi + yj + zk + w

        e_R = 0.5 * self.vee_map(R_des.T@R - R.T@R_des)
        e_W = state["w"] - 0  # let w_des = 0 for now

        # print('error R: ', e_R)
        # print('state w', state['w'])
        # print('w des', w_des)
        # print('error W:', e_W)

        #desired inputs
        b3 = R @ np.array([[0], [0], [1]])
        u1 = np.matmul(b3.T, F_des)
        u2 = self.inertia @ (-np.matmul(self.K_r, e_R) - np.matmul(self.K_w, e_W))

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

        #print('K: ', K)

        cmd_motor_speeds = np.matmul(np.linalg.inv(K),u) #motor speeds^2

        #make sure motor speeds are physically feasible:
        for i in range(len(cmd_motor_speeds)):
            if cmd_motor_speeds[i] > self.rotor_speed_max**2:
                cmd_motor_speeds[i] = self.rotor_speed_max**2
            elif cmd_motor_speeds[i] < -1*self.rotor_speed_min**2:
                cmd_motor_speeds[i] = -1*self.rotor_speed_min**2

        motor_signs = np.clip(cmd_motor_speeds.astype(int), -1, 1)
        cmd_thrust = self.k_thrust*cmd_motor_speeds #motor thrusts
        cmd_moment = self.k_drag*cmd_motor_speeds #motor moments
        cmd_motor_speeds = np.sqrt(abs(cmd_motor_speeds))*motor_signs

        #print('cmd motor speeds: ', cmd_motor_speeds)
        #print('cmd thrust: ', cmd_thrust)
        #print('cmd moment: ', cmd_moment)

        cmd_q = q_des

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q,
                         'cmd_o':self.quad_sign,
                         'cmd_m':motor_signs}

        #exit()
        #if self.i == 400:
            #exit()
        #self.i = self.i +1
        return control_input
