import numpy as np
from scipy.spatial.transform import Rotation as Rotation

class SE3Control(object):
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

        k_p =5 #5
        k_d = 3.5#3.5  # bro what the fuck
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
        F_des = self.mass*r_dott_des + np.array([0, 0, self.mass*self.g]) #desired force
        #print('F des: ', F_des)

        R = Rotation.from_quat(state["q"]).as_matrix()

        #print('R: ', R)

        a_yaw = np.array([np.cos(flat_output["yaw"]),
                          np.sin(flat_output["yaw"]),
                          0])

        #print('a yaw: ', a_yaw)

        #[a, b, c] = F_des / np.linalg.norm(F_des)
        #print('c', c)

        b3_des = F_des / np.linalg.norm(F_des)
        b2_des = np.cross(b3_des, a_yaw) / np.linalg.norm(np.cross(b3_des, a_yaw))
        b1_des = np.cross(b2_des, b3_des)

        #print('b3 des: ', b3_des)
        #print('b2 des: ', b2_des)
        #print('b1 des: ', b1_des)

        R_des = np.array([b1_des, b2_des, b3_des]).T


        e_R = 0.5*self.vee_map(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_W = state["w"] - 0 #setting desired angular velocity to 0

        #print('error R: ', e_R)
        #print('error W:', e_W)

        #desired inputs
        b3 = R@np.array([[0],[0],[1]])
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
            elif cmd_motor_speeds[i] < self.rotor_speed_min:
                cmd_motor_speeds[i] = self.rotor_speed_min
        cmd_thrust = self.k_thrust*cmd_motor_speeds #motor thrusts
        cmd_moment = self.k_drag*cmd_motor_speeds #motor moments
        cmd_motor_speeds = np.sqrt(cmd_motor_speeds)

        #print('cmd motor speeds: ', cmd_motor_speeds)
        #print('cmd thrust: ', cmd_thrust)
        #print('cmd moment: ', cmd_moment)

        cmd_q = Rotation.from_matrix(R_des).as_quat()

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q,
                         'cmd_o': 1}

        #exit()
        #if self.i == 400:
            #exit()
        #self.i = self.i +1
        return control_input
