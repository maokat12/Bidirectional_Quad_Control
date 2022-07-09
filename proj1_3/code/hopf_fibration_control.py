import numpy as np
from scipy.spatial.transform import Rotation as Rotation
import csv
from pyquaternion import Quaternion

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
        k_r = 3700

        k_p = 4.5
        k_d = 3.5
        k_w = 23
        k_r = 430

        self.K_p = np.array([[k_p, 0, 0],
                             [0, k_p, 0],
                             [0, 0, k_p]])

        self.K_d = np.array([[k_d, 0, 0],
                             [0, k_d, 0],
                             [0, 0, k_d]])

        self.K_w = np.array([[72, 0, 0],
                             [0, 72, 0],
                             [0, 0, 72]])

        self.K_r = np.array([[430, 0, 0],
                             [0, 430, 0],
                             [0, 0, 131]])
        self.i = 0

    def vee_map(self, R):
        c = R[1][0]
        b = R[0][2]
        a = R[2][1]
        return np.array([a, b, c])

    def get_q(self, abc, yaw, mode):
        [a, b, c] = abc
        yaw_n = np.radians(yaw)
        q_des = None
        #print('abc', abc)
        #print(mode)

        if mode == 1 and c > -0.95: #north chart, but ensure we don't get too close to the singularity
            q_abc = (1/np.sqrt(2*(1+c))) * np.array([1+c, -b, a, 0])
            q_yaw_n = np.array([np.cos(yaw_n/2), 0, 0, np.sin(yaw_n/2)])
            q_abc = Quaternion(np.array(q_abc))
            q_yaw_n = Quaternion(np.array(q_yaw_n))
            q_des = q_abc * q_yaw_n
        else: #c <= 0 #south chart
            yaw_s = 2*np.arctan2(a, b) + yaw_n
            try:
                q_abc_bar = (1/np.sqrt(2*(1-c))) * np.array([-b, 1-c, 0, a])
                q_yaw_s = np.array([np.cos(yaw_s/2), 0, 0, np.sin(yaw_s/2)])
                q_abc_bar = Quaternion(np.array(q_abc_bar))
                q_yaw_s = Quaternion(np.array(q_yaw_s))
                q_des = q_abc_bar * q_yaw_s
            except:
                print(mode)
                print(abc)

        return q_des

    def get_w(self, abc, yaw, F_des, r_dddot, quad_o, mode):
        sign_f = 1 if F_des[2] > 0 else -1
        F_des = np.array([[F_des[0]],
                          [F_des[1]],
                          [F_des[2]]])
        F_dot = self.mass*r_dddot
        a = abc[0]
        b = abc[1]
        c = abc[2]
        yaw_n = np.radians(yaw)

        if quad_o == 1:
            abc_dot = (F_des.T@F_des*np.identity(3) - F_des@F_des.T)/np.linalg.norm(F_des)**3 @ F_dot
        else:
            abc_dot = -1*(F_des.T@F_des*np.identity(3) - F_des@F_des.T)/np.linalg.norm(F_des)**3 @ F_dot

        [a_dot, b_dot, c_dot] = abc_dot

        if mode == 1 and c > -0.95: #north chart, but ensure we don't get too close to the singularity
            w1_n = np.sin(yaw_n)*a_dot - np.cos(yaw_n)*b_dot - (a*np.sin(yaw_n) - b*np.cos(yaw_n)) * c_dot/(1+c)
            w2_n = np.cos(yaw_n)*a_dot + np.sin(yaw_n)*b_dot - (a*np.cos(yaw_n) + b*np.sin(yaw_n)) * c_dot/(1+c)
            w3_n = 0

            if w1_n > 100:
                print('w1_n', w1_n)
                print('w2_n', w2_n)
                print('abc', abc)
                print('F_des', F_des)
                print('F_dot', F_dot)
                print('abc dot', abc_dot)

            return [w1_n, w2_n, w3_n]
        else: #mode <= 0: #south chart
            yaw_s = 2*np.arctan2(a, b) + yaw_n
            w1_s = np.sin(yaw_s)*a_dot + np.cos(yaw_s)*b_dot - (a*np.sin(yaw_s) + b*np.cos(yaw_s)) * c_dot/(c-1)
            w2_s = np.cos(yaw_s)*a_dot - np.sin(yaw_s)*b_dot - (a*np.cos(yaw_s) - b*np.sin(yaw_s)) * c_dot/(c-1)
            w3_s = 0

            if w1_s > 100:
                print('w1_n', w1_s)
                print('w2_n', w2_s)
                print('abc', abc)
                print('F_des', F_des)
                print('F_dot', F_dot)
                print('abc dot', abc_dot)

            return [w1_s, w2_s, w3_s]

    def get_abc_yaw(self, q): #for debugging only
        q = Rotation.from_quat(q)
        abc = q.apply(np.array([0, 0, 1]))
        a = abc[0]
        b = abc[1]
        c = abc[2]

        q_abc = 1/np.sqrt(2*(1-c)) * np.array([-b, 1-c, 0, a])
        q_abc = np.hstack((q_abc[1:4], q_abc[0]))
        q_abc = Rotation.from_quat(q_abc)
        q_abc_inv = q_abc.inv()
        #q_abc_inv = q_abc * np.array([1, -1, -1, -1])
        #q_abc_inv = np.hstack((q_abc_inv[1:4], q_abc_inv[0])) #xi + yj + zk + w
        #q_abc_inv = Rotation.from_quat(q_abc_inv)

        q_yaw = q_abc_inv * q
        q_yaw = q_yaw.as_quat()
        yaw = np.arctan2(q_yaw[2], q_yaw[3])*2

        print('get abc yaw function')
        print('q_abc', q_abc.as_quat())
        print('q_abc_inv', q_abc_inv.as_quat())
        print('identity', (q_abc*q_abc_inv).as_quat())
        print('abc', abc)
        print('q_yaw', q_yaw)
        print('yaw', yaw)

    def get_des_traj(self, acc, mode, quad_o, yaw, jerk):
        F_des = (self.mass*acc + np.array([0, 0, self.mass*self.g]))#desired force
        F_mag = np.linalg.norm(F_des)
        sign_f = 1 if F_des[2] > 0 else -1

        if quad_o == 1:
            [a, b, c] = F_des / np.linalg.norm(F_des)
        else:
            [a, b, c] = -1 * F_des / np.linalg.norm(F_des)
        if c > 0:
            mode = 1
        else:
            mode = -1

        w_des = self.get_w([a, b, c], yaw, F_des, jerk, quad_o, mode)

        return mode, sign_f, quad_o, a, b, c, F_des[0], F_des[1], F_des[2], w_des[0], w_des[1], w_des[2], F_mag

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
        print('desired velocity: ', flat_output["x_dot"])j
        '''

        #self.K_d = np.identity(3)*0
        #self.K_p = np.identity(3)*0
        quad_o = flat_output["quad_o"] #b3 parallel/antiparallel to F
        mode = -1 #North/South Chart - constantly stay in the north chart in 1st pass

        r_dott_des = flat_output["x_ddot"] - self.K_d@(state["v"]-flat_output["x_dot"]) - self.K_p@(state["x"] - flat_output["x"]) #desired acceleration
        F_des = (self.mass*r_dott_des + np.array([0, 0, self.mass*self.g]))#desired force
        F_mag = np.linalg.norm(F_des)
        a_mag = flat_output["x_ddot"][2]+9.81
        gain_mag = self.K_d@(state["v"]-flat_output["x_dot"]) + self.K_p@(state["x"]-flat_output["x"])
        gain_mag = gain_mag[2]

        #calculate rotation matrix from input quaternion
        R = Rotation.from_quat(state["q"]).as_matrix()
        abc_state = R @ np.array([[0], [0], [1]])
        #print('abc_state', abc_state)

        sign_f = 1 if F_des[2] > 0 else -1
        if quad_o == 1:
            [a, b, c] = F_des / np.linalg.norm(F_des)
        else:
            [a, b, c] = -1 * F_des / np.linalg.norm(F_des)

        if c > 0:
            mode = 1
        else:
            mode = -1

        # little hack I added to override [a, b, c] when trajectory approaches the singularity to avoid gain errors
        if F_mag < 0.01: #N
            F_des = (self.mass*flat_output["x_ddot"] + np.array([0, 0, self.mass*self.g]))#desired force
            F_mag = np.linalg.norm(F_des)
            sign_f = 1 if F_des[2] > 0 else -1

            if quad_o == 1:
                [a, b, c] = F_des / np.linalg.norm(F_des)
            else:
                [a, b, c] = -1 * F_des / np.linalg.norm(F_des)
            if c > 0:
                mode = 1
            else:
                mode = -1


        yaw = flat_output["yaw"]


        ideal_traj = self.get_des_traj(flat_output["x_ddot"], mode, quad_o, yaw, flat_output['x_dddot'])


        q_des = self.get_q([a, b, c], yaw, mode) #yaw in degree
        R_des = q_des.rotation_matrix

        w_des = self.get_w([a, b, c], yaw, F_des, flat_output['x_dddot'], quad_o, mode)

        e_R = 0.5 * self.vee_map(R_des.T@R - R.T@R_des)
        e_W = state["w"] - w_des

        #debugging outputs for MATLAB
        #write R_des * e3 and R * e3 to csv file
        #with open("check_gains.csv", 'a+', newline = '') as csvfile:
            #csvwriter = csv.writer(csvfile)
            #b3_des = Rotation.from_matrix(R_des).apply(np.array([0, 0, 1]))
            #b3 = Rotation.from_matrix(R).apply(np.array([0, 0, 1]))
            #x = np.hstack([r_dott_des, flat_output["x_dddot"]])
            #x = np.vstack((x, np.zeros((2, 9))))
            #x = np.hstack([x, R_des, R])
            #x = np.hstack([t, r_dott_des[0], r_dott_des[1], r_dott_des[2], F_des[0], F_des[1], F_des[2], a, b, c, quad_o, w_des[0], w_des[1], w_des[2]])
            #for i in x:
            #    csvwriter.writerow(i)
            #csvwriter.writerow(x)

        #desired inputs
        b3 = R @ np.array([[0], [0], [1]])
        u1 = np.matmul(b3.T, F_des)
        u2 = self.inertia @ (-np.matmul(self.K_r, e_R) - np.matmul(self.K_w, e_W))

        #calculate corresponding forces
        u = np.array([u1[0], u2[0], u2[1], u2[2]]) #input matrix

        #follows the mellinger paper
        K = np.array([[self.k_thrust, self.k_thrust, self.k_thrust, self.k_thrust],
                      [0, self.k_thrust*self.arm_length, 0, -self.k_thrust*self.arm_length],
                      [-self.k_thrust*self.arm_length, 0, self.k_thrust*self.arm_length, 0],
                      [self.k_drag, -self.k_drag, self.k_drag, -self.k_drag]])

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

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':q_des.elements,
                         'cmd_o':flat_output['quad_o'],
                         'cmd_m':motor_signs,
                         'mode':mode,
                         'F_des':F_des,
                         'F_mag':F_mag,
                         'abc':[a, b, c],
                         'sign_f':sign_f,
                         'r_des':r_dott_des,
                         'w_des':w_des,
                         'acc_des':r_dott_des,
                         'ideal_traj':ideal_traj,
                         'acc_plan':flat_output["x_ddot"],
                         'a_mag':a_mag,
                         'gain_mag':gain_mag}

        return control_input
