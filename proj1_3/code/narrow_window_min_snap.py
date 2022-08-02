import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

class MinSnapNW(object):
    def __init__(self, points, dist_vel, vel_max, dist_threshold):
        # dist_vel (m/s) - velocity used to calculate time segment times
        # vel_max (m/s) - max/min velocity quad can travel
        # dist_threshold (m) - corridor radius for quad to stray

        self.distances = []
        self.time_segments = []
        self.time_cumulative = []
        self.num_segments = None
        self.dist_vel = dist_vel
        self.vel_max = vel_max
        self.dist_threshold = dist_threshold
        self.plt = False #plot min snap before sent to controller

        #set constraints
        self.points = points
        self.start = self.points[0]
        self.acc_cons = None
        self.no_acc_cons = True
        self.o_cons = None
        self.no_o_cons = True
        self.narrow_window_waypoint = None
        self.no_narrow_window = True

    def get_distance(self, p1, p2):  # return distance between two points
        p1 = np.array(p1)
        p2 = np.array(p2)
        dist = abs(np.linalg.norm(p2 - p1))
        return dist

    def set_acc_cons(self, acc_cons):
        self.acc_cons = np.append(self.acc_cons, acc_cons)
        self.no_acc_cons = False

    def set_o_cons(self, o_cons):
        self.o_cons = o_cons
        self.no_o_cons = False

        #check if any switches happen
        if not np.all(o_cons == o_cons[0]):
            acc_cons = np.array(np.zeros(4))
            for i in range(1, len(o_cons)):
                if o_cons[i-1] != o_cons[i]:
                    acc_cons = np.vstack((acc_cons, [0, 0, -9.81, i]))
            acc_cons = np.delete(acc_cons, 0, 0)
            self.acc_cons = acc_cons
            self.no_acc_cons = False
            print(self.acc_cons)

    def set_narrow_window_waypoint(self, waypoint):
        self.narrow_window_waypoint = waypoint #integer
        self.no_narrow_window = False
        print("narrow window point set")

    def get_Hessian(self):
        H = None
        for i in range(self.num_segments):
            T = self.time_segments[i]
            H_sub = 2*96*np.array([[1050*T**7,  525*T**6,   210*T**5,   52.5*T**4],
                                  [525*T**6,  270*T**5,   112.5*T**4, 30*T**3],
                                  [210*T**5,  112.5*T**4, 50*T**3,    15*T**2],
                                  [52.5*T**4, 30*T**3,    15*T**2,    6*T]])
            H_sub = np.block([[H_sub, np.zeros((4, 4))],
                              [np.zeros((4, 8))]])
            if i == 0: #start of construction
                H = H_sub
            else:
                H = np.block([[H, np.zeros((8*i, 8))],
                              [np.zeros((8, 8*i)), H_sub]])
        return H

    def get_Aeq(self):

        A_eq_start = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 2, 0, 0],
                               [0, 0, 0, 0, 6, 0, 0, 0]])
        tf = self.time_segments[-1]
        A_eq_end = np.array([[tf**7, tf**6, tf**5, tf**4, tf**3, tf**2, tf**1, 1],
                             [7*(tf**6), 6*(tf**5), 5*(tf**4), 4*(tf**3), 3*(tf**2), 2*tf, 1, 0],
                             [42*(tf**5), 30*(tf**4), 20*(tf**3), 12*(tf**2), 6*tf, 2, 0, 0],
                             [210*(tf**4), 120*(tf**3), 60*(tf**2), 24*tf, 6, 0, 0, 0]])
        A_eq = np.block([[A_eq_start, np.zeros((4, 8*(self.num_segments-1)))],
                         [np.zeros((4, 8*(self.num_segments-1))), A_eq_end]])

        beq_x = np.array([self.points[0][0], 0, 0, 0, self.points[-1][0], 0, 0, 0])
        beq_y = np.array([self.points[0][1], 0, 0, 0, self.points[-1][1], 0, 0, 0])
        beq_z = np.array([self.points[0][2], 0, 0, 0, self.points[-1][2], 0, 0, 0])

        if self.num_segments == 1:
            return A_eq, A_eq, A_eq, beq_x, beq_y, beq_z
        else: #intermediary constraints for 1+ segments
            #waypoints constraints
            for i in range(self.num_segments-1):
                T = self.time_segments[i]
                A_eq_sub = np.array([T**7, T**6, T**5, T**4, T**3, T**2, T, 1])
                RHS = np.array([0, 0, 0, 0, 0, 0, 0, 1])
                A_eq_sub = np.block([[np.zeros((1, 8*i)), A_eq_sub, np.zeros((1, 8*(self.num_segments-(i+1))))],
                                    [np.zeros((1, 8*(i+1))), RHS, np.zeros((1, max(0, 8*(self.num_segments-(i+2)))))]])
                A_eq = np.vstack((A_eq,
                                  A_eq_sub))
                beq_x = np.hstack((beq_x, self.points[i+1][0], self.points[i+1][0]))
                beq_y = np.hstack((beq_y, self.points[i+1][1], self.points[i+1][1]))
                beq_z = np.hstack((beq_z, self.points[i+1][2], self.points[i+1][2]))
            #no acceleration constraints
            if self.no_acc_cons:
                #continuity constraints
                for i in range(self.num_segments-1):
                    T = self.time_segments[i]
                    A_eq_sub_LHS = np.block([[7*T**6, 6*T**5, 5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
                                             [42*T**5, 30*T**4, 20*T**3, 12*T**2, 6*T, 2, 0, 0],
                                             [210*T**4, 120*T**3, 60*T**2, 24*T, 6, 0, 0, 0]])
                    A_eq_sub_RHS = np.block([[0, 0, 0, 0, 0, 0, -1, 0],
                                             [0, 0, 0, 0, 0, -2, 0, 0],
                                             [0, 0, 0, 0, -6, 0, 0, 0]])
                    A_eq_sub = np.block([np.zeros((3,8*i)), A_eq_sub_LHS, A_eq_sub_RHS, np.zeros((3, 8*(self.num_segments-(i+2))))])
                    A_eq = np.vstack((A_eq,
                                      A_eq_sub))
                    beq_x = np.hstack((beq_x, np.zeros(3)))
                    beq_y = np.hstack((beq_y, np.zeros(3)))
                    beq_z = np.hstack((beq_z, np.zeros(3)))
                return A_eq, A_eq, A_eq, beq_x, beq_y, beq_z
            #acceleration constraint
            else:
                A_eq_x = A_eq
                A_eq_y = A_eq
                A_eq_z = A_eq
                for i in range(self.num_segments-1):
                    T = self.time_segments[i]
                    A_eq_sub_LHS = np.block([[7*T**6, 6*T**5, 5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0], #velocity
                                             [210*T**4, 120*T**3, 60*T**2, 24*T, 6, 0, 0, 0]]) #jerk
                    A_eq_sub_RHS = np.block([[0, 0, 0, 0, 0, 0, -1, 0],
                                             [0, 0, 0, 0, -6, 0, 0, 0]])
                    A_eq_sub = np.block([np.zeros((2,8*i)), A_eq_sub_LHS, A_eq_sub_RHS, np.zeros((2, 8*(self.num_segments-(i+2))))])

                    A_eq_x = np.vstack((A_eq_x,A_eq_sub))
                    A_eq_y = np.vstack((A_eq_y,A_eq_sub))
                    A_eq_z = np.vstack((A_eq_z,A_eq_sub))
                    beq_x = np.hstack((beq_x, np.zeros(2)))
                    beq_y = np.hstack((beq_y, np.zeros(2)))
                    beq_z = np.hstack((beq_z, np.zeros(2)))

                    #check if there's an acceleration constraint for this waypoint
                    #you're constraining jerk right now dumbass
                    acc_con_index = np.where(self.acc_cons[:, 3] == (i+1))
                    if len(acc_con_index[0]) == 1: #acc constraint at this waypoint found
                        index = acc_con_index[0][0] #get index value
                        for j in range(3): #cycle through x(0),y(1),z(2)
                            #no specified acceleration
                            if self.acc_cons[index][j] == -1:
                                #add normal continuity constraint
                                A_eq_sub_LHS = np.block([[42*T**5, 30*T**4, 20*T**3, 12*T**2, 6*T, 2, 0, 0]])
                                A_eq_sub_RHS = np.block([[0, 0, 0, 0, 0, -2, 0, 0]])
                                A_eq_sub = np.block([np.zeros((1,8*i)), A_eq_sub_LHS, A_eq_sub_RHS, np.zeros((1, 8*(self.num_segments-(i+2))))])
                                if j == 0: #x
                                    A_eq_x = np.vstack((A_eq_x, A_eq_sub))
                                    beq_x = np.hstack((beq_x, np.zeros(1)))
                                elif j == 1: #y
                                    A_eq_y = np.vstack((A_eq_y, A_eq_sub))
                                    beq_y = np.hstack((beq_y, np.zeros(1)))
                                else: #z
                                    A_eq_z = np.vstack((A_eq_z,A_eq_sub))
                                    beq_z = np.hstack((beq_z, np.zeros(1)))
                            else:
                                print('j', j)
                                print(self.acc_cons[index][j])
                                A_eq_sub_LHS = np.array([42*T**5, 30*T**4, 20*T**3, 12*T**2, 6*T, 2, 0, 0])
                                A_eq_sub_RHS = np.array([0, 0, 0, 0, 0, -2, 0, 0])
                                A_eq_sub = np.block([[np.zeros((1, 8*i)), A_eq_sub_LHS, np.zeros((1, 8*(self.num_segments-(i+1))))],
                                                     [np.zeros((1, 8*(i+1))), A_eq_sub_RHS, np.zeros((1, 8*(self.num_segments-(i+2))))]])
                                if j == 0: #x
                                    A_eq_x = np.vstack((A_eq_x,A_eq_sub))
                                    beq_x = np.hstack((beq_x, self.acc_cons[index][0], -self.acc_cons[index][0]))
                                elif j == 1: #y
                                    A_eq_y = np.vstack((A_eq_y,A_eq_sub))
                                    beq_y = np.hstack((beq_y, self.acc_cons[index][1], -self.acc_cons[index][1]))
                                else: #z
                                    A_eq_z = np.vstack((A_eq_z,A_eq_sub))
                                    beq_z = np.hstack((beq_z, self.acc_cons[index][2], -self.acc_cons[index][2]))
                    else: #no acc constraint for this waypoint
                    #add normal continuity constraint
                        A_eq_sub_LHS = np.block([[42*T**5, 30*T**4, 20*T**3, 12*T**2, 6*T, 2, 0, 0]])
                        A_eq_sub_RHS = np.block([[0, 0, 0, 0, 0, -2, 0, 0]])
                        A_eq_sub = np.block([np.zeros((1,8*i)), A_eq_sub_LHS, A_eq_sub_RHS, np.zeros((1, 8*(self.num_segments-(i+2))))])

                        A_eq_x = np.vstack((A_eq_x, A_eq_sub))
                        beq_x = np.hstack((beq_x, np.zeros(1)))
                        A_eq_y = np.vstack((A_eq_y, A_eq_sub))
                        beq_y = np.hstack((beq_y, np.zeros(1)))
                        A_eq_z = np.vstack((A_eq_z,A_eq_sub))
                        beq_z = np.hstack((beq_z, np.zeros(1)))

                return A_eq_x, A_eq_y, A_eq_z, beq_x, beq_y, beq_z

    def get_Aineq(self, vel_max, dist_threshold):
        #inequality constraints - position
        i = 10 #number of intervals
        m = dist_threshold #distance deviance threshold

        A = np.array([])
        b_x = np.array([])
        b_y = np.array([])
        b_z = np.array([])
        for k in range(self.num_segments):
            T = self.time_segments[k]
            start = self.points[k]
            stop = self.points[k+1]

            #get slopes
            m_x = (stop[0] - start[0])/T
            m_y = (stop[1] - start[1])/T
            m_z = (stop[2] - start[2])/T

            #print('start', start)
            #print('stop', stop)
            #print('Time segment', T)
            #print('m_x', m_x)

            for j in range(i+1):
                A_sub = np.block([[(T*j/i)**7, (T*j/i)**6, (T*j/i)**5, (T*j/i)**4, (T*j/i)**3, (T*j/i)**2, (T*j/i), 1],
                                  [-(T*j/i)**7, -(T*j/i)**6, -(T*j/i)**5, -(T*j/i)**4, -(T*j/i)**3, -(T*j/i)**2, -(T*j/i), -1]])
                if j == 0 and k == 0:
                    A = np.block([A_sub, np.zeros((2, 8*(self.num_segments-(k+1))))])
                else:
                    A = np.block([[A],
                                  [np.zeros((2, 8*k)), A_sub, np.zeros((2, 8*(self.num_segments-(k+1))))]])

                #print('time', T*j/i)
                curr_pos = [start[0] + T*j/i*m_x, start[1] + T*j/i*m_y, start[2] + T*j/i*m_z]

                b_x = np.hstack((b_x, curr_pos[0]+m, -1*curr_pos[0]+m))
                b_y = np.hstack((b_y, curr_pos[1]+m, -1*curr_pos[1]+m))
                b_z = np.hstack((b_z, curr_pos[2]+m, -1*curr_pos[2]+m))
        #exit()

        #inequality constraints - velocity
        for k in range(self.num_segments):
            T = self.time_segments[k]
            for j in range(i+1):
                A_sub = np.block([[7*(T*j/i)**6, 6*(T*j/i)**5, 5*(T*j/i)**4, 4*(T*j/i)**3, 3*(T*j/i)**2, 2*T*j/i, 1, 0],
                                  [-7*(T*j/i)**6, -6*(T*j/i)**5, -5*(T*j/i)**4, -4*(T*j/i)**3, -3*(T*j/i)**2, -2*(T*j/i), -1, 0]])
                A = np.block([[A],
                              [np.zeros((2, 8*k)), A_sub, np.zeros((2, 8*(self.num_segments-(k+1))))]])
                b_x = np.hstack((b_x, vel_max, vel_max))
                b_y = np.hstack((b_y, vel_max, vel_max))
                b_z = np.hstack((b_z, vel_max, vel_max))

        return A, b_x, b_y, b_z

    def get_z_ineq_narrow_window(self, A_ineq_z, b_z):
        #assume only one window
        index = self.narrow_window_waypoint
        T_left = self.time_segments[index-1] #segment before waypoint
        T_right = self.time_segments[index] #segment after waypoint
        i = 5 #number of intervals

        for j in range(i+1):
            T_L = (0.5*T_left*(1+j/i))
            T_R = (0.5*T_right*(1-j/i))
            A_sub_L = np.array([-42*T_L**5, -30*T_L**4, -20*T_L**3, -12*T_L**2, -6*T_L, -2, 0, 0])
            A_sub_R = np.array([-42*T_R**5, -30*T_R**4, -20*T_R**3, -12*T_R**2, -6*T_R, -2, 0, 0])

            A_ineq_z = np.block([[A_ineq_z],
                                [np.zeros(8*(index-2)), A_sub_L, np.zeros(8*(self.num_segments-(index-1)))],
                                [np.zeros(8*(index-1)), A_sub_R, np.zeros(8*(self.num_segments-index))]])
            b_z = np.hstack((b_z, 9.81, 9.81))

        print(A_ineq_z.shape)
        print(b_z.shape)

        return A_ineq_z, b_z

    def get_y_eq_narrow_window(self, A_eq_y, beq_y):
        print('test')
        #assume only one window
        index = self.narrow_window_waypoint
        T_left = self.time_segments[index-1] #segment before waypoint
        T_right = self.time_segments[index] #segment after waypoint
        i = 10 #number of intervals

        for j in range(i+1):
            T_L = (0.25*T_left*(3+j/i))
            T_R = (0.25*T_right*(1-j/i))
            print('T_L', T_L)
            print('T_R', T_R)
            A_sub_L = np.array([42*T_L**5, 30*T_L**4, 20*T_L**3, 12*T_L**2, 6*T_L, 2, 0, 0])
            A_sub_R = np.array([42*T_R**5, 30*T_R**4, 20*T_R**3, 12*T_R**2, 6*T_R, 2, 0, 0])

            A_eq_y = np.block([[A_eq_y],
                               [np.zeros(8*(index-2)), A_sub_L, np.zeros(8*(self.num_segments-(index-1)))],
                               [np.zeros(8*(index-1)), A_sub_R, np.zeros(8*(self.num_segments-index))]])
            beq_y = np.hstack((beq_y, 0, 0))

        print(A_eq_y.shape)
        print(beq_y.shape)

        return A_eq_y, beq_y

    def get_x_ineq_narrow_window(self, A_ineq_x, b_x, c_z):
        theta = np.radians(5) #deg
        index = self.narrow_window_waypoint
        T_left = self.time_segments[index-1] #segment before waypoint
        T_right = self.time_segments[index] #segment after waypoint
        i = 2 #number of intervals

        #get z acceleration values
        c_z_L = c_z[0+8*(index-2):8+8*(index-2)]
        c_z_R = c_z[0+8*(index-1):8+8*(index-1)]
        acc_z_L = [0, 0, 42*c_z_L[0], 30*c_z_L[1], 20*c_z_L[2], 12*c_z_L[3], 6*c_z_L[4], 2*c_z_L[5]]
        acc_z_R = [0, 0, 42*c_z_R[0], 30*c_z_R[1], 20*c_z_R[2], 12*c_z_R[3], 6*c_z_R[4], 2*c_z_R[5]]

        for j in range(i+1):
            T_L = (0.5*T_left*(1+j/i))
            T_R = (0.5*T_right*(1-j/i))
            A_sub_L = np.array([42*T_L**5, 30*T_L**4, 20*T_L**3, 12*T_L**2, 6*T_L, 2, 0, 0])
            A_sub_R = np.array([42*T_R**5, 30*T_R**4, 20*T_R**3, 12*T_R**2, 6*T_R, 2, 0, 0])
            A_ineq_x = np.block([[A_ineq_x],
                                [np.zeros(8*(index-2)), A_sub_L, np.zeros(8*(self.num_segments-(index-1)))],
                                [np.zeros(8*(index-1)), A_sub_R, np.zeros(8*(self.num_segments-index))],
                                [np.zeros(8*(index-2)), -A_sub_L, np.zeros(8*(self.num_segments-(index-1)))],
                                [np.zeros(8*(index-1)), A_sub_R, np.zeros(8*(self.num_segments-index))]])
            z_L = np.polyval(np.array(acc_z_L), T_L)
            z_R = np.polyval(np.array(acc_z_R), T_R)
            b_x = np.hstack((b_x, (z_L+9.81)*np.tan(theta), -(z_R+9.81)*np.tan(theta), 0, 0))

            #print('time left', T_L)
            #print('LHSz', z_L)
            #print('LHS', (z_L+9.81)*np.tan(theta))
            #print('time right', T_R)
            #print('RHS z', z_R)
            #print('RHS', -(z_R+9.81)*np.tan(theta))
            print(b_x)
            print("\n")

        return A_ineq_x, b_x

    def get_coeffs(self, Aeq, beq, A, b, H):
        def loss(w, sign=1.):
            return sign * (0.5 * np.dot(w.T, np.dot(H, w)))

        def jac(w, sign=1.):
            return sign * (np.dot(w.T, H))

        eq_cons = {'type': 'eq',
             'fun':lambda x: beq - np.dot(Aeq, x),
             'jac':lambda x: -Aeq}
        ineq_cons = {'type': 'ineq',
                       'fun': lambda x: b - np.dot(A, x),
                       'jac':lambda x: -A}
        #cons = [eq_cons, ineq_cons]
        x0 = np.random.randn(len(H)) #declares variables for the QP sovler
        res_cons = optimize.minimize(loss, x0, jac=jac,constraints=[eq_cons, ineq_cons], method='SLSQP', options={'disp':False})
        return res_cons['x']

    def get_trajectory(self):
        #get distances & corresponding time segments
        for i in range(len(self.points)-1):
            dist = self.get_distance(self.points[i+1], self.points[i])
            self.distances.append(dist)
        self.distances = np.array(self.distances)
        self.time_segments = self.distances/self.dist_vel
        #self.time_segments = np.ones(len(self.distances)) * 5

        #add a little extra time to start and stop
        self.time_segments[0] = self.time_segments[0]+0.2
        self.time_segments[-1] = self.time_segments[-1]+0.2
        self.num_segments = len(self.time_segments)

        #trim short time segments just a little
        for i in range(self.num_segments):
            if self.time_segments[i] < 0.4:
                self.time_segments[i] = self.time_segments[i] + 0.2
            elif self.time_segments[i] > 0.8:
                self.time_segments[i] = self.time_segments[i] - 0.25

        #convert time segments to cumulative time
        self.time_cumulative = [sum(self.time_segments[0:i+1]) for i in range(self.num_segments)]
        print(self.time_segments)
        print(self.time_cumulative)

        #get hessian
        H = self.get_Hessian()

        #get A_eq
        Aeq_x, Aeq_y, Aeq_z, beq_x, beq_y, beq_z = self.get_Aeq()

        #get A_ineq
        A, b_x, b_y, b_z = self.get_Aineq(self.vel_max, self.dist_threshold)
        A_ineq_x = A
        A_ineq_y = A
        A_ineq_z = A

        #z narrow window inequality constraints
        if not self.no_narrow_window:
            A_ineq_z, b_z = self.get_z_ineq_narrow_window(A_ineq_z, b_z)
            #Aeq_y, beq_y = self.get_y_eq_narrow_window(Aeq_y, beq_y)

            y_coeff = self.get_coeffs(Aeq_y, beq_y, A_ineq_y, b_y, H)
            z_coeff = self.get_coeffs(Aeq_z, beq_z, A_ineq_z, b_z, H)

            print(z_coeff)

            #x narrow window inequality constraints
            A_ineq_x, b_x = self.get_x_ineq_narrow_window(A_ineq_x, b_x, z_coeff)

            #print(A_ineq_x.shape)
            print(b_x)

            #exit()

            #get x coeffs
            x_coeff = self.get_coeffs(Aeq_x, beq_x, A_ineq_x, b_x, H)
        else:
            x_coeff = self.get_coeffs(Aeq_x, beq_x, A, b_x, H)
            y_coeff = self.get_coeffs(Aeq_y, beq_y, A, b_y, H)
            z_coeff = self.get_coeffs(Aeq_z, beq_z, A, b_z, H)

        #set orientation constraints
        o_cons = np.ones(self.num_segments) #1 - upright, -1 - inverted
        if not self.no_o_cons:
            o_cons = self.o_cons

        traj_struct = (self.time_cumulative, x_coeff, y_coeff, z_coeff, o_cons)

        #plot generated trajectory
        if self.plt:
            self.plot(x_coeff, y_coeff, z_coeff)
            #self.plot_circle(x_coeff, y_coeff, z_coeff)
            #exit()

        #return x_coeff, y_coeff, z_coeff
        return traj_struct, self.num_segments, self.time_segments

    def plot(self, x_coeff, y_coeff, z_coeff):
        cont_time = 0
        plt.figure(5)
        ax = plt.axes(projection='3d')
        for i in range(self.num_segments):
            T = self.time_segments[i]
            time_math = np.linspace(0, T, 50)
            time_plot = np.linspace(cont_time, cont_time+T, 50)

            #pos
            x = x_coeff[0+8*i:8+8*i]
            y = y_coeff[0+8*i:8+8*i]
            z = z_coeff[0+8*i:8+8*i]
            #velocity
            dx = [0, 7*x[0], 6*x[1], 5*x[2], 4*x[3], 3*x[4], 2*x[5], x[6]]
            dy = [0, 7*y[0], 6*y[1], 5*y[2], 4*y[3], 3*y[4], 2*y[5], y[6]]
            dz = [0, 7*z[0], 6*z[1], 5*z[2], 4*z[3], 3*z[4], 2*z[5], z[6]]
            #acceleration
            ddx = [0, 0, 42*x[0], 30*x[1], 20*x[2], 12*x[3], 6*x[4], 2*x[5]]
            ddy = [0, 0, 42*y[0], 30*y[1], 20*y[2], 12*y[3], 6*y[4], 2*y[5]]
            ddz = [0, 0, 42*z[0], 30*z[1], 20*z[2], 12*z[3], 6*z[4], 2*z[5]]
            #jerk
            dddx = [0, 0, 0, 210*x[0], 120*x[1], 60*x[2], 24*x[3], 6*x[4]]
            dddy = [0, 0, 0, 210*y[0], 120*y[1], 60*y[2], 24*y[3], 6*y[4]]
            dddz = [0, 0, 0, 210*z[0], 120*z[1], 60*z[2], 24*z[3], 6*z[4]]

            x_pos = np.polyval(np.array(x), time_math)
            x_vel = np.polyval(np.array(dx), time_math)
            x_acc = np.polyval(np.array(ddx), time_math)
            x_jerk = np.polyval(np.array(dddx), time_math)

            y_pos = np.polyval(np.array(y), time_math)
            y_vel = np.polyval(np.array(dy), time_math)
            y_acc = np.polyval(np.array(ddy), time_math)
            y_jerk = np.polyval(np.array(dddy), time_math)

            z_pos = np.polyval(np.array(z), time_math)
            z_vel = np.polyval(np.array(dz), time_math)
            z_acc = np.polyval(np.array(ddz), time_math)
            z_jerk = np.polyval(np.array(dddz), time_math)

            plt.figure(1)
            plt.plot(time_plot, x_pos, 'r', label = 'x')
            plt.plot(time_plot, y_pos, 'g', label = 'y')
            plt.plot(time_plot, z_pos, 'b', label = 'z')
            plt.title('position')
            plt.figure(2)
            plt.plot(time_plot, x_vel, 'r')
            plt.plot(time_plot, y_vel, 'g')
            plt.plot(time_plot, z_vel, 'b')
            plt.title('velocity')
            plt.figure(3)
            plt.plot(time_plot, x_acc, 'r')
            plt.plot(time_plot, y_acc, 'g')
            plt.plot(time_plot, z_acc, 'b')
            plt.title('acceleration')
            plt.figure(4)
            plt.plot(time_plot, x_jerk, 'r')
            plt.plot(time_plot, y_jerk, 'g')
            plt.plot(time_plot, z_jerk, 'b')
            plt.title('jerk')
            plt.figure(5)
            ax.plot3D(x_pos, y_pos, z_pos, 'gray')
            plt.title('trajectory')

            cont_time = cont_time + T
        plt.show()
        #exit()

