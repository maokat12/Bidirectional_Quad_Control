import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

class MinJerk(object):
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
        self.plt = False #plot min jerk before sent to controller

        #set constraints
        self.points = points
        self.start = self.points[0]
        self.acc_cons = None
        self.no_acc_cons = True
        self.o_cons = None
        self.no_o_cons = True

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
                    acc_cons = np.vstack((acc_cons, [-1, 0, -9.81, i]))
            acc_cons = np.delete(acc_cons, 0, 0)
            self.acc_cons = acc_cons
            self.no_acc_cons = False
            print(self.acc_cons)

    def get_Hessian(self):
        H = None
        for i in range(self.num_segments):
            T = self.time_segments[i]
            H_sub = 2*96*np.array([[720*T**5,  360*T**4,  120*T**3],
                                   [360*T**4,  192*T**3,  72*T**2],
                                   [120*T**3,  72*T**2,   36*T]])
            H_sub = np.block([[H_sub, np.zeros((3, 3))],
                              [np.zeros((3, 6))]])
            if i == 0: #start of construction
                H = H_sub
            else:
                H = np.block([[H, np.zeros((6*i, 6))],
                              [np.zeros((6, 6*i)), H_sub]])
        return H

    def get_Aeq(self):

        A_eq_start = np.array([[0, 0, 0, 0, 0, 1],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 2, 0, 0]])
        tf = self.time_segments[-1]
        A_eq_end = np.array([[tf**5, tf**4, tf**3, tf**2, tf**1, 1],
                             [5*(tf**4), 4*(tf**3), 3*(tf**2), 2*tf, 1, 0],
                             [20*(tf**3), 12*(tf**2), 6*tf, 2, 0, 0]])
        A_eq = np.block([[A_eq_start, np.zeros((3, 6*(self.num_segments-1)))],
                         [np.zeros((3, 6*(self.num_segments-1))), A_eq_end]])

        beq_x = np.array([self.points[0][0], 0, 0, self.points[-1][0], 0, 0])
        beq_y = np.array([self.points[0][1], 0, 0, self.points[-1][1], 0, 0])
        beq_z = np.array([self.points[0][2], 0, 0, self.points[-1][2], 0, 0])

        if self.num_segments == 1:
            return A_eq, A_eq, A_eq, beq_x, beq_y, beq_z
        else: #intermediary constraints for 1+ segments
            #waypoints constraints
            for i in range(self.num_segments-1):
                T = self.time_segments[i]
                A_eq_sub = np.array([T**5, T**4, T**3, T**2, T, 1])
                RHS = np.array([0, 0, 0, 0, 0, 1])
                A_eq_sub = np.block([[np.zeros((1, 6*i)), A_eq_sub, np.zeros((1, 6*(self.num_segments-(i+1))))],
                                    [np.zeros((1, 6*(i+1))), RHS, np.zeros((1, max(0, 6*(self.num_segments-(i+2)))))]])
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
                    A_eq_sub_LHS = np.block([[5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
                                             [20*T**3, 12*T**2, 6*T, 2, 0, 0]])
                    A_eq_sub_RHS = np.block([[0, 0, 0, 0, -1, 0],
                                             [0, 0, 0, -2, 0, 0]])
                    A_eq_sub = np.block([np.zeros((2,6*i)), A_eq_sub_LHS, A_eq_sub_RHS, np.zeros((2, 6*(self.num_segments-(i+2))))])
                    A_eq = np.vstack((A_eq,
                                      A_eq_sub))
                    beq_x = np.hstack((beq_x, np.zeros(2)))
                    beq_y = np.hstack((beq_y, np.zeros(2)))
                    beq_z = np.hstack((beq_z, np.zeros(2)))
                return A_eq, A_eq, A_eq, beq_x, beq_y, beq_z
            #acceleration constraint
            else:
                A_eq_x = A_eq
                A_eq_y = A_eq
                A_eq_z = A_eq
                for i in range(self.num_segments-1):
                    T = self.time_segments[i]
                    A_eq_sub_LHS = np.block([[5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0]]) #velocity
                    A_eq_sub_RHS = np.block([[0, 0, 0, 0, -1, 0]])
                    A_eq_sub = np.block([np.zeros((1,6*i)), A_eq_sub_LHS, A_eq_sub_RHS, np.zeros((1, 6*(self.num_segments-(i+2))))])

                    A_eq_x = np.vstack((A_eq_x,A_eq_sub))
                    A_eq_y = np.vstack((A_eq_y,A_eq_sub))
                    A_eq_z = np.vstack((A_eq_z,A_eq_sub))
                    beq_x = np.hstack((beq_x, np.zeros(1)))
                    beq_y = np.hstack((beq_y, np.zeros(1)))
                    beq_z = np.hstack((beq_z, np.zeros(1)))

                    #check if there's an acceleration constraint for this waypoint
                    #you're constraining jerk right now dumbass
                    acc_con_index = np.where(self.acc_cons[:, 3] == (i+1))
                    if len(acc_con_index[0]) == 1: #acc constraint at this waypoint found
                        index = acc_con_index[0][0] #get index value
                        for j in range(3): #cycle through x(0),y(1),z(2)
                            #no specified acceleration
                            if self.acc_cons[index][j] == -1:
                                #add normal continuity constraint
                                A_eq_sub_LHS = np.block([[20*T**3, 12*T**2, 6*T, 2, 0, 0]])
                                A_eq_sub_RHS = np.block([[0, 0, 0, -2, 0, 0]])
                                A_eq_sub = np.block([np.zeros((1,6*i)), A_eq_sub_LHS, A_eq_sub_RHS, np.zeros((1, 6*(self.num_segments-(i+2))))])
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
                                A_eq_sub_LHS = np.array([20*T**3, 12*T**2, 6*T, 2, 0, 0])
                                A_eq_sub_RHS = np.array([0, 0, 0, -2, 0, 0])
                                A_eq_sub = np.block([[np.zeros((1, 6*i)), A_eq_sub_LHS, np.zeros((1, 6*(self.num_segments-(i+1))))],
                                                     [np.zeros((1, 6*(i+1))), A_eq_sub_RHS, np.zeros((1, 6*(self.num_segments-(i+2))))]])
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
                        A_eq_sub_LHS = np.block([[20*T**3, 12*T**2, 6*T, 2, 0, 0]])
                        A_eq_sub_RHS = np.block([[0, 0, 0, -2, 0, 0]])
                        A_eq_sub = np.block([np.zeros((1,6*i)), A_eq_sub_LHS, A_eq_sub_RHS, np.zeros((1, 6*(self.num_segments-(i+2))))])

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
                A_sub = np.block([[(T*j/i)**5, (T*j/i)**4, (T*j/i)**3, (T*j/i)**2, (T*j/i), 1],
                                  [-(T*j/i)**5, -(T*j/i)**4, -(T*j/i)**3, -(T*j/i)**2, -(T*j/i), -1]])
                if j == 0 and k == 0:
                    A = np.block([A_sub, np.zeros((2, 6*(self.num_segments-(k+1))))])
                else:
                    A = np.block([[A],
                                  [np.zeros((2, 6*k)), A_sub, np.zeros((2, 6*(self.num_segments-(k+1))))]])

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
                A_sub = np.block([[5*(T*j/i)**4, 4*(T*j/i)**3, 3*(T*j/i)**2, 2*T*j/i, 1, 0],
                                  [-5*(T*j/i)**4, -4*(T*j/i)**3, -3*(T*j/i)**2, -2*(T*j/i), -1, 0]])
                A = np.block([[A],
                              [np.zeros((2, 6*k)), A_sub, np.zeros((2, 6*(self.num_segments-(k+1))))]])
                b_x = np.hstack((b_x, vel_max, vel_max))
                b_y = np.hstack((b_y, vel_max, vel_max))
                b_z = np.hstack((b_z, vel_max, vel_max))

        return A, b_x, b_y, b_z

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
        print(self.time_cumulative)

        #get hessian
        H = self.get_Hessian()

        #get A_eq
        Aeq_x, Aeq_y, Aeq_z, beq_x, beq_y, beq_z = self.get_Aeq()

        #get A_ineq
        A, b_x, b_y, b_z = self.get_Aineq(self.vel_max, self.dist_threshold)

        x_coeff = self.get_coeffs(Aeq_x, beq_x, A, b_x, H)
        y_coeff = self.get_coeffs(Aeq_y, beq_y, A, b_y, H)
        z_coeff = self.get_coeffs(Aeq_z, beq_z, A, b_z, H)

        #set orientation constraints
        o_cons = np.ones(self.num_segments) #1 - upright, -1 - inverted
        if not self.no_o_cons:
            o_cons = self.o_cons

        traj_struct = (self.time_cumulative, x_coeff, y_coeff, z_coeff, o_cons)
        #print('cum time', self.time_cumulative)
        #print('seg time', self.time_segments)

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
            x = x_coeff[0+6*i:6+6*i]
            y = y_coeff[0+6*i:6+6*i]
            z = z_coeff[0+6*i:6+6*i]
            #velocity
            dx = [0, 5*x[0], 4*x[1], 3*x[2], 2*x[3], x[4]]
            dy = [0, 5*y[0], 4*y[1], 3*y[2], 2*y[3], y[4]]
            dz = [0, 5*z[0], 4*z[1], 3*z[2], 2*z[3], z[4]]
            #acceleration
            ddx = [0, 0, 20*x[0], 12*x[1], 6*x[2], 2*x[3]]
            ddy = [0, 0, 20*y[0], 12*y[1], 6*y[2], 2*y[3]]
            ddz = [0, 0, 20*z[0], 12*z[1], 6*z[2], 2*z[3]]
            #jerk
            dddx = [0, 0, 0, 60*x[0], 24*x[1], 6*x[2]]
            dddy = [0, 0, 0, 60*y[0], 24*y[1], 6*y[2]]
            dddz = [0, 0, 0, 60*z[0], 24*z[1], 6*z[2]]

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
        exit()


