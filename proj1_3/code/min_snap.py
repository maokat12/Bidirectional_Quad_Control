import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

class MinSnap(object):
    def __init__(self, points, dist_vel, vel_max, dist_threshold):
        # dist_vel (m/s) - velocity used to calculate time segment times
        # vel_max (m/s) - max/min velocity quad can travel
        # dist_threshold (m) - corridor radius for quad to stray

        self.points = points
        self.distances = []
        self.time_segments = []
        self.time_cumulative = []
        self.num_segments = None
        self.dist_vel = dist_vel
        self.vel_max = vel_max
        self.dist_threshold = dist_threshold
        self.plt = False
        self.start = self.points[0]
        self.acc_cons = None

        #remove second point in maze
        #print(self.points)
        #self.points = np.delete(self.points, 1, 0)
        #print(self.points)
        #exit()

    def get_distance(self, p1, p2):  # return distance between two points
        p1 = np.array(p1)
        p2 = np.array(p2)
        dist = abs(np.linalg.norm(p2 - p1))
        return dist

    def set_acc_cons(self, acc_cons):
        self.acc_cons = acc_cons

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
        #intermediary constraints for 1+ segments
        if self.num_segments > 1:
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

            if self.acc_cons == None: #no acceleration constraints
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
            else: #acceleration constraints
                j = 0
                for i in range(self.num_segments-1):
                    T = self.time_segments[i]
                    A_eq_sub_LHS = np.block([[7*T**6, 6*T**5, 5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
                                             [42*T**5, 30*T**4, 20*T**3, 12*T**2, 6*T, 2, 0, 0]])
                    A_eq_sub_RHS = np.block([[0, 0, 0, 0, 0, 0, -1, 0],
                                             [0, 0, 0, 0, 0, -2, 0, 0]])
                    A_eq_sub = np.block([np.zeros((2,8*i)), A_eq_sub_LHS, A_eq_sub_RHS, np.zeros((2, 8*(self.num_segments-(i+2))))])
                    A_eq = np.vstack((A_eq,
                                      A_eq_sub))
                    beq_x = np.hstack((beq_x, np.zeros(2)))
                    beq_y = np.hstack((beq_y, np.zeros(2)))
                    beq_z = np.hstack((beq_z, np.zeros(2)))

                    #all acc constraints
                    A_eq_sub_LHS = np.array([210*T**4, 120*T**3, 60*T**2, 24*T, 6, 0, 0, 0])
                    A_eq_sub_RHS = np.array([0, 0, 0, 0, -6, 0, 0, 0])
                    A_eq_sub = np.block([[np.zeros(8*i), A_eq_sub_LHS, np.zeros(8*(self.num_segments-(i+1)))],
                                         [np.zeros(8*(i+1)), A_eq_sub_RHS, np.zeros(8*(self.num_segments-(i+2)))]])
                    A_eq = np.vstack((A_eq,
                                      A_eq_sub))
                    beq_x = np.hstack((beq_x, self.acc_cons[i+1][0], self.acc_cons[i+1][0]))
                    beq_y = np.hstack((beq_y, self.acc_cons[i+1][1], self.acc_cons[i+1][1]))
                    beq_z = np.hstack((beq_z, self.acc_cons[i+1][2], self.acc_cons[i+1][2]))

                    ''' only top and bottom acc constraints
                    if i % 8 == 4 or i % 8 == 0: #force acc constraints on only some locations THIS IS STILL NOT WORKING
                        A_eq_sub_LHS = np.array([210*T**4, 120*T**3, 60*T**2, 24*T, 6, 0, 0, 0])
                        A_eq_sub_RHS = np.array([0, 0, 0, 0, -6, 0, 0, 0])
                        A_eq_sub = np.block([[np.zeros(8*i), A_eq_sub_LHS, np.zeros(8*(self.num_segments-(i+1)))],
                                             [np.zeros(8*(i+1)), A_eq_sub_RHS, np.zeros(8*(self.num_segments-(i+2)))]])
                        A_eq = np.vstack((A_eq,
                                          A_eq_sub))
                        #beq_x = np.hstack((beq_x, self.acc_cons[i+1][0], self.acc_cons[i+1][0]))
                        #beq_y = np.hstack((beq_y, self.acc_cons[i+1][1], self.acc_cons[i+1][1]))
                        #beq_z = np.hstack((beq_z, self.acc_cons[i+1][2], self.acc_cons[i+1][2]))
                        beq_x = np.hstack((beq_x, self.acc_cons[j][0], self.acc_cons[j][0]))
                        beq_y = np.hstack((beq_y, self.acc_cons[j][1], self.acc_cons[j][1]))
                        beq_z = np.hstack((beq_z, self.acc_cons[j][2], self.acc_cons[j][2]))
                        j = j+1
                    '''
        '''
        import csv
        with open('a_eq.csv', 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            # writing the data rows
            #a = np.hstack([x, x_des, a_des])
            csvwriter.writerows(A_eq)
        with open('b_eq.csv', 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            # writing the data rows
            a = np.vstack([beq_x, beq_y, beq_z])
            csvwriter.writerows(a)
        exit()
        '''
        return A_eq, beq_x, beq_y, beq_z

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

        #get hessian,
        #f = np.zeros((1, 8*self.num_segments))
        H = self.get_Hessian()

        #get A_eq
        Aeq, beq_x, beq_y, beq_z = self.get_Aeq()

        #get A_ineq
        A, b_x, b_y, b_z = self.get_Aineq(self.vel_max, self.dist_threshold)

        x_coeff = self.get_coeffs(Aeq, beq_x, A, b_x, H)
        y_coeff = self.get_coeffs(Aeq, beq_y, A, b_y, H)
        z_coeff = self.get_coeffs(Aeq, beq_z, A, b_z, H)

        traj_struct = (self.time_cumulative, x_coeff, y_coeff, z_coeff)
        #print('cum time', self.time_cumulative)
        #print('seg time', self.time_segments)

        if self.plt:
            self.plot(x_coeff, y_coeff, z_coeff)
            #self.plot_circle(x_coeff, y_coeff, z_coeff)
            exit()

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

    def plot_circle(self, x_coeff, y_coeff, z_coeff):
        cont_time = 0
        plt.figure(5)
        ax = plt.axes(projection='3d')
        w_c = 1
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

            #if i == 0:
            #print('coeffs', x)
            #print('poly only', np.polyval(np.array(x), time_math))
            #print('x_array', np.polyval(np.array(x), time_math)*np.cos(w_c*time_plot))
            #print('time_math', time_math)
            #print('time_plot', time_plot)


            x_pos = np.polyval(np.array(x), time_math)*np.cos(w_c*time_plot)
            x_vel = np.polyval(np.array(dx), time_math)*np.cos(w_c*time_plot) - x_pos*w_c*np.sin(w_c*time_plot)
            x_acc = np.polyval(np.array(ddx), time_math)*np.sin(w_c*time_plot) - 2*w_c*x_vel*np.sin(w_c*time_plot) - w_c**2*x_pos*np.cos(w_c*time_plot)
            x_jerk = (np.polyval(np.array(dddx), time_math)-3*w_c**2*x_vel)*np.cos(w_c*time_plot) - (3*w_c*x_acc - w_c**3*x_pos)*np.sin(w_c*time_plot)

            y_pos = np.polyval(np.array(x), time_math)*np.sin(w_c*time_plot) + self.start[1]
            y_vel = np.polyval(np.array(dx), time_math)*np.sin(w_c*time_plot) + x_pos*w_c*np.cos(w_c*time_plot)
            y_acc = np.polyval(np.array(ddx), time_math)*np.sin(w_c*time_plot) + 2*w_c*x_vel*np.cos(w_c*time_plot) - w_c**2*x_pos*np.sin(w_c*time_plot)
            y_jerk = (np.polyval(np.array(dddx), time_math)-3*w_c**2*x_vel)*np.sin(w_c*time_plot) + (3*w_c*x_acc - w_c**3*x_pos)*np.cos(w_c*time_plot)

            z_pos = np.polyval(np.array(z), time_math)*0.0
            z_vel = np.polyval(np.array(dz), time_math)*0.0
            z_acc = np.polyval(np.array(ddz), time_math)*0.0
            z_jerk = np.polyval(np.array(dddz), time_math)*0.0

            #print('x_pos', x_pos)
            #print('time_math', time_math)
            #print('time_plot', time_plot)


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
        #ax.set_aspect('equal')
        plt.show()


