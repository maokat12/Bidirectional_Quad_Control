import numpy as np

from .graph_search import graph_search
#from scipy import optimize, rosen, rosen_der

class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # these values don't change
        self.vel = 3.5  # m/s - self selected
        self.x_dot = np.zeros((3,))
        self.x_ddot = np.zeros((3,))
        self.x_dddot = np.zeros((3,))
        self.x_ddddot = np.zeros((3,))
        self.yaw = 0
        self.yaw_dot = 0


        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.2, 0.2, 0.2])
        self.margin = 0.5

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = np.zeros((1,3)) # shape=(n_pts,3)

        self.points = self.sparse_waypoints(self.path)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE

        self.start = self.points[0]  # m
        self.end = self.points[-1] #m
        self.points = np.asarray(self.points)  # convert points list to np array

        # declare starting position as first point
        # self.start = points[0]  # m
        self.x = self.start
        self.t_start = 0  # starting time at each update loop
        # points = np.asarray(points)  # convert points list to np array
        self.i = 0  # to account for the np.inf case

        self.traj_struct, self.max_dist = self.naive_trajectory(self.points, self.start, self.x_dot, self.vel)
        #self.traj_struct = self.min_snap_traj(self.points, self.start, self.x_dot, self.vel)
        #print(self.traj_struct[0])
        #exit()
        print(self.traj_struct)
        #self.max_dist = np.max(np.transpose(np.array(self.traj_struct))[0])
        #print(self.max_dist)
        #print(self.traj_struct)
        #exit()

    def min_snap_traj(self, points, start, x_dot, vel):
        time_segments = [0]
        #get timestamps - cumulative time
        for i in range(len(points)):
            dist = self.get_distance(points[i - 1], points[i])
            time_segments.append(self.get_time_seg(dist, vel))
        print('time segments', time_segments)

        '''
        print('time cumulative', time_cumulative)

        #convert to time per segement
        time_segments = [0]
        print(len(time_cumulative[1:]))
        for i in range(len(time_cumulative[1:])):
            time_segments.append(time_cumulative[i] - time_cumulative[i-1])
        '''

        num_segments = len(time_segments) - 1
        print('number of segments', num_segments)

        #starting bound conditions
        A_start = np.array([[0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 2, 0, 0],
                           [0, 0, 0, 6, 0, 0, 0]])
        bx_start = np.array([self.start[0], #starting x
                            0, #starting vel
                            0, #starting acc
                            0]) #starting jerk
        by_start = np.array([self.start[1], 0, 0, 0])
        bz_start = np.array([self.start[2], 0, 0, 0])

        print('A_start', A_start)
        print('bx start', bx_start)

        #ending bound conditions
        tf = time_segments[-1]
        A_end = np.array([[tf**6, tf**5, tf**4, tf**3, tf**2, tf**1, 1],
                         [6*tf**5, 5*tf**4, 4*tf**3, 3*tf**2, 2*tf, 1, 0],
                         [30*tf**4, 20*tf**3, 12*tf**2, 6*tf, 2, 0, 0],
                         [120*tf**3, 60*tf**2, 24*tf, 6, 0, 0, 0]])
        bx_end = np.array([self.end[0], 0, 0, 0])
        by_end = np.array([self.end[1], 0, 0, 0])
        bz_end = np.array([self.end[2], 0, 0, 0])

        print('A end', A_end)
        print('bx end', bx_end)

        #intermediate bound condditions (LHS only)
        A_waypoints = np.zeros(7)
        bx_waypoints = np.array([])
        by_waypoints = np.array([])
        bz_waypoints = np.array([])
        for i in range(len(time_segments[1:-1])): #ignore first & last waypoints
            t = time_segments[i]
            A_waypoints = np.vstack((A_waypoints, np.array([t**6, t**5, t**4, t**3, t**2, t**1, 1])))
            bx_waypoints = np.append(bx_waypoints, [points[i+1][0], points[i+1][0]])
            by_waypoints = np.append(bx_waypoints, [points[i+1][1], points[i+1][1]])
            bz_waypoints = np.append(bx_waypoints, [points[i+1][2], points[i+1][2]])
        A_waypoints = np.delete(A_waypoints, 0, 0) # remove first row of zeroes
        RHS_waypoints = np.array([0, 0, 0, 0, 0, 0, 1])

        print('A waypoints', A_waypoints)
        print('bx waypoint', bx_waypoints)

        #boundary conditions (LHS only)
        A_continuity = np.zeros(7)
        for i in time_segments[1:-1]: # ignore first & last waypoints
            A_continuity = np.vstack((A_continuity, np.array([[6*i**5, 5*i**4, 4*i**3, 3*i**2, 2*i, 1, 0],
                                                             [30*i**4, 20*i**3, 12*i**2, 6*i, 2, 0, 0],
                                                             [120*i**3, 60*i**2, 24*i, 6, 0, 0, 0],
                                                             [360*i**2, 120*i, 24, 0, 0, 0, 0],
                                                             [720*i, 120, 0, 0, 0, 0, 0]])))
        A_continuity = np.delete(A_continuity, 0, 0)  # remove first line of 0s
        RHS_continuity = np.array([[0, 0, 0, 0, 0, -1, 0],
                                   [0, 0, 0, 0, -2, 0, 0],
                                   [0, 0, 0, 0, -6, 0, 0],
                                   [0, 0, 0, -24, 0, 0, 0],
                                   [0, 0, -120, 0, 0, 0, 0]])

        print('RHS Continuity', RHS_continuity)
        print('A continuity', A_continuity)

        #build A from top to bottom following order in the lecture 10 slides
        #start/end
        A = np.block([[A_start, np.zeros((4, 7*(num_segments-1)))],
                      [np.zeros((4, 7*(num_segments-1))), A_end]])
        #intermediate waypoint positions
        for i in range(num_segments-1):
            block = np.block([[np.zeros((1, 7 * i)), A_waypoints[i], np.zeros((1, 7 * (num_segments -1-i)))],
                              [np.zeros((1, 7 * (i+1))), RHS_waypoints, np.zeros((1, 7 * (num_segments -2-i)))]])
            A = np.vstack((A, block))
        #continuity constraints
        for i in range(num_segments-1):
            block = np.block([[np.zeros((1, 7*i)), A_continuity[i*5], RHS_continuity[0], np.zeros((1, 7*(num_segments-2-i)))],
                              [np.zeros((1, 7*i)), A_continuity[i*5+1], RHS_continuity[1],  np.zeros((1, 7*(num_segments-2-i)))],
                              [np.zeros((1, 7*i)), A_continuity[i*5+2], RHS_continuity[2],  np.zeros((1, 7*(num_segments-2-i)))],
                              [np.zeros((1, 7*i)), A_continuity[i*5+3], RHS_continuity[3],  np.zeros((1, 7*(num_segments-2-i)))],
                              [np.zeros((1, 7*i)), A_continuity[i*5+4], RHS_continuity[4],  np.zeros((1, 7*(num_segments-2-i)))]])
            A = np.vstack((A, block))

        Bx = np.append(bx_start, bx_end)
        Bx = np.append(Bx, bx_waypoints)
        Bx = np.append(Bx, np.zeros(5*(num_segments-1)))
        By = np.append(by_start, by_end)
        By = np.append(By, by_waypoints)
        By = np.append(By, np.zeros(5*(num_segments-1)))
        Bz = np.append(bz_start, bz_end)
        Bz = np.append(Bz, bz_waypoints)
        Bz = np.append(Bz, np.zeros(5 * (num_segments - 1)))

        #linear_constraint = optimize.LinearConstraint(A, Bx, Bx)
        #x0 = np.ones(7*num_segments) #initial guess

        #optimize x
        #test = optimize.minimize(rosen,x0, method = 'trust_constr', jac = rosen_der,
        #                         constraints = [linear_constraint], options = {'verbose':1})
        #res = optimize.minimize(rosen, x0, method = 'trust-constr', jac = rosen_der, hess = rosen_hess_p,
        #                        constraints = [linear_constraint, []],
        #                        options = {'verbose':1}, bounds = [])


        exit()

    def naive_trajectory(self, points, start, x_dot, vel):
        # create trajectory structure
        traj_struct = []  # structure for trajectory
        distance = []
        # (segment distance, segment direction (unit vector), time on segment, velocity vector)

        time_seg = 0  # ongoing time

        traj_struct.append([0, start, time_seg, x_dot])

        for i in range(len(points)):
            if i != 0:
                # print('\ninput: ', points[i])

                dist = self.get_distance(points[i - 1], points[i])
                direction = self.get_direction(points[i - 1], points[i], dist)
                time_seg = time_seg + self.get_time_seg(dist, vel)
                x_dot = vel * direction
                distance.append(dist)

                # print('dist: ', dist)
                # print('direction: ', direction)
                # print('time_seg: ', time_seg)
                # print('x_dot: ', x_dot)

                traj_struct.append([dist, direction, time_seg, x_dot])

        max_dist = max(distance)
        '''
        #print(max_dist)
        #correct velocities for turning points
        time_seg = 0
        for i in range(len(traj_struct)):
            vel_mod = vel
            if i != 0:
                if traj_struct[i][0] < 0.1*max_dist:
                    vel_mod = vel*0.5
                time_seg = time_seg + self.get_time_seg(traj_struct[i][0], vel_mod)
                x_dot = vel_mod * traj_struct[i][1]
                traj_struct[i][2] = time_seg
                traj_struct[i][3] = x_dot
        #print(traj_struct)
        #exit()
        '''
        return traj_struct, max_dist


    def sparse_waypoints(self, dense_path):
        sparse_points = [dense_path[0]]
        direction = (dense_path[1]-dense_path[0])/np.linalg.norm(dense_path[1]-dense_path[0])
        #print('dense path length', len(dense_path))

        for i in range(len(dense_path[1:-1])): #start from the third point
            segment = dense_path[i+1]-dense_path[i]
            if (segment/np.linalg.norm(segment)).tolist() != direction.tolist():
                #print(i)
                #print('direction', direction)
                #print('segment', segment)
                #print('norm segment', np.linalg.norm(segment))
                #print('unit direction', (segment) / np.linalg.norm(segment))

                sparse_points.append(dense_path[i+1])
                direction = segment/np.linalg.norm(segment)

        sparse_points.append(dense_path[-1]) #add final location

        print('sparse path length', len(sparse_points))
        print(sparse_points)
        return sparse_points

    def get_distance(self, p1, p2): #return distance between two points
        dist = abs(np.linalg.norm(p2 - p1))
        return dist

    def get_direction(self, p1, p2, distance): #return unit vector between two points
        #p2 = p_(i+1), p1 = p_i
        direction = (p2 - p1) / distance
        return direction

    def get_time_seg(self, distance, velocity): #return time to travel segment
        time = distance/(velocity-1)
        return time


    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        #self.x        = np.zeros((3,))
        #self.x_dot    = np.zeros((3,))
        #self.x_ddot   = np.zeros((3,))
        #self.x_dddot  = np.zeros((3,))
        #self.x_ddddot = np.zeros((3,))
        #self.yaw = 0
        #self.yaw_dot = 0

        # STUDENT CODE HERE
        #if t != np.inf:
        #accounting for the initial np.inf runs
        if self.i == 0 and t == np.inf:
            #self.x_dot = float(0) * self.x_dot
            self.x = self.end
            self.i = self.i+1
            #self.t_start = 0
        elif self.i == 1 and t == np.inf:
            self.yaw = self.yaw
            self.i = self.i+1
        #print(t)
        #check if quadrotor has reached final location
        else:
            if self.i == 2:
                self.x = self.start
                self.i = self.i + 1
            if t > self.traj_struct[-1][2]:
                #exit()
                self.x_dot = float(0) * self.x_dot
                self.x = self.end
                self.t_start = t

            for i in range(len(self.traj_struct)):
                if t < self.traj_struct[i][2]:
                    self.x_dot = self.traj_struct[i][3]
                    self.x = self.x + self.x_dot*(t - self.t_start)
                    self.t_start = t
                    break


        flat_output = { 'x':self.x, 'x_dot':self.x_dot, 'x_ddot':self.x_ddot, 'x_dddot':self.x_dddot, 'x_ddddot':self.x_ddddot,
                        'yaw':self.yaw, 'yaw_dot':self.yaw_dot}
        #print('flat output: ', flat_output)
        return flat_output


    '''
    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
    '''