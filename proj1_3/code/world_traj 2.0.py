import numpy as np
import cvxpy as cp

from .graph_search import graph_search

## FILE MADE SO I CAN LOOK AT WORLD_TRAJ.PY IN SPLIT SCREEN
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
        self.vel = 3.45  # m/s - self selected
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
        self.resolution = np.array([0.25, 0.25, 0.25])
        self.margin = 0.6

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = np.zeros((1, 3))  # shape=(n_pts,3)

        self.points = self.sparse_waypoints(self.path)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE

        self.start = self.points[0]  # m
        self.end = self.points[-1]  # m
        self.points = np.asarray(self.points)  # convert points list to np array

        # declare starting position as first point
        # self.start = points[0]  # m
        self.x = self.start
        self.t_start = 0  # starting time at each update loop
        # points = np.asarray(points)  # convert points list to np array
        self.i = 0  # to account for the np.inf case

        self.traj_struct = self.naive_trajectory(self.points, self.start, self.x_dot, self.vel)

    def min_jerk_traj_kinda(self, points, start, x_dot, vel):
        time_segments = [0]
        # get timestamps - time per segment
        for i in range(len(points[:-1])):
            dist = self.get_distance(points[i +1], points[i])
            time_segments.append(self.get_time_seg(dist, vel)*2)
        print('time segments', time_segments)

        # convert to time per cumulative
        time_cumulative = []
        for i in range(len(time_segments)):
            time_cumulative.append(sum(time_segments[0:i+1]))
        print('time cumulative', time_cumulative)
        #exit()

        num_segments = len(time_segments) - 1
        print('number of segments', num_segments)

        # starting bound conditions
        A_start = np.array([[0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 2, 0, 0],
                            [0, 0, 0, 6, 0, 0, 0]])
        bx_start = np.array([self.start[0],  # starting x
                             0,  # starting vel
                             0,  # starting acc
                             0])  # starting jerk
        by_start = np.array([self.start[1], 0, 0, 0])
        bz_start = np.array([self.start[2], 0, 0, 0])

        #print('A_start', A_start)
        #print('bx start', bx_start)

        # ending bound conditions
        tf = time_segments[-1]
        A_end = np.array([[tf**6, tf**5, tf**4, tf**3, tf**2, tf**1, 1],
                          [6*(tf**5), 5*(tf**4), 4*(tf**3), 3*(tf**2), 2*tf, 1, 0],
                          [30*(tf**4), 20*(tf**3), 12*(tf**2), 6*tf, 2, 0, 0],
                          [120*(tf**3), 60*(tf**2), 24*tf, 6, 0, 0, 0]])
        print(self.end[0])
        bx_end = np.array([self.end[0], 0, 0, 0])
        by_end = np.array([self.end[1], 0, 0, 0])
        bz_end = np.array([self.end[2], 0, 0, 0])

        #print('A end', A_end)
        #print('bx end', bx_end)

        # intermediate bound condditions (LHS only)
        A_waypoints = np.zeros(7)
        bx_waypoints = np.array([])
        by_waypoints = np.array([])
        bz_waypoints = np.array([])
        for i in range(len(time_segments[1:-1])):  # ignore first & last waypoints
            t = time_segments[i]
            A_waypoints = np.vstack((A_waypoints, np.array([t**6, t**5, t**4, t**3, t**2, t**1, 1])))
            bx_waypoints = np.append(bx_waypoints, [points[i+1][0], points[i+1][0]])
            by_waypoints = np.append(by_waypoints, [points[i+1][1], points[i+1][1]])
            bz_waypoints = np.append(bz_waypoints, [points[i+1][2], points[i+1][2]])
        A_waypoints = np.delete(A_waypoints, 0, 0)  # remove first row of zeroes
        RHS_waypoints = np.array([0, 0, 0, 0, 0, 0, 1])

        #print('A waypoints', A_waypoints)
        #print('bx waypoint', bx_waypoints)

        # boundary conditions (LHS only)
        A_continuity = np.zeros(7)
        for i in time_segments[1:-1]:  # ignore first & last waypoints
            A_continuity = np.vstack(
                (A_continuity, np.array([[6*(i**5), 5*(i**4), 4*(i**3), 3*(i**2), 2*i, 1, 0],
                                         [30*(i**4), 20*(i**3), 12*(i**2), 6*i, 2, 0, 0],
                                         [120*(i**3), 60*(i**2), 24*i, 6, 0, 0, 0],
                                         [360*(i**2), 120*i, 24, 0, 0, 0, 0],
                                         [720*i, 120, 0, 0, 0, 0, 0]])))
        A_continuity = np.delete(A_continuity, 0, 0)  # remove first line of 0s
        RHS_continuity = np.array([[0, 0, 0, 0, 0, -1, 0],
                                   [0, 0, 0, 0, -2, 0, 0],
                                   [0, 0, 0, -6, 0, 0, 0],
                                   [0, 0, -24, 0, 0, 0, 0],
                                   [0, -120, 0, 0, 0, 0, 0]])

        #print('RHS Continuity', RHS_continuity)
        #print('A continuity', A_continuity)

        # build A from top to bottom following order in the lecture 10 slides
        # start/end
        A = np.block([[A_start, np.zeros((4, 7 * (num_segments-1)))],
                      [np.zeros((4, 7 * (num_segments-1))), A_end]])
        print(np.shape(A_start))
        print(np.shape(A))
        # intermediate waypoint positions
        for i in range(num_segments - 1):
            block = np.block([[np.zeros((1, 7 * i)), A_waypoints[i], np.zeros((1, 7 * (num_segments - 1 - i)))],
                              [np.zeros((1, 7 * (i + 1))), RHS_waypoints, np.zeros((1, 7 * (num_segments - 2 - i)))]])
            A = np.vstack((A, block))
        # continuity constraints
        for i in range(num_segments - 1):
            block = np.block([[np.zeros((1, 7*i)), A_continuity[i * 5], RHS_continuity[0],
                               np.zeros((1, 7*(num_segments - 2-i)))],
                              [np.zeros((1, 7*i)), A_continuity[i*5 + 1], RHS_continuity[1],
                               np.zeros((1, 7*(num_segments - 2-i)))],
                              [np.zeros((1, 7*i)), A_continuity[i*5 + 2], RHS_continuity[2],
                               np.zeros((1, 7*(num_segments - 2-i)))],
                              [np.zeros((1, 7*i)), A_continuity[i*5 + 3], RHS_continuity[3],
                               np.zeros((1, 7*(num_segments - 2-i)))],
                              [np.zeros((1, 7*i)), A_continuity[i*5 + 4], RHS_continuity[4],
                               np.zeros((1, 7*(num_segments - 2-i)))]])
            A = np.vstack((A, block))

        Bx = np.append(bx_start, bx_end)
        Bx = np.append(Bx, bx_waypoints)
        Bx = np.append(Bx, np.zeros(5 * (num_segments - 1)))
        By = np.append(by_start, by_end)
        By = np.append(By, by_waypoints)
        By = np.append(By, np.zeros(5 * (num_segments - 1)))
        Bz = np.append(bz_start, bz_end)
        Bz = np.append(Bz, bz_waypoints)
        Bz = np.append(Bz, np.zeros(5 * (num_segments - 1)))

        # linear_constraint = optimize.LinearConstraint(A, Bx, Bx)
        # x0 = np.ones(7*num_segments) #initial guess

        # optimize x
        # test = optimize.minimize(rosen,x0, method = 'trust_constr', jac = rosen_der,
        #                         constraints = [linear_constraint], options = {'verbose':1})
        # res = optimize.minimize(rosen, x0, method = 'trust-constr', jac = rosen_der, hess = rosen_hess_p,
        #                        constraints = [linear_constraint, []],
        #                        options = {'verbose':1}, bounds = [])
        #print('A shape', np.shape(A))
        #print('Bx shape', np.shape(Bx))
        #print('By shape', np.shape(By))
        #print('Bz shape', np.shape(Bz))

        #Bx=Bx.astype('float64')

        cx = np.linalg.lstsq(A, Bx, rcond=None)[0]
        cy = np.linalg.lstsq(A, By, rcond=None)[0]
        cz = np.linalg.lstsq(A, Bz, rcond=None)[0]
        print(cx)
        print(cy)
        print(cz)

        return time_cumulative, cx, cy, cz

    def min_snap_1_traj(self, start, end, max_vel):
        print('start', start)
        print('end', end)
        #get time
        dist = self.get_distance(end, start)
        tf = self.get_time_seg(dist, max_vel)
        time_segments = [0, tf]
        time_cumulative = [0, tf]
        num_segments = 1
        points = np.array([start, end])
        print('final time', tf)

        #create main matrix
        H = np.array([[1050*tf**7, 525*tf**6, 210*tf**5, 52.5*tf**4],
                      [525*tf**6, 270*tf**5, 112.5*tf**4, 30*tf**3],
                      [210*tf**5, 112.5*tf**4, 50*tf**3, 15*tf**2],
                      [52.5*tf**4, 30*tf**3, 15*tf**2, 6*tf]])
        H = np.block([[96*H, np.zeros((4, 4))],
                      [np.zeros((4, 8))]])

        for i in range(len(time_segments[2:])):
            tf = time_segments[i+2]
            H_sub = np.array([[1050*tf**7, 525*tf**6, 210*tf**5, 52.5*tf**4],
                              [525*tf**6, 270*tf**5, 112.5*tf**4, 30*tf**3],
                              [210*tf**5, 112.5*tf**4, 50*tf**3, 15*tf**2],
                              [52.5*tf**4, 30*tf**3, 15*tf**2, 6*tf]])
            H_sub = np.block([[96 * H_sub, np.zeros((4, 4))],
                              [np.zeros((4, 8))]])
            H = np.block([[H, np.zeros((8*(i+1),8))],
                          [np.zeros((8,8*(i+1))), H_sub]])

        #equality constraints
        '''
        A_eq = np.block([[np.zeros((1,7)), 1],
                        [tf**7, tf**6, tf**5, tf**4, tf**3, tf**2, tf, 1]])
        bx_eq = [start[0], end[0]]
        by_eq = [start[1], end[1]]
        bz_eq = [start[2], end[2]]
        '''
        # starting bound conditions
        A_start = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                            #[0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 2, 0, 0],
                            [0, 0, 0, 0, 6, 0, 0, 0]])
        bx_start = np.array([start[0],  # starting x
                             #1.5,  # starting vel
                             0,  # starting acc
                             0])  # starting jerk
        by_start = np.array([start[1],
                             #1.5, #vel
                             0, #acc
                             0]) #jerk
        bz_start = np.array([start[2],
                             #1.5, #vel
                             0, #acc
                             0]) #jerk

        #ending bound conditions
        A_end = np.array([[tf**7, tf**6, tf**5, tf**4, tf**3, tf**2, tf**1, 1]])#,
                          #[7*(tf**6), 6*(tf**5), 5*(tf**4), 4*(tf**3), 3*(tf**2), 2*tf, 1, 0],
                          #[42*(tf**5), 30*(tf**4), 20*(tf**3), 12*(tf**2), 6*tf, 2, 0, 0],
                          #[210*(tf**4), 120*(tf**3), 60*(tf**2), 24*tf, 6, 0, 0, 0]])
        #print(self.end[0])
        bx_end = np.array([end[0]])#, 0, 0, 0])
        by_end = np.array([end[1]])#, 0, 0, 0])
        bz_end = np.array([end[2]])#, 0, 0, 0])

        # intermediate bound conditions (LHS only)
        A_waypoints = np.zeros(8)
        bx_waypoints = np.array([])
        by_waypoints = np.array([])
        bz_waypoints = np.array([])
        for i in range(len(time_segments[1:-1])):  # ignore first and last time segment
            t = time_segments[i+1]
            A_waypoints = np.vstack((A_waypoints, np.array([t**7, t**6, t**5, t**4, t**3, t**2, t**1, 1])))
            bx_waypoints = np.append(bx_waypoints, [points[i+1][0], points[i+1][0]])
            by_waypoints = np.append(by_waypoints, [points[i+1][1], points[i+1][1]])
            bz_waypoints = np.append(bz_waypoints, [points[i+1][2], points[i+1][2]])
        A_waypoints = np.delete(A_waypoints, 0, 0)  # remove first row of zeroes
        RHS_waypoints = np.array([0, 0, 0, 0, 0, 0, 0, 1])

        # boundary conditions (LHS only)
        A_continuity = np.zeros(8)
        for i in time_segments[1:-1]:  # ignore first & last waypoints
            A_continuity = np.vstack(
                (A_continuity, np.array([[7*(i**6), 6*(i**5), 5*(i**4), 4*(i**3), 3*(i**2), 2*i, 1, 0],
                                         [42*(i**5), 30*(i**4), 20*(i**3), 12*(i**2), 6*i, 2, 0, 0],
                                         [210*(i**4), 120*(i**3), 60*(i**2), 24*i, 6, 0, 0, 0]])))#,
                                         #[840*(i**3), 360*(i**2), 120*i, 24, 0, 0, 0, 0]])))#,
                                         #[2520*(i**2), 720*i, 120, 0, 0, 0, 0, 0],
                                         #[5040*i, 720, 0, 0, 0, 0, 0, 0]])))
        A_continuity = np.delete(A_continuity, 0, 0)  # remove first line of 0s
        RHS_continuity = np.array([[0, 0, 0, 0, 0, 0, -1, 0],
                                   [0, 0, 0, 0, 0, -2, 0, 0],
                                   [0, 0, 0, 0, -6, 0, 0, 0]])#,,w
                                   #[0, 0, 0, -24, 0, 0, 0, 0]])#,
                                   #[0, 0, -120, 0, 0, 0, 0, 0],
                                   #[0, -720, 0, 0, 0, 0, 0, 0]])

        #combine together
        A_eq = np.block([[A_start, np.zeros((len(bx_start), 8*(num_segments-1)))],
                      [np.zeros((1, 8*(num_segments-1))), A_end]])
        # intermediate waypoint positions
        for i in range(num_segments - 1):
            block = np.block([[np.zeros((1, 8*i)), A_waypoints[i], np.zeros((1, 8*(num_segments-1-i)))],
                              [np.zeros((1, 8*(i+1))), RHS_waypoints, np.zeros((1, 8*(num_segments-2-i)))]])
            A_eq = np.vstack((A_eq, block))
        # continuity constraints
        for i in range(num_segments - 1):
            block = np.block([[np.zeros((1, 8*i)), A_continuity[i*3 + 0], RHS_continuity[0], np.zeros((1, 8*(num_segments-2-i)))],
                              [np.zeros((1, 8*i)), A_continuity[i*3 + 1], RHS_continuity[1], np.zeros((1, 8*(num_segments-2-i)))],
                              [np.zeros((1, 8*i)), A_continuity[i*3 + 2], RHS_continuity[2], np.zeros((1, 8*(num_segments-2-i)))]])#,
                              #[np.zeros((1, 8*i)), A_continuity[i*4 + 3], RHS_continuity[3], np.zeros((1, 8*(num_segments-2-i)))]])#,
                              #[np.zeros((1, 8*i)), A_continuity[i*6 + 4], RHS_continuity[4], np.zeros((1, 8*(num_segments-2-i)))],
                              #[np.zeros((1, 8*i)), A_continuity[i*6 + 5], RHS_continuity[5], np.zeros((1, 8*(num_segments-2-i)))]])
            A_eq = np.vstack((A_eq, block))
        bx_eq = np.append(bx_start, bx_end)
        bx_eq = np.append(bx_eq, bx_waypoints)
        bx_eq = np.append(bx_eq, np.zeros(3*(num_segments-1)))

        by_eq = np.append(by_start, by_end)
        by_eq = np.append(by_eq, by_waypoints)
        by_eq = np.append(by_eq, np.zeros(3*(num_segments-1)))

        bz_eq = np.append(bz_start, bz_end)
        bz_eq = np.append(bz_eq, bz_waypoints)
        bz_eq = np.append(bz_eq, np.zeros(3*(num_segments-1)))
        '''
        #inequality constraints
        A = np.zeros(8)
        b_x = np.array([])
        b_y = np.array([])
        b_z = np.array([])

        #max_vel = max_vel*0.7
        sample_size = 2
        for i in range(1, sample_size):
            frac = 1/sample_size*i
            A = np.vstack((A,
                           #np.array([(frac*tf)**7, (frac*tf)**6, (frac*tf)**5, (frac*tf)**4, (frac*tf)**3, (frac*tf)**2, (frac*tf), 1]), #upper pos bound
                           np.array([7*(frac*tf)**6, 6*(frac*tf)**5, 5*(frac*tf)**4, 4*(frac*tf)**3, 3*(frac*tf)**2, 2*(frac*tf), 1, 0])))
            A = np.vstack((A,
                           #np.array([-(frac*tf)**7, -(frac*tf)**6, -(frac*tf)**5, -(frac*tf)**4, -(frac*tf)**3, -(frac*tf)**2, -(frac*tf), -1]), #lower pos
                           np.array([-7*(frac*tf)**6, -6*(frac*tf)**5, -5*(frac*tf)**4, -4*(frac*tf)**3, -3*(frac*tf)**2, -2*(frac*tf), -1, 0])))
            b_x = np.append(b_x, [max_vel, -1*max_vel])
            b_y = np.append(b_y, [max_vel, -1*max_vel])
            b_z = np.append(b_z, [max_vel, -1*max_vel])
        A = np.delete(A, 0, 0)
        '''
        #create inequality constraints
        tf = time_cumulative[1]
        A_ineq = np.array([[7*(tf**6), 6*(tf**5), 5*(tf**4), 4*(tf**3), 3*(tf**2), 2*tf, 1, 0]])#,
                           #[-7*(tf**6), -6*(tf**5), -5*(tf**4), -4*(tf**3), -3*(tf**2), -2*tf, -1, 0]])
        A_ineq = np.block([[A_ineq, np.zeros((1, 8*(num_segments-1)))]])
        bx_ineq = np.array([max_vel*2])#, -vel])
        by_ineq = np.array([max_vel*2])#, -vel])
        bz_ineq = np.array([max_vel*2])#, -vel])

        for i in range(len(time_cumulative[2:])):
            tf = time_cumulative[i+2]
            A_ineq_sub = np.array([[7*(tf**6), 6*(tf**5), 5*(tf**4), 4*(tf**3), 3*(tf**2), 2*tf, 1, 0]])#,
                                   #[-7*(tf**6), -6*(tf**5), -5*(tf**4), -4*(tf**3), -3*(tf**2), -2*tf, -1, 0]])
            A_ineq = np.block([[A_ineq],
                               [np.zeros((1, 8*(i+1))), A_ineq_sub, np.zeros((1, 8*(num_segments-i-2)))]])
            bx_ineq = np.append(bx_ineq, np.array([max_vel*2.5]))#, -vel]))
            by_ineq = np.append(by_ineq, np.array([max_vel*2.5]))#, -vel]))
            bz_ineq = np.append(bz_ineq, np.array([max_vel*2.5]))#, -vel]))


        ### solving
        x = cp.Variable(8)
        prob_x = cp.Problem(cp.Minimize(cp.quad_form(x, H)),
                          [A_eq @ x == bx_eq,
                           A_ineq @ x <= bx_ineq])
        try:
            prob_x.solve()
            #print("\nThe optimal value is", prob.value)
            print("A solution x is")
            print(x.value)
            print("A dual solution corresponding to the inequality constraints is")
            print(prob_x.constraints[0].dual_value)
        except:
            print("x solve failed")
        exit()
        y = cp.Variable(8)
        prob_y = cp.Problem(cp.Minimize(cp.quad_form(y, H)),
                          [A_eq @ y == by_eq,
                           A_ineq @ y <= by_ineq])
        prob_y.solve()
        #print("\nThe optimal value is", prob_y.value)
        print("A solution y is")
        print(y.value)
        print("A dual solution corresponding to the inequality constraints is")
        print(prob_y.constraints[0].dual_value)

        z = cp.Variable(8)
        prob_z = cp.Problem(cp.Minimize(cp.quad_form(z, H)),
                            [A_eq @ z == bz_eq,
                             A_ineq @ z <= bz_ineq])
        prob_z.solve()
        #print("\nThe optimal value is", prob_z.value)
        print("A solution z is")
        print(z.value)
        print("A dual solution corresponding to the inequality constraints is")
        print(prob_z.constraints[0].dual_value)

        #cx = np.append([0, 0, 0], np.round(prob_x.constraints[0].dual_value, 2))
        #cy = np.append([0, 0, 0], np.round(prob_y.constraints[0].dual_value, 2))
        #cz = np.append([0, 0, 0], np.round(prob_z.constraints[0].dual_value, 2))
        cx = np.round(x.value)
        cy = np.round(y.value)
        cz = np.round(z.value)

        print('cx', cx)
        print('cy', cy)
        print('cz', cz)
        #exit()

        return time_cumulative, cx, cy, cz

    def naive_trajectory(self, points, start, x_dot, vel):
        # create trajectory structure
        traj_struct = []  # structure for trajectory
        distance = []
        # (segment distance, segment direction (unit vector), time on segment, velocity vector)

        time_seg = 0  # ongoing time

        traj_struct.append([0, start, time_seg, x_dot])

        for i in range(len(points)):
            if i != 0:
                dist = self.get_distance(points[i - 1], points[i])
                direction = self.get_direction(points[i - 1], points[i], dist)
                time_seg = time_seg + self.get_time_seg(dist, vel)
                x_dot = vel * direction
                distance.append(dist)

                traj_struct.append([dist, direction, time_seg, x_dot])
        return traj_struct
        '''
        traj_struct.append([0, start])
        for i in range(len(points)):
            if i != 0:
                dist = self.get_distance(points[i - 1], points[i])
                direction = self.get_direction(points[i - 1], points[i], dist)
                distance.append(dist)

                traj_struct.append([dist, direction])
        path = []

        vel_damp = 0.5
        max_dist = max(distance)
        time_seg = 0
        for dist, direction in traj_struct:
            vel_mod = vel
            if i != 0:
                if dist < (0.1*max_dist):
                    vel_mod = vel*vel_damp
                time_seg = time_seg + self.get_time_seg(dist, vel_mod)
                x_dot = vel_mod * direction

                path.append([dist, direction, time_seg, x_dot])

        #slow down for the last 0.1m before goal
        if path[-1][0] > 0.1*max_dist:
            dist, direction, time_seg, x_dot = path.pop()
            dist_1 = dist*0.91
            time_seg_1 = path[-1][2] + self.get_time_seg(dist_1, vel)
            x_dot_1 = vel*direction

            dist_2 = dist*0.09
            time_seg_2 = path[-1][2] + self.get_time_seg(dist_2, vel*vel_damp)
            x_dot_2 = vel*vel_damp * direction

            path.append([dist_1, direction, time_seg_1, x_dot_1])
            path.append([dist_2, direction, time_seg_2, x_dot_2])

        print(path)
        
        return path
        '''
    def sparse_waypoints(self, dense_path):
        sparse_points = [dense_path[0]]
        direction = (dense_path[1] - dense_path[0]) / np.linalg.norm(dense_path[1] - dense_path[0])
        # print('dense path length', len(dense_path))

        for i in range(len(dense_path[1:-1])):  # start from the third point
            segment = dense_path[i + 1] - dense_path[i]
            if (segment / np.linalg.norm(segment)).tolist() != direction.tolist():
                sparse_points.append(dense_path[i + 1])
                direction = segment / np.linalg.norm(segment)

        sparse_points.append(dense_path[-1])  # add final location

        print('sparse path length', len(sparse_points))
        print(sparse_points)
        return sparse_points

    def get_distance(self, p1, p2):  # return distance between two points
        dist = abs(np.linalg.norm(p2 - p1))
        return dist

    def get_direction(self, p1, p2, distance):  # return unit vector between two points
        # p2 = p_(i+1), p1 = p_i
        direction = (p2 - p1) / distance
        return direction

    def get_time_seg(self, distance, velocity):  # return time to travel segment
        time = distance / velocity
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
        # self.x        = np.zeros((3,))
        # self.x_dot    = np.zeros((3,))
        # self.x_ddot   = np.zeros((3,))
        # self.x_dddot  = np.zeros((3,))
        # self.x_ddddot = np.zeros((3,))
        # self.yaw = 0
        # self.yaw_dot = 0

        # STUDENT CODE HERE
        # if t != np.inf:
        # accounting for the initial np.inf runs
        if self.i == 0 and t == np.inf:
            # self.x_dot = float(0) * self.x_dot
            self.x = self.end
            self.i = self.i + 1
            # self.t_start = 0
        elif self.i == 1 and t == np.inf:
            self.yaw = self.yaw
            self.i = self.i + 1
        # print(t)
        # check if quadrotor has reached final location
        else:
            if self.i == 2:
                self.x = self.start
                self.i = self.i + 1
            if t > self.traj_struct[-1][2]:
                # exit()
                self.x_dot = float(0) * self.x_dot
                self.x = self.end
                self.t_start = t

            for i in range(len(self.traj_struct)):
                if t < self.traj_struct[i][2]:
                    self.x_dot = self.traj_struct[i][3]
                    # print('x_dot: ', self.x_dot)
                    self.x = self.x + (t - self.t_start) * self.x_dot
                    self.t_start = t
                    break

        flat_output = {'x': self.x, 'x_dot': self.x_dot, 'x_ddot': self.x_ddot, 'x_dddot': self.x_dddot,
                       'x_ddddot': self.x_ddddot,
                       'yaw': self.yaw, 'yaw_dot': self.yaw_dot}
        # print('flat output: ', flat_output)
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