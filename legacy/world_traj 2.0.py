import numpy as np

from .graph_search import graph_search


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