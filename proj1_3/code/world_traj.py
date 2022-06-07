import numpy as np
import scipy
import cvxpy as cp
from .graph_search import graph_search
import copy
from rdp import rdp
from sympy import cos, sin, diff, symbols
from .min_snap import MinSnap
import shape_traj

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
        self.vel = 2 # m/s - self selected
        self.max_vel = 5 #m/s
            #use 3.4 for naive
        self.x_dot = np.zeros((3,))
        self.x_ddot = np.zeros((3,))
        self.x_dddot = np.zeros((3,))
        self.x_ddddot = np.zeros((3,))
        self.yaw = 0
        self.yaw_dot = 0
        self.k = -1

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.25, 0.25, 0.25])
        self.margin = 0.6
        self.naive = False
        self.num_segments = None

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = np.zeros((0, 3))  # shape=(n_pts,3)

        maze_points = [([1., 5., 1.5]), ([1.125, 4.625, 1.625]), ([1.375, 2.375, 1.625]), ([2.875, 1.125, 1.625]),
                       ([8.625, 1.375, 1.625]), ([8.875, 1.875, 1.625]), ([8.625, 2.625, 1.625]),
                       ([8.125, 2.875, 1.625]), ([6.375, 3.125, 1.625]), ([5.375, 3.875, 1.625]),
                       ([4.625, 4.125, 1.625]), ([4.125, 4.875, 1.625]), ([4.375, 7.125, 1.625]),
                       ([4.875, 7.375, 1.625]), ([7.625, 7.125, 1.625]), ([7.875, 7.125, 1.625]), ([9., 7., 1.5])]

        maze_points_short_removed = [([1., 5., 1.5]), ([1.375, 2.375, 1.625]), ([2.875, 1.125, 1.625]),
                       ([8.625, 1.375, 1.625]), ([8.875, 1.875, 1.625]), ([8.625, 2.625, 1.625]),
                       ([8.125, 2.875, 1.625]), ([6.375, 3.125, 1.625]), ([5.375, 3.875, 1.625]),
                       ([4.625, 4.125, 1.625]), ([4.125, 4.875, 1.625]), ([4.375, 7.125, 1.625]),
                       ([4.875, 7.375, 1.625]), ([7.625, 7.125, 1.625]), ([9., 7., 1.5])]

        window_points = [([0.7, -4.3, 0.7]), ([0.625, -3.875, 0.625]), ([0.875, -1.375, 0.625]),
                         ([2.375, 0.125, 0.875]),
                         ([3.875, 1.875, 2.375]), ([4.125, 12.875, 2.625]), ([5.875, 14.875, 4.375]),
                         ([6.125, 16.125, 4.375]),
                         ([6.625, 16.625, 4.125]), ([7.875, 17.875, 3.125]), ([8., 18., 3.])]

        #self.points = self.sparse_waypoints_rdp(self.path)
        self.points = self.sparse_waypoints(self.path)
        #self.points = np.array(self.points)
        #self.points = maze_points
        #self.points = self.points[0:5]
        #self.points = maze_points_short_removed
        #self.points = self.path
        #self.points = self.points[1:]

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE
        #steps = 4
        self.start = self.points[0]  # m
        self.end = self.points[-1]  # m
        self.points = np.asarray(self.points)
        #self.end = self.points[start_point+steps]
        #self.points = np.asarray(self.points[start_point:start_point+steps+1])  # convert points list to np array
        #print('points', self.points)


        # declare starting position as first point
        self.x = self.start
        self.t_start = 0  # starting time at each update loop
        self.i = 0  # to account for the np.inf case

        #self.points = self.points[2:4] #FOR DEBUGGING

        #print('shortened point list', self.points)

        if self.naive:
            self.traj_struct = self.naive_trajectory(self.points, self.start, self.x_dot, self.vel)
        else: #min snap
            dist_vel = self.vel #m/s
            vel_max = 1.5*self.vel #m/s
            dist_threshold = 0.2 #m

            my_min_snap = MinSnap(self.points, dist_vel, vel_max, dist_threshold)
            self.traj_struct, self.num_segments, self.time_segments = my_min_snap.get_trajectory()
            #print(self.num_segments)
            #print('cum time', self.traj_struct[0])
            #print('seg time', self.time_segments)
            #exit()

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

    def sparse_waypoints(self, dense_path):
        sparse_points = [dense_path[0]]
        direction = (dense_path[1] - dense_path[0]) / np.linalg.norm(dense_path[1] - dense_path[0])
        # print('dense path length', len(dense_path))

        for i in range(len(dense_path[1:-1])):  # start from the third point
            segment = dense_path[i+1] - dense_path[i]
            segment_norm = segment / np.linalg.norm(segment)
            if self.get_distance(dense_path[i], sparse_points[-1]) > 0.65: #make sure segments aren't too close
                if segment_norm.tolist() != direction.tolist(): #check directions aren't the same
                    angle = np.arccos(np.clip(np.dot(segment_norm, direction),-1.0, 1.0))
                    if abs(np.degrees(angle) > 5): # check directions aren't really close together
                        length = np.linalg.norm(segment)
                        #if length > 0.2:
                        sparse_points.append(dense_path[i])
                        direction = segment / length

        sparse_points.append(dense_path[-1])  # add final location

        #print('sparse path length', len(sparse_points))
        #print(sparse_points)
        return sparse_points

    def sparse_waypoints_rdp(self, points):
        sparse_points = rdp(points)
        #print('sparse path ', sparse_points.tolist())
        return rdp(points)

    def get_distance(self, p1, p2):  # return distance between two points
        #print('p1',p1)
        #print('p2',p2)
        dist = abs(np.linalg.norm(p2 - p1))
        #print('dist', dist)
        return dist

    def get_direction(self, p1, p2, distance):  # return unit vector between two points
        # p2 = p_(i+1), p1 = p_i
        direction = (p2 - p1) / distance
        return direction

    def get_time_seg(self, distance, velocity):  # return time to travel segment
        time = distance / velocity
        return time

    def update(self, t):
        if self.naive:
            flat_output = self.naive_update(t)
        else:
            #flat_output = self.min_snap_update(t)
            #flat_output = self.min_snap_circle_update(t)
            #flat_output = self.min_snap_angled_circle_update(t)
            #flat_output = self.min_snap_lemniscate_update(t)
            flat_output = self.min_snap_traj_update(t)
        return flat_output

    def min_snap_traj_update(self, t):
        #lemniscate params
        #w_c = 1.6
        #p = 2.5
        #theta = 0
        #shape = 'lemniscate'

        #circle params
        w_c = 3
        p = 1
        theta = 0
        shape = 'circle'

        #start trajectory
        if self.i == 0 and t == np.inf: #first inf - pass end location
            self.x = self.end
            self.i = self.i + 1
        elif self.i == 1 and t == np.inf: #second inf - pass end yaw
            self.yaw = self.yaw
            self.i = self.i + 1
        else:
            if self.i == 2: #reset starting location after np.inf passes
                self.x = self.start
                self.i = self.i + 1
            else:
                pos_array, vel_array, acc_array, jerk_array = shape_traj.get_traj(self.start, p, w_c, np.radians(theta), t, shape)
                self.x = pos_array
                self.x_dot = vel_array
                self.x_ddot = acc_array
                self.x_dddot = jerk_array

        flat_output = {'x': self.x, 'x_dot': self.x_dot, 'x_ddot': self.x_ddot, 'x_dddot': self.x_dddot,
                       'x_ddddot': self.x_ddddot, 'yaw': self.yaw, 'yaw_dot': self.yaw_dot}

        return flat_output

    def min_snap_lemniscate_update(self, t):
        w_c = 1.24

        if self.i == 0 and t == np.inf: #first inf - pass end location
            self.x = self.end
            self.i = self.i + 1
        elif self.i == 1 and t == np.inf: #second inf - pass end yaw
            self.yaw = self.yaw
            self.i = self.i + 1

        # check if quadrotor has reached final location
        else:
            if self.i == 2: #reset starting location after np.inf passes
                start = [0, self.start[1], 0]
                self.x = start
                self.i = self.i + 1
            elif t > self.traj_struct[0][-1]: #stop once last waypoint reached
                self.x = np.array([])
                self.x_dot = np.array([])
                self.x_ddot = np.array([])
                self.x_dddot = np.array([])

                T = self.traj_struct[0][-1] - self.traj_struct[0][-2]

                pos = self.traj_struct[1][0+8*(self.num_segments-1):8+8*(self.num_segments-1)]
                vel = [0, 7*pos[0], 6*pos[1], 5*pos[2], 4*pos[3], 3*pos[4], 2*pos[5], pos[6]]
                acc = [0, 0, 42*pos[0], 30*pos[1], 20*pos[2], 12*pos[3], 6*pos[4], 2*pos[5]]
                jerk = [0, 0, 0, 210*pos[0], 120*pos[1], 60*pos[2], 24*pos[3], 6*pos[4]]

                #p = np.polyval(np.array(pos), T) #pos
                #p_dot = np.polyval(np.array(vel), T) #vel
                #p_ddot = np.polyval(np.array(acc), T) #acc
                #p_dddot = np.polyval(np.array(jerk), T) #jerk

                #constant polynomial
                p = 4
                p_dot, p_ddot, p_dddot = 0, 0, 0

                x_pos = p*cos(w_c*t)/(sin(w_c*t)**2 + 1)
                x_vel = -w_c*p*sin(t*w_c)/(sin(t*w_c)**2 + 1) - 2*w_c*p*sin(t*w_c)*cos(t*w_c)**2/(sin(t*w_c)**2 + 1)**2 + p_dot*cos(t*w_c)/(sin(t*w_c)**2 + 1)
                x_acc = 0
                x_jerk = 0
                #x_acc = (-w_c**2*p*cos(t*w_c) + 2*w_c**2*(sin(t*w_c)**2 - cos(t*w_c)**2 + 4*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1))*p*cos(t*w_c)/(sin(t*w_c)**2 + 1) + 4*w_c**2*p*sin(t*w_c)**2*cos(t*w_c)/(sin(t*w_c)**2 + 1) - 2*w_c*p_dot*sin(t*w_c) - 4*w_c*p_dot*sin(t*w_c)*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) + 2*p_ddot*cos(t*w_c))/(sin(t*w_c)**2 + 1)
                #x_jerk = (w_c**3*p*sin(t*w_c) - 6*w_c**3*(sin(t*w_c)**2 - cos(t*w_c)**2 + 4*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1))*p*sin(t*w_c)/(sin(t*w_c)**2 + 1) + 8*w_c**3*(1 - 3*sin(t*w_c)**2/(sin(t*w_c)**2 + 1) + 3*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) - 6*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1)**2)*p*sin(t*w_c)*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) + 6*w_c**3*p*sin(t*w_c)*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) - 3*w_c**2*p_dot*cos(t*w_c) + 6*w_c**2*(sin(t*w_c)**2 - cos(t*w_c)**2 + 4*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1))*p_dot*cos(t*w_c)/(sin(t*w_c)**2 + 1) + 12*w_c**2*p_dot*sin(t*w_c)**2*cos(t*w_c)/(sin(t*w_c)**2 + 1) - 6*w_c*p_ddot*sin(t*w_c) - 12*w_c*p_ddot*sin(t*w_c)*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) + 6*p_dddot*cos(t*w_c))/(sin(t*w_c)**2 + 1)

                z_pos = p*cos(t)*sin(t)/(1+sin(t)**2)
                z_vel = -w_c*p*sin(t*w_c)**2/(sin(t*w_c)**2 + 1) + w_c*p*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) - 2*w_c*p*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1)**2 + p_dot*sin(t*w_c)*cos(t*w_c)/(sin(t*w_c)**2 + 1)
                z_acc = 0
                z_jerk = 0
                #z_acc = 2*(-2*w_c**2*p*sin(t*w_c)*cos(t*w_c) + w_c**2*(sin(t*w_c)**2 - cos(t*w_c)**2 + 4*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1))*p*sin(t*w_c)*cos(t*w_c)/(sin(t*w_c)**2 + 1) + 2*w_c**2*p*sin(t*w_c)**3*cos(t*w_c)/(sin(t*w_c)**2 + 1) - 2*w_c**2*p*sin(t*w_c)*cos(t*w_c)**3/(sin(t*w_c)**2 + 1) - w_c*p_dot*sin(t*w_c)**2 + w_c*p_dot*cos(t*w_c)**2 - 2*w_c*p_dot*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) + p_ddot*sin(t*w_c)*cos(t*w_c))/(sin(t*w_c)**2 + 1)
                #z_jerk = 2*(2*w_c**3*p*sin(t*w_c)**2 - 2*w_c**3*p*cos(t*w_c)**2 - 3*w_c**3*(sin(t*w_c)**2 - cos(t*w_c)**2 + 4*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1))*(t**7 + t**6 + t**5 + t**4 + t**3 + t**2 + t + 1)*sin(t*w_c)**2/(sin(t*w_c)**2 + 1) + 3*w_c**3*(sin(t*w_c)**2 - cos(t*w_c)**2 + 4*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1))*p*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) + 4*w_c**3*(1 - 3*sin(t*w_c)**2/(sin(t*w_c)**2 + 1) + 3*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) - 6*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1)**2)*p*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) + 12*w_c**3*p*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) - 6*w_c**2*p_dot*sin(t*w_c)*cos(t*w_c) + 3*w_c**2*(sin(t*w_c)**2 - cos(t*w_c)**2 + 4*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1))*p_dot*sin(t*w_c)*cos(t*w_c)/(sin(t*w_c)**2 + 1) + 6*w_c**2*p_dot*sin(t*w_c)**3*cos(t*w_c)/(sin(t*w_c)**2 + 1) - 6*w_c**2*p_dot*sin(t*w_c)*cos(t*w_c)**3/(sin(t*w_c)**2 + 1) - 3*w_c*p_ddot*sin(t*w_c)**2 + 3*w_c*p_ddot*cos(t*w_c)**2 - 6*w_c*p_ddot*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) + 3*p_ddot*sin(t*w_c)*cos(t*w_c))/(sin(t*w_c)**2 + 1)

                y_pos = self.start[1]
                y_vel, y_acc, y_jerk = 0, 0, 0

                pos_array = np.array([x_pos, y_pos, z_pos])
                vel_array = np.array([x_vel, y_vel, z_vel])
                acc_array = np.array([x_acc, y_acc, z_acc])
                jerk_array = np.array([x_jerk, y_jerk, z_jerk])

                self.x = np.append(self.x, pos_array)
                self.x_dot = np.append(self.x_dot, vel_array)
                self.x_ddot = np.append(self.x_ddot, acc_array)
                self.x_dddot = np.append(self.x_dddot, jerk_array)
            else:
                self.x = np.array([])
                self.x_dot = np.array([])
                self.x_ddot = np.array([])
                self.x_dddot = np.array([])

                for j in range(self.num_segments):
                    if t < self.traj_struct[0][j]: #while within time structure
                        T = t

                        if j != 0: #get time segment time, not cumulative time
                            T = t - self.traj_struct[0][j-1]

                        #get x pos/vel/acc/jerk (don't care about y, z for circles)
                        pos = self.traj_struct[1][0+8*j:8+8*j]
                        vel = [0, 7*pos[0], 6*pos[1], 5*pos[2], 4*pos[3], 3*pos[4], 2*pos[5], pos[6]]
                        acc = [0, 0, 42*pos[0], 30*pos[1], 20*pos[2], 12*pos[3], 6*pos[4], 2*pos[5]]
                        jerk = [0, 0, 0, 210*pos[0], 120*pos[1], 60*pos[2], 24*pos[3], 6*pos[4]]

                        #p = np.polyval(np.array(pos), T) #pos
                        #p_dot = np.polyval(np.array(vel), T) #vel
                        #p_ddot = np.polyval(np.array(acc), T) #acc
                        #p_dddot = np.polyval(np.array(jerk), T) #jerk

                        #constant polynomial
                        p = 4
                        p_dot, p_ddot, p_dddot = 0, 0, 0

                        x_pos = p*cos(w_c*t)/(sin(w_c*t)**2 + 1)
                        x_vel = -w_c*p*sin(t*w_c)/(sin(t*w_c)**2 + 1) - 2*w_c*p*sin(t*w_c)*cos(t*w_c)**2/(sin(t*w_c)**2 + 1)**2 + p_dot*cos(t*w_c)/(sin(t*w_c)**2 + 1)
                        x_acc = (-w_c**2*p*cos(t*w_c) + 2*w_c**2*(sin(t*w_c)**2 - cos(t*w_c)**2 + 4*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1))*p*cos(t*w_c)/(sin(t*w_c)**2 + 1) + 4*w_c**2*p*sin(t*w_c)**2*cos(t*w_c)/(sin(t*w_c)**2 + 1) - 2*w_c*p_dot*sin(t*w_c) - 4*w_c*p_dot*sin(t*w_c)*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) + 2*p_ddot*cos(t*w_c))/(sin(t*w_c)**2 + 1)
                        x_jerk = (w_c**3*p*sin(t*w_c) - 6*w_c**3*(sin(t*w_c)**2 - cos(t*w_c)**2 + 4*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1))*p*sin(t*w_c)/(sin(t*w_c)**2 + 1) + 8*w_c**3*(1 - 3*sin(t*w_c)**2/(sin(t*w_c)**2 + 1) + 3*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) - 6*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1)**2)*p*sin(t*w_c)*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) + 6*w_c**3*p*sin(t*w_c)*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) - 3*w_c**2*p_dot*cos(t*w_c) + 6*w_c**2*(sin(t*w_c)**2 - cos(t*w_c)**2 + 4*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1))*p_dot*cos(t*w_c)/(sin(t*w_c)**2 + 1) + 12*w_c**2*p_dot*sin(t*w_c)**2*cos(t*w_c)/(sin(t*w_c)**2 + 1) - 6*w_c*p_ddot*sin(t*w_c) - 12*w_c*p_ddot*sin(t*w_c)*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) + 6*p_dddot*cos(t*w_c))/(sin(t*w_c)**2 + 1)

                        #x_vel = p_dot*cos(t)/(sin(t)**2 + 1) - p*sin(t)/(sin(t)**2 + 1) - 2*p*sin(t)*cos(t)**2/(sin(t)**2 + 1)**2
                        #x_acc = (2*p_ddot*cos(t) - 2*p_dot*sin(t) - p*cos(t) + 2*(sin(t)**2 - cos(t)**2 + 4*sin(t)**2*cos(t)**2/(sin(t)**2 + 1))*p*cos(t)/(sin(t)**2 + 1) - 4*p_dot*sin(t)*cos(t)**2/(sin(t)**2 + 1) + 4*p*sin(t)**2*cos(t)/(sin(t)**2 + 1))/(sin(t)**2 + 1)
                        #x_jerk = (6*p_dddot*cos(t) - 6*p_ddot*sin(t) - 3*p_dot*cos(t) + p*sin(t) + 6*(sin(t)**2 - cos(t)**2 + 4*sin(t)**2*cos(t)**2/(sin(t)**2 + 1))*p_dot*cos(t)/(sin(t)**2 + 1) - 6*(sin(t)**2 - cos(t)**2 + 4*sin(t)**2*cos(t)**2/(sin(t)**2 + 1))*p*sin(t)/(sin(t)**2 + 1) + 8*(1 - 3*sin(t)**2/(sin(t)**2 + 1) + 3*cos(t)**2/(sin(t)**2 + 1) - 6*sin(t)**2*cos(t)**2/(sin(t)**2 + 1)**2)*p*sin(t)*cos(t)**2/(sin(t)**2 + 1) - 12*p_ddot*sin(t)*cos(t)**2/(sin(t)**2 + 1) + 12*p_dot*sin(t)**2*cos(t)/(sin(t)**2 + 1) + 6*p*sin(t)*cos(t)**2/(sin(t)**2 + 1))/(sin(t)**2 + 1)

                        z_pos = p*cos(w_c*t)*sin(w_c*t)/(1+sin(w_c*t)**2)
                        z_vel = -w_c*p*sin(t*w_c)**2/(sin(t*w_c)**2 + 1) + w_c*p*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) - 2*w_c*p*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1)**2 + p_dot*sin(t*w_c)*cos(t*w_c)/(sin(t*w_c)**2 + 1)
                        z_acc = 2*(-2*w_c**2*p*sin(t*w_c)*cos(t*w_c) + w_c**2*(sin(t*w_c)**2 - cos(t*w_c)**2 + 4*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1))*p*sin(t*w_c)*cos(t*w_c)/(sin(t*w_c)**2 + 1) + 2*w_c**2*p*sin(t*w_c)**3*cos(t*w_c)/(sin(t*w_c)**2 + 1) - 2*w_c**2*p*sin(t*w_c)*cos(t*w_c)**3/(sin(t*w_c)**2 + 1) - w_c*p_dot*sin(t*w_c)**2 + w_c*p_dot*cos(t*w_c)**2 - 2*w_c*p_dot*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) + p_ddot*sin(t*w_c)*cos(t*w_c))/(sin(t*w_c)**2 + 1)
                        z_jerk = 2*(2*w_c**3*p*sin(t*w_c)**2 - 2*w_c**3*p*cos(t*w_c)**2 - 3*w_c**3*(sin(t*w_c)**2 - cos(t*w_c)**2 + 4*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1))*p*sin(t*w_c)**2/(sin(t*w_c)**2 + 1) + 3*w_c**3*(sin(t*w_c)**2 - cos(t*w_c)**2 + 4*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1))*p*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) + 4*w_c**3*(1 - 3*sin(t*w_c)**2/(sin(t*w_c)**2 + 1) + 3*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) - 6*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1)**2)*p*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) + 12*w_c**3*p*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) - 6*w_c**2*p_dot*sin(t*w_c)*cos(t*w_c) + 3*w_c**2*(sin(t*w_c)**2 - cos(t*w_c)**2 + 4*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1))*p_dot*sin(t*w_c)*cos(t*w_c)/(sin(t*w_c)**2 + 1) + 6*w_c**2*p_dot*sin(t*w_c)**3*cos(t*w_c)/(sin(t*w_c)**2 + 1) - 6*w_c**2*p_dot*sin(t*w_c)*cos(t*w_c)**3/(sin(t*w_c)**2 + 1) - 3*w_c*p_ddot*sin(t*w_c)**2 + 3*w_c*p_ddot*cos(t*w_c)**2 - 6*w_c*p_ddot*sin(t*w_c)**2*cos(t*w_c)**2/(sin(t*w_c)**2 + 1) + 3*p_ddot*sin(t*w_c)*cos(t*w_c))/(sin(t*w_c)**2 + 1)

                        #z_vel = p_dot*sin(t)*cos(t)/(sin(t)**2 + 1) - p*sin(t)**2/(sin(t)**2 + 1) + p*cos(t)**2/(sin(t)**2 + 1) - 2*p*sin(t)**2*cos(t)**2/(sin(t)**2 + 1)**2
                        #z_acc = 2*(p_ddot*sin(t)*cos(t) - p_dot*sin(t)**2 + p_dot*cos(t)**2 - 2*p*sin(t)*cos(t) + (sin(t)**2 - cos(t)**2 + 4*sin(t)**2*cos(t)**2/(sin(t)**2 + 1))*p*sin(t)*cos(t)/(sin(t)**2 + 1) - 2*p_dot*sin(t)**2*cos(t)**2/(sin(t)**2 + 1) + 2*p*sin(t)**3*cos(t)/(sin(t)**2 + 1) - 2*p*sin(t)*cos(t)**3/(sin(t)**2 + 1))/(sin(t)**2 + 1)
                        #z_jerk = 2*(3*p_dddot*sin(t)*cos(t) - 3*p_ddot*sin(t)**2 + 3*p_ddot*cos(t)**2 - 6*p_dot*sin(t)*cos(t) + 2*p*sin(t)**2 - 2*p*cos(t)**2 + 3*(sin(t)**2 - cos(t)**2 + 4*sin(t)**2*cos(t)**2/(sin(t)**2 + 1))*p_dot*sin(t)*cos(t)/(sin(t)**2 + 1) - 3*(sin(t)**2 - cos(t)**2 + 4*sin(t)**2*cos(t)**2/(sin(t)**2 + 1))*p*sin(t)**2/(sin(t)**2 + 1) + 3*(sin(t)**2 - cos(t)**2 + 4*sin(t)**2*cos(t)**2/(sin(t)**2 + 1))*p*cos(t)**2/(sin(t)**2 + 1) + 4*(1 - 3*sin(t)**2/(sin(t)**2 + 1) + 3*cos(t)**2/(sin(t)**2 + 1) - 6*sin(t)**2*cos(t)**2/(sin(t)**2 + 1)**2)*p*sin(t)**2*cos(t)**2/(sin(t)**2 + 1) - 6*p_ddot*sin(t)**2*cos(t)**2/(sin(t)**2 + 1) + 6*p_dot*sin(t)**3*cos(t)/(sin(t)**2 + 1) - 6*p_dot*sin(t)*cos(t)**3/(sin(t)**2 + 1) + 12*p*sin(t)**2*cos(t)**2/(sin(t)**2 + 1))/(sin(t)**2 + 1)

                        y_pos = self.start[1]
                        y_vel, y_acc, y_jerk = 0, 0, 0

                        pos_array = np.array([x_pos, y_pos, z_pos])
                        vel_array = np.array([x_vel, y_vel, z_vel])
                        acc_array = np.array([x_acc, y_acc, z_acc])
                        jerk_array = np.array([x_jerk, y_jerk, z_jerk])

                        self.x = np.append(self.x, pos_array)
                        self.x_dot = np.append(self.x_dot, vel_array)
                        self.x_ddot = np.append(self.x_ddot, acc_array)
                        self.x_dddot = np.append(self.x_dddot, jerk_array)
                        break

                #once we're past min snap denoted time


        flat_output = {'x': self.x, 'x_dot': self.x_dot, 'x_ddot': self.x_ddot, 'x_dddot': self.x_dddot,
                       'x_ddddot': self.x_ddddot, 'yaw': self.yaw, 'yaw_dot': self.yaw_dot}

        return flat_output

    def min_snap_update(self, t):
        if self.i == 0 and t == np.inf: #first inf - pass end location
            self.x = self.end
            self.i = self.i + 1
        elif self.i == 1 and t == np.inf: #second inf - pass end yaw
            self.yaw = self.yaw
            self.i = self.i + 1

        # check if quadrotor has reached final location
        else:
            if self.i == 2: #reset starting location after np.inf passes
                self.x = self.start
                self.i = self.i + 1
            elif t > self.traj_struct[0][-1]: #stop once last waypoint reached
                self.x_dot = float(0) * self.x_dot
                self.x = self.end
                self.t_start = t
            else:
                #print(t)
                #for j in range(len(self.traj_struct[0])-1): #per waypoint
                for j in range(self.num_segments):
                    if t < self.traj_struct[0][j]:
                        self.x = np.array([])
                        self.x_dot = np.array([])
                        self.x_ddot = np.array([])
                        self.x_dddot = np.array([])

                        T = t
                        if j != 0: #get time segment time, not cumulative time
                            T = t - self.traj_struct[0][j-1]
                        for i in range(1,4): #loop to build x(1),y(2),z(3) components
                            pos = self.traj_struct[i][0+8*j:8+8*j]
                            vel = [0, 7*pos[0], 6*pos[1], 5*pos[2], 4*pos[3], 3*pos[4], 2*pos[5], pos[6]]
                            acc = [0, 0, 42*pos[0], 30*pos[1], 20*pos[2], 12*pos[3], 6*pos[4], 2*pos[5]]
                            jerk = [0, 0, 0, 210*pos[0], 120*pos[1], 60*pos[2], 24*pos[3], 6*pos[4]]

                            self.x = np.append(self.x, np.polyval(np.array(pos), T))
                            self.x_dot = np.append(self.x_dot, np.polyval(np.array(vel), T))
                            self.x_ddot = np.append(self.x_ddot, np.polyval(np.array(acc), T))
                            self.x_dddot = np.append(self.x_dddot, np.polyval(np.array(jerk), T))
                        break
        flat_output = {'x': self.x, 'x_dot': self.x_dot, 'x_ddot': self.x_ddot, 'x_dddot': self.x_dddot,
                       'x_ddddot': self.x_ddddot, 'yaw': self.yaw, 'yaw_dot': self.yaw_dot}
        return flat_output

    def naive_update(self, t):
        if self.i == 0 and t == np.inf: #first inf - pass end location
            self.x = self.end
            self.i = self.i + 1
        elif self.i == 1 and t == np.inf: #second inf - pass end yaw
            self.yaw = self.yaw
            self.i = self.i + 1

        # check if quadrotor has reached final location
        else:
            if self.i == 2:
                self.x = self.start
                self.i = self.i + 1

            if t > self.traj_struct[-1][2]: #stop once last waypoint reached
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
                       'x_ddddot': self.x_ddddot, 'yaw': self.yaw, 'yaw_dot': self.yaw_dot}
        # print('flat output: ', flat_output)
        return flat_output

    def min_snap_circle_update(self, t):
        w_c = 3

        if self.i == 0 and t == np.inf: #first inf - pass end location
            self.x = self.end
            self.i = self.i + 1
        elif self.i == 1 and t == np.inf: #second inf - pass end yaw
            self.yaw = self.yaw
            self.i = self.i + 1

        # check if quadrotor has reached final location
        else:
            if self.i == 2: #reset starting location after np.inf passes
                self.x = self.start
                self.i = self.i + 1
            elif t > self.traj_struct[0][-1]: #stop once last waypoint reached
                self.x = np.array([])
                self.x_dot = np.array([])
                self.x_ddot = np.array([])
                self.x_dddot = np.array([])

                T = self.traj_struct[0][-1] - self.traj_struct[0][-2]

                pos = self.traj_struct[1][0+8*(self.num_segments-1):8+8*(self.num_segments-1)]
                vel = [0, 7*pos[0], 6*pos[1], 5*pos[2], 4*pos[3], 3*pos[4], 2*pos[5], pos[6]]
                acc = [0, 0, 42*pos[0], 30*pos[1], 20*pos[2], 12*pos[3], 6*pos[4], 2*pos[5]]
                jerk = [0, 0, 0, 210*pos[0], 120*pos[1], 60*pos[2], 24*pos[3], 6*pos[4]]

                #p = np.polyval(np.array(pos), T) #pos
                #p_dot = np.polyval(np.array(vel), T) #vel
                #p_ddot = np.polyval(np.array(acc), T) #acc
                #p_dddot = np.polyval(np.array(jerk), T) #jerk

                p = 1
                p_dot = 0
                p_ddot = 0
                p_dddot = 0

                #position
                x_pos = p*np.cos(w_c*t)
                z_pos = p*np.sin(w_c*t)
                y_pos = self.start[1]
                #velocity
                x_vel = p_dot*np.cos(w_c*t) - p*w_c*np.sin(w_c*t)
                z_vel = p_dot*np.sin(w_c*t) + p*w_c*np.cos(w_c*t)
                y_vel = 0
                #acceleration
                x_acc = p_ddot*np.cos(w_c*t) - 2*w_c*p_dot*np.sin(w_c*t) - w_c**2*p*np.cos(w_c*t)
                z_acc = p_ddot*np.sin(w_c*t) + 2*w_c*p_dot*np.cos(w_c*t) - w_c**2*p*np.sin(w_c*t)
                y_acc = 0
                #jerk
                x_jerk = (p_dddot - 3*w_c**2*x_vel)*np.cos(w_c*t) - (3*w_c*p_ddot - w_c**3*p)*np.sin(w_c*t)
                z_jerk = (p_dddot - 3*w_c**2*x_vel)*np.sin(w_c*t) + (3*w_c*p_ddot - w_c**3*p)*np.cos(w_c*t)
                y_jerk = 0

                pos_array = np.array([x_pos, y_pos, z_pos])
                vel_array = np.array([x_vel, y_vel, z_vel])
                acc_array = np.array([x_acc, y_acc, z_acc])
                jerk_array = np.array([x_jerk, y_jerk, z_jerk])

                self.x = np.append(self.x, pos_array)
                self.x_dot = np.append(self.x_dot, vel_array)
                self.x_ddot = np.append(self.x_ddot, acc_array)
                self.x_dddot = np.append(self.x_dddot, jerk_array)
            else:
                #print(t)
                #for j in range(len(self.traj_struct[0])-1): #per waypoint
                for j in range(self.num_segments):
                    if t < self.traj_struct[0][j]:
                        self.x = np.array([])
                        self.x_dot = np.array([])
                        self.x_ddot = np.array([])
                        self.x_dddot = np.array([])

                        T = t
                        if j != 0: #get time segment time, not cumulative time
                            T = t - self.traj_struct[0][j-1]

                        #get x pos/vel/acc/jerk (don't care about y, z for circles)
                        pos = self.traj_struct[1][0+8*j:8+8*j]
                        #pos = [0, 0, 0, 0, 1, 2, 3, self.traj_struct[1][8]]
                        vel = [0, 7*pos[0], 6*pos[1], 5*pos[2], 4*pos[3], 3*pos[4], 2*pos[5], pos[6]]
                        acc = [0, 0, 42*pos[0], 30*pos[1], 20*pos[2], 12*pos[3], 6*pos[4], 2*pos[5]]
                        jerk = [0, 0, 0, 210*pos[0], 120*pos[1], 60*pos[2], 24*pos[3], 6*pos[4]]

                        #p = np.polyval(np.array(pos), T) #pos
                        #p_dot = np.polyval(np.array(vel), T) #vel
                        #p_ddot = np.polyval(np.array(acc), T) #acc
                        #p_dddot = np.polyval(np.array(jerk), T) #jerk

                        p = 1
                        p_dot = 0
                        p_ddot = 0
                        p_dddot = 0

                        #position
                        x_pos = p*np.cos(w_c*t)
                        z_pos = p*np.sin(w_c*t)
                        y_pos = self.start[1]
                        #velocity
                        x_vel = p_dot*np.cos(w_c*t) - p*w_c*np.sin(w_c*t)
                        z_vel = p_dot*np.sin(w_c*t) + p*w_c*np.cos(w_c*t)
                        y_vel = 0
                        #acceleration
                        x_acc = p_ddot*np.cos(w_c*t) - 2*w_c*p_dot*np.sin(w_c*t) - w_c**2*p*np.cos(w_c*t)
                        z_acc = p_ddot*np.sin(w_c*t) + 2*w_c*p_dot*np.cos(w_c*t) - w_c**2*p*np.sin(w_c*t)
                        y_acc = 0
                        #jerk
                        x_jerk = (p_dddot - 3*w_c**2*x_vel)*np.cos(w_c*t) - (3*w_c*p_ddot - w_c**3*p)*np.sin(w_c*t)
                        z_jerk = (p_dddot - 3*w_c**2*x_vel)*np.sin(w_c*t) + (3*w_c*p_ddot - w_c**3*p)*np.cos(w_c*t)
                        y_jerk = 0

                        pos_array = np.array([x_pos, y_pos, z_pos])
                        vel_array = np.array([x_vel, y_vel, z_vel])
                        acc_array = np.array([x_acc, y_acc, z_acc])
                        jerk_array = np.array([x_jerk, y_jerk, z_jerk])

                        self.x = np.append(self.x, pos_array)
                        self.x_dot = np.append(self.x_dot, vel_array)
                        self.x_ddot = np.append(self.x_ddot, acc_array)
                        self.x_dddot = np.append(self.x_dddot, jerk_array)
                        break
        flat_output = {'x': self.x, 'x_dot': self.x_dot, 'x_ddot': self.x_ddot, 'x_dddot': self.x_dddot,
                       'x_ddddot': self.x_ddddot, 'yaw': self.yaw, 'yaw_dot': self.yaw_dot}

        return flat_output

    def min_snap_angled_circle_update(self, t):
        w_c = 3
        theta = 70
        p = 1
        p_dot = 0
        p_ddot = 0
        p_dddot = 0

        if self.i == 0 and t == np.inf: #first inf - pass end location
            self.x = self.end
            self.i = self.i + 1
        elif self.i == 1 and t == np.inf: #second inf - pass end yaw
            self.yaw = self.yaw
            self.i = self.i + 1

        # check if quadrotor has reached final location
        else:
            if self.i == 2: #reset starting location after np.inf passes

                start = [-np.cos(theta)*self.start[0], self.start[1], np.sin(theta)*self.start[2]]

                self.start = [-np.cos(theta) * p * np.cos(w_c * 0),
                             -1 * (p * np.sin(w_c * 0)) + self.start[1],
                             np.sin(theta) * p * np.cos(w_c * 0)]

                self.x = self.start
                self.i = self.i + 1
            elif t > self.traj_struct[0][-1]: #stop once last waypoint reached
                self.x = np.array([])
                self.x_dot = np.array([])
                self.x_ddot = np.array([])
                self.x_dddot = np.array([])

                T = self.traj_struct[0][-1] - self.traj_struct[0][-2]

                pos = self.traj_struct[1][0 + 8 * (self.num_segments - 1):8 + 8 * (self.num_segments - 1)]
                vel = [0, 7 * pos[0], 6 * pos[1], 5 * pos[2], 4 * pos[3], 3 * pos[4], 2 * pos[5], pos[6]]
                acc = [0, 0, 42 * pos[0], 30 * pos[1], 20 * pos[2], 12 * pos[3], 6 * pos[4], 2 * pos[5]]
                jerk = [0, 0, 0, 210 * pos[0], 120 * pos[1], 60 * pos[2], 24 * pos[3], 6 * pos[4]]

                # p = np.polyval(np.array(pos), T) #pos
                # p_dot = np.polyval(np.array(vel), T) #vel
                # p_ddot = np.polyval(np.array(acc), T) #acc
                # p_dddot = np.polyval(np.array(jerk), T) #jerk

                # position
                x_pos = -np.cos(theta)*p*np.cos(w_c*t)
                y_pos = -1*(p*np.sin(w_c*t)) + self.start[1]
                z_pos = np.sin(theta)*p*np.cos(w_c*t)
                # print('x_poly', x_poly)
                # print('x_pos', x_pos)
                # print("T", T)
                # print('t', t)
                # velocity
                x_vel = -np.cos(theta) * (p_dot * np.cos(w_c * t) - p * w_c * np.sin(w_c * t))
                y_vel = -1 * (p_dot * np.sin(w_c * t) + p * w_c * np.cos(w_c * t))
                z_vel = np.sin(theta) * p_dot * np.cos(w_c * t) - p * w_c * np.sin(w_c * t)
                # acceleration
                x_acc = -np.cos(theta) * (p_ddot * np.cos(w_c * t) - 2 * w_c * p_dot * np.sin(w_c * t) - w_c ** 2 * p * np.cos(w_c * t))
                y_acc = -1 * (p_ddot * np.sin(w_c * t) + 2 * w_c * p_dot * np.cos(w_c * t) - w_c ** 2 * p * np.sin(w_c * t))
                z_acc = np.sin(theta) * (p_ddot * np.cos(w_c * t) - 2 * w_c * p_dot * np.sin(w_c * t) - w_c ** 2 * p * np.cos(w_c * t))
                # jerk
                x_jerk = -np.cos(theta) * ((p_dddot - 3 * w_c ** 2 * x_vel) * np.cos(w_c * t) - (
                            3 * w_c * p_ddot - w_c ** 3 * p) * np.sin(w_c * t))
                y_jerk = -1 * ((p_dddot - 3 * w_c ** 2 * x_vel) * np.sin(w_c * t) + (
                            3 * w_c * p_ddot - w_c ** 3 * p) * np.cos(w_c * t))
                z_jerk = np.sin(theta) * ((p_dddot - 3 * w_c ** 2 * x_vel) * np.cos(w_c * t) - (
                            3 * w_c * p_ddot - w_c ** 3 * p) * np.sin(w_c * t))

                pos_array = np.array([x_pos, y_pos, z_pos])
                vel_array = np.array([x_vel, y_vel, z_vel])
                acc_array = np.array([x_acc, y_acc, z_acc])
                jerk_array = np.array([x_jerk, y_jerk, z_jerk])

                self.x = np.append(self.x, pos_array)
                self.x_dot = np.append(self.x_dot, vel_array)
                self.x_ddot = np.append(self.x_ddot, acc_array)
                self.x_dddot = np.append(self.x_dddot, jerk_array)
            else:
                #print(t)
                #for j in range(len(self.traj_struct[0])-1): #per waypoint
                for j in range(self.num_segments):
                    if t < self.traj_struct[0][j]:
                        self.x = np.array([])
                        self.x_dot = np.array([])
                        self.x_ddot = np.array([])
                        self.x_dddot = np.array([])

                        T = t
                        if j != 0: #get time segment time, not cumulative time
                            T = t - self.traj_struct[0][j-1]

                        #get x pos/vel/acc/jerk (don't care about y, z for circles)
                        pos = self.traj_struct[1][0+8*j:8+8*j]
                        #pos = [0, 0, 0, 0, 1, 2, 3, self.traj_struct[1][8]]
                        vel = [0, 7*pos[0], 6*pos[1], 5*pos[2], 4*pos[3], 3*pos[4], 2*pos[5], pos[6]]
                        acc = [0, 0, 42*pos[0], 30*pos[1], 20*pos[2], 12*pos[3], 6*pos[4], 2*pos[5]]
                        jerk = [0, 0, 0, 210*pos[0], 120*pos[1], 60*pos[2], 24*pos[3], 6*pos[4]]

                        #p = np.polyval(np.array(pos), T) #pos
                        #p_dot = np.polyval(np.array(vel), T) #vel
                        #p_ddot = np.polyval(np.array(acc), T) #acc
                        #p_dddot = np.polyval(np.array(jerk), T) #jerk

                        #position
                        x_pos = -np.cos(theta)*p*np.cos(w_c*t)
                        y_pos = -1*(p*np.sin(w_c*t)) + self.start[1]
                        z_pos = np.sin(theta)*p*np.cos(w_c*t)
                        #print('x_poly', x_poly)
                        #print('x_pos', x_pos)
                        #print("T", T)
                        #print('t', t)
                        #velocity
                        x_vel = -np.cos(theta)*(p_dot*np.cos(w_c*t) - p*w_c*np.sin(w_c*t))
                        y_vel = -1*(p_dot*np.sin(w_c*t) + p*w_c*np.cos(w_c*t))
                        z_vel = np.sin(theta)*p_dot*np.cos(w_c*t) - p*w_c*np.sin(w_c*t)
                        #acceleration
                        x_acc = -np.cos(theta)*(p_ddot*np.cos(w_c*t) - 2*w_c*p_dot*np.sin(w_c*t) - w_c**2*p*np.cos(w_c*t))
                        y_acc = -1*(p_ddot*np.sin(w_c*t) + 2*w_c*p_dot*np.cos(w_c*t) - w_c**2*p*np.sin(w_c*t))
                        z_acc = np.sin(theta)*(p_ddot*np.cos(w_c*t) - 2*w_c*p_dot*np.sin(w_c*t) - w_c**2*p*np.cos(w_c*t))
                        #jerk
                        x_jerk = -np.cos(theta)*((p_dddot - 3*w_c**2*x_vel)*np.cos(w_c*t) - (3*w_c*p_ddot - w_c**3*p)*np.sin(w_c*t))
                        y_jerk = -1*((p_dddot - 3*w_c**2*x_vel)*np.sin(w_c*t) + (3*w_c*p_ddot - w_c**3*p)*np.cos(w_c*t))
                        z_jerk = np.sin(theta)*((p_dddot - 3*w_c**2*x_vel)*np.cos(w_c*t) - (3*w_c*p_ddot - w_c**3*p)*np.sin(w_c*t))

                        pos_array = np.array([x_pos, y_pos, z_pos])
                        vel_array = np.array([x_vel, y_vel, z_vel])
                        acc_array = np.array([x_acc, y_acc, z_acc])
                        jerk_array = np.array([x_jerk, y_jerk, z_jerk])

                        self.x = np.append(self.x, pos_array)
                        self.x_dot = np.append(self.x_dot, vel_array)
                        self.x_ddot = np.append(self.x_ddot, acc_array)
                        self.x_dddot = np.append(self.x_dddot, jerk_array)
                        break
        flat_output = {'x': self.x, 'x_dot': self.x_dot, 'x_ddot': self.x_ddot, 'x_dddot': self.x_dddot,
                       'x_ddddot': self.x_ddddot, 'yaw': self.yaw, 'yaw_dot': self.yaw_dot}

        return flat_output
