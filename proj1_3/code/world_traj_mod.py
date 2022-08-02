import numpy as np
import csv
import scipy
import cvxpy as cp
from .graph_search import graph_search
import copy
from rdp import rdp
from sympy import cos, sin, diff, symbols
from .min_snap import MinSnap
from .narrow_window_min_snap import MinSnapNW
import shape_traj

class WorldTrajMod(object):
    """

    """

    def __init__(self, world, start, cons):
        self.vel = 2 # m/s - self selected
        self.max_vel = 5 #m/s,  #use 3.4 for naive
        self.dist_threshold = 1 #m
        self.x_dot = np.zeros((3,))

        self.x_ddot = np.zeros((3,))
        self.x_dddot = np.zeros((3,))
        self.x_ddddot = np.zeros((3,))
        self.yaw = 0
        self.yaw_dot = 0
        self.quad_o = 1
        self.k = -1
        self.naive = False
        self.num_segments = None
        self.acc_cons = None
        self.narrow_window = True

        #break out constraints
        self.points = cons[0] #position constraint
        self.o_cons = cons[1] #upright vs inverse
        if len(cons) == 3: #acceleration constraint exists
            self.acc_cons = cons[2]

        #declare start/stop
        self.start = self.points[0]  # m
        self.end = self.points[-1]  # m

        # declare starting position as first point
        self.x = self.start
        self.t_start = 0  # starting time at each update loop
        self.i = 0  # to account for the np.inf case

        if self.naive:
            self.traj_struct = self.naive_trajectory(self.points, self.start, self.x_dot, self.vel)
        elif self.narrow_window:
            my_min_snap = MinSnapNW(self.points, self.vel, self.max_vel, self.dist_threshold)
            my_min_snap.set_o_cons(self.o_cons)
            my_min_snap.set_narrow_window_waypoint(2)
            if len(cons) == 3:
                my_min_snap.set_acc_cons(self.acc_cons)
            self.traj_struct, self.num_segments, self.time_segments = my_min_snap.get_trajectory()
        else: #min snap
            my_min_snap = MinSnap(self.points, self.vel, self.max_vel, self.dist_threshold)
            my_min_snap.set_o_cons(self.o_cons)
            if len(cons) == 3:
                my_min_snap.set_acc_cons(self.acc_cons)
            self.traj_struct, self.num_segments, self.time_segments = my_min_snap.get_trajectory()

            #print(self.traj_struct)
            '''
            with open('traj_struct.csv', 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                # writing the data rows
                csvwriter.writerows(self.traj_struct)
            '''

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

    def update(self, t):
        if self.naive:
            flat_output = self.naive_update(t)
        else:
            flat_output = self.min_snap_update(t)
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
                       'x_ddddot': self.x_ddddot, 'yaw': self.yaw, 'yaw_dot': self.yaw_dot, 'quad_o': self.quad_o}
        # print('flat output: ', flat_output)
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
                self.quad_o = self.o_cons[0]
            elif t > self.traj_struct[0][-1]: #stop once last waypoint reached
                self.x_dot = float(0)*self.x_dot
                self.x_ddot = float(0)*self.x_ddot
                self.x_dddot = float(0)*self.x_dddot
                self.x = self.end
                self.t_start = t
                self.quad_o = self.o_cons[-1]
            else:
                #print(t)
                #for j in range(len(self.traj_struct[0])-1): #per waypoint
                for j in range(self.num_segments):
                    if t < self.traj_struct[0][j]:
                        self.x = np.array([])
                        self.x_dot = np.array([])
                        self.x_ddot = np.array([])
                        self.x_dddot = np.array([])
                        self.quad_o = self.traj_struct[4][j]

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
                       'x_ddddot': self.x_ddddot, 'yaw': self.yaw, 'yaw_dot': self.yaw_dot, 'quad_o': self.quad_o}
        return flat_output

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
