import numpy as np
from flightsim.crazyflie_params import quad_params
import scipy
import cvxpy as cp
from .graph_search import graph_search
import copy
from rdp import rdp
from sympy import cos, sin, diff, symbols
from .min_snap import MinSnap
import shape_traj

class WaypointTraj(object):
    def __init__(self, center, r, w_c, a, points, loops, shape):
        self.center = center #numpy array
        self.r = r #m
        self.w_c = w_c #rad/s
        self.a = np.radians(a) #angle of rotation for circle, deg
        self.points = points # number of points equidistance for one loop
        self.loops = loops # number of loops about shape
        self.des_thrust = quad_params['k_thrust']*4*(0.5*quad_params['rotor_speed_max'])**2 #desired thrust at each waypoint

        #shape parametrizations
        self.x_pos = None
        self.y_pos = None
        self.z_pos = None
        self.set_shape(shape)

    def get_contraints(self):
        pos = self.get_points()
        acc = self.get_acc_constraints(pos)
        return pos, acc

    def set_shape(self, shape):
        if shape == 'circle':
            self.x_pos = lambda theta : self.r*np.cos(theta) + self.center[0]-self.r
            self.y_pos = lambda theta : self.r*np.sin(theta)*np.sin(self.a) + self.center[1]
            self.z_pos = lambda theta : self.r*np.sin(theta)*np.cos(self.a) + self.center[2]
        elif shape == 'lemniscate':
            self.x_pos = lambda theta : self.r*np.cos(theta)/(1+np.sin(theta)**2) + self.center[0]-self.r
            self.y_pos = lambda theta : 0*theta + self.center[1]
            self.z_pos = lambda theta : self.r*np.cos(theta)*np.sin(theta)/(1+np.sin(theta)**2) + self.center[2]

    def get_points(self): #get points equidistant about shape
        pos = np.array([0, 0, 0]) #fake starting point
        for i in range(self.loops): #number of loops around shape
            for j in range(self.points):
                theta = 2*np.pi*j/self.points #angle - n/N
                point = [self.x_pos(theta), self.y_pos(theta), self.z_pos(theta)]
                pos = np.vstack([pos, point])
        #delete first point
        pos = np.delete(pos, 0, 0)
        return pos

    def get_acc_constraints(self, pos):
        acc_constraints = np.array([0, 0, 0])
        #start at rest
        for i in range(len(pos)):
            '''
            #only get the point at apex of circle
            if i % self.points  == self.points/2 or i % self.points == 0:
                #find acc constrain
                point = pos[i]
                b_3 = np.array([point[0] - self.center[0]+self.r,
                               point[1] - self.center[1],
                               point[2] - self.center[2]])
                if np.linalg.norm(b_3) == 0:
                    acc_des = -1*np.array([0, 0, 9.81])
                else:
                    b_3 = b_3/np.linalg.norm(b_3) #unit vector [a, b, c]
                    acc_des = 1/quad_params['mass']*b_3*self.des_thrust - np.array([0, 0, 9.81])
                acc_constraints = np.vstack([acc_constraints, acc_des])
            '''
            if i < np.ceil(self.points/4): #ramp up first quater turn
                des_thrust = i/np.ceil(self.points/4)*self.des_thrust
            elif i > (len(pos) -1-np.ceil(self.points/4)): #ramp down last quarter turn
                des_thrust = (self.points-1-(i%self.points))/np.ceil(self.points/4)*self.des_thrust
            else:
                des_thrust = self.des_thrust
            #find acc constrain
            point = pos[i]
            b_3 = np.array([point[0] - self.center[0]+self.r,
                           point[1] - self.center[1],
                           point[2] - self.center[2]])
            if np.linalg.norm(b_3) == 0:
                acc_des = -1*np.array([0, 0, 9.81])
            else:
                b_3 = b_3/np.linalg.norm(b_3) #unit vector [a, b, c]
                acc_des = 1/quad_params['mass']*b_3*des_thrust - np.array([0, 0, 9.81])
            acc_constraints = np.vstack([acc_constraints, acc_des])

        acc_constraints = np.delete(acc_constraints, 0, 0)
        #print(acc_constraints)
        return acc_constraints








