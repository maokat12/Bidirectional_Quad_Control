import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

class Shapes(object):
    def __init__(self, start, r, w_c, theta, t, shape):
        self.r = r #radius of trajectory
        self.r_dot = 0 #vel
        self.r_ddot = 0 #acc
        self.r_dddot = 0 #jerk
        self.w_c = w_c #angular velocity
        self.theta = theta #angle of circle
        self.shape = shape #what trajectory shape (circle, angled circle, lemniscate
        self.start = start #starting position

    def get_traj(self, t):
        if self.shape == 'circle':
            return self.circle_traj(t)
        elif self.shape == 'lemniscate':
            return self.lemniscate(t)

    def circle_traj(self, t):
        #pos
        x_pos = -np.cos(self.theta)*self.r*np.cos(self.w_c*t)
        y_pos = -1*self.r*np.sin(self.w_c*t)
        z_pos = np.sin(self.theta)*self.r*np.cos(self.w_c*t)
        #vel
        x_vel = -np.cos(self.theta)*(self.r_dot*np.cos(self.w_c*t) - self.r*self.w_c*np.sin(self.w_c*t))
        y_vel = -1*(self.r_dot*np.sin(self.w_c*t) + self.r*self.w_c*np.cos(self.w_c*t))
        z_vel = np.sin(self.theta)*self.r_dot*np.cos(self.w_c*t) - self.r*self.w_c*np.sin(self.w_c*t)
        #acc
        x_acc = np.cos(self.theta)*(self.r_ddot*np.cos(self.w_c*t) - 2*self.w_c*self.r_dot*np.sin(self.w_c*t) - self.w_c**2*self.r*np.cos(self.w_c*t))
        y_acc = -1 * (self.r_ddot * np.sin(self.w_c*t) + 2*self.w_c*self.r_dot*np.cos(self.w_c*t) - self.w_c**2*self.r*np.sin(self.w_c*t))
        z_acc = np.sin(self.theta)*(self.r_ddot*np.cos(self.w_c*t) - 2*self.w_c*self.r_dot * np.sin(self.w_c*t) - self.w_c**2*self.r*np.cos(self.w_c*t))
        #jerk
        x_jerk = -np.cos(self.theta)*((self.r_dddot-3*self.w_c**2*x_vel)*np.cos(self.w_c*t) - (3*self.w_c*self.r_ddot-self.w_c**3*self.r)*np.sin(self.w_c*t))
        y_jerk = -1*((self.r_dddot-3*self.w_c**2*x_vel)*np.sin(self.w_c*t) + (3*self.w_c*self.r_ddot-self.w_c**3*self.r)*np.cos(self.w_c*t))
        z_jerk = np.sin(self.theta)*((self.r_dddot-3*self.w_c**2*x_vel)*np.cos(self.w_c*t) - (3*self.w_c*self.r_ddot-self.w_c**3*self.r)*np.sin(self.w_c*t))

        pos_array = np.array([x_pos, y_pos, z_pos])
        vel_array = np.array([x_vel, y_vel, z_vel])
        acc_array = np.array([x_acc, y_acc, z_acc])
        jerk_array = np.array([x_jerk, y_jerk, z_jerk])

        return pos_array, vel_array, acc_array, jerk_array

    def lemniscate(self, t):
        #pos
        x_pos = self.r*np.cos(self.w_c*t)/(np.sin(self.w_c*t)**2 + 1)
        y_pos = 0
        z_pos = self.r*np.cos(self.w_c*t)*np.sin(self.w_c*t)/(1+np.sin(self.w_c*t)**2)
        #vel
        x_vel = -self.w_c*self.r*np.sin(t*self.w_c)/(np.sin(t*self.w_c)**2 + 1) - 2*self.w_c*self.r*np.sin(t*self.w_c)*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1)**2 + self.r_dot*np.cos(t*self.w_c)/(np.sin(t*self.w_c)**2 + 1)
        y_vel = 0
        z_vel = -self.w_c*self.r*np.sin(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) + self.w_c*self.r*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) - 2*self.w_c*self.r*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1)**2 + self.r_dot*np.sin(t*self.w_c)*np.cos(t*self.w_c)/(np.sin(t*self.w_c)**2 + 1)
        #acc
        x_acc = (-self.w_c**2*self.r*np.cos(t*self.w_c) + 2*self.w_c**2*(np.sin(t*self.w_c)**2 - np.cos(t*self.w_c)**2 + 4*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1))*self.r*np.cos(t*self.w_c)/(np.sin(t*self.w_c)**2 + 1) + 4*self.w_c**2*self.r*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)/(np.sin(t*self.w_c)**2 + 1) - 2*self.w_c*self.r_dot*np.sin(t*self.w_c) - 4*self.w_c*self.r_dot*np.sin(t*self.w_c)*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) + 2*self.r_ddot*np.cos(t*self.w_c))/(np.sin(t*self.w_c)**2 + 1)
        y_acc = 0
        z_acc = 2*(-2*self.w_c**2*self.r*np.sin(t*self.w_c)*np.cos(t*self.w_c) + self.w_c**2*(np.sin(t*self.w_c)**2 - np.cos(t*self.w_c)**2 + 4*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1))*self.r*np.sin(t*self.w_c)*np.cos(t*self.w_c)/(np.sin(t*self.w_c)**2 + 1) + 2*self.w_c**2*self.r*np.sin(t*self.w_c)**3*np.cos(t*self.w_c)/(np.sin(t*self.w_c)**2 + 1) - 2*self.w_c**2*self.r*np.sin(t*self.w_c)*np.cos(t*self.w_c)**3/(np.sin(t*self.w_c)**2 + 1) - self.w_c*self.r_dot*np.sin(t*self.w_c)**2 + self.w_c*self.r_dot*np.cos(t*self.w_c)**2 - 2*self.w_c*self.r_dot*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) + self.r_ddot*np.sin(t*self.w_c)*np.cos(t*self.w_c))/(np.sin(t*self.w_c)**2 + 1)
        #jerk
        x_jerk = (self.w_c**3*self.r*np.sin(t*self.w_c) - 6*self.w_c**3*(np.sin(t*self.w_c)**2 - np.cos(t*self.w_c)**2 + 4*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1))*self.r*np.sin(t*self.w_c)/(np.sin(t*self.w_c)**2 + 1) + 8*self.w_c**3*(1-3*np.sin(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) + 3*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) - 6*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1)**2)*self.r*np.sin(t*self.w_c)*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) + 6*self.w_c**3*self.r*np.sin(t*self.w_c)*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) - 3*self.w_c**2*self.r_dot*np.cos(t*self.w_c) + 6*self.w_c**2*(np.sin(t*self.w_c)**2 - np.cos(t*self.w_c)**2 + 4*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1))*self.r_dot*np.cos(t*self.w_c)/(np.sin(t*self.w_c)**2 + 1) + 12*self.w_c**2*self.r_dot*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)/(np.sin(t*self.w_c)**2 + 1) - 6*self.w_c*self.r_ddot*np.sin(t*self.w_c) - 12*self.w_c*self.r_ddot*np.sin(t*self.w_c)*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) + 6*self.r_dddot*np.cos(t*self.w_c))/(np.sin(t*self.w_c)**2 + 1)
        y_jerk = 0
        z_jerk = 2*(2*self.w_c**3*self.r*np.sin(t*self.w_c)**2 - 2*self.w_c**3*self.r*np.cos(t*self.w_c)**2 - 3*self.w_c**3*(np.sin(t*self.w_c)**2 - np.cos(t*self.w_c)**2 + 4*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1))*self.r*np.sin(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) + 3*self.w_c**3*(np.sin(t*self.w_c)**2 - np.cos(t*self.w_c)**2 + 4*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1))*self.r*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) + 4*self.w_c**3*(1 - 3*np.sin(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) + 3*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) - 6*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1)**2)*self.r*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) + 12*self.w_c**3*self.r*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) - 6*self.w_c**2*self.r_dot*np.sin(t*self.w_c)*np.cos(t*self.w_c) + 3*self.w_c**2*(np.sin(t*self.w_c)**2 - np.cos(t*self.w_c)**2 + 4*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1))*self.r_dot*np.sin(t*self.w_c)*np.cos(t*self.w_c)/(np.sin(t*self.w_c)**2 + 1) + 6*self.w_c**2*self.r_dot*np.sin(t*self.w_c)**3*np.cos(t*self.w_c)/(np.sin(t*self.w_c)**2 + 1) - 6*self.w_c**2*self.r_dot*np.sin(t*self.w_c)*np.cos(t*self.w_c)**3/(np.sin(t*self.w_c)**2 + 1) - 3*self.w_c*self.r_ddot*np.sin(t*self.w_c)**2 + 3*self.w_c*self.r_ddot*np.cos(t*self.w_c)**2 - 6*self.w_c*self.r_ddot*np.sin(t*self.w_c)**2*np.cos(t*self.w_c)**2/(np.sin(t*self.w_c)**2 + 1) + 3*self.r_ddot*np.sin(t*self.w_c)*np.cos(t*self.w_c))/(np.sin(t*self.w_c)**2 + 1)


        pos_array = np.array([x_pos, y_pos, z_pos])
        vel_array = np.array([x_vel, y_vel, z_vel])
        acc_array = np.array([x_acc, y_acc, z_acc])
        jerk_array = np.array([x_jerk, y_jerk, z_jerk])

        return pos_array, vel_array, acc_array, jerk_array