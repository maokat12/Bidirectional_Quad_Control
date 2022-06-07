import numpy as np

def get_traj(start, r, w_c, theta, t, shape):
    if shape == 'circle':
        return circle_traj(start, r, w_c, theta, t)
    elif shape == 'lemniscate':
        return lemniscate(start, r, w_c, t)
    else:
        print('please check spelling')
        exit()

def circle_traj(start, r, w_c, theta, t):
    r_dot = 0
    r_ddot = 0
    r_dddot = 0

    #pos
    x_pos = r*np.cos(w_c*t) + start[0]-r
    y_pos = r*np.sin(w_c*t)*np.sin(theta) + start[1]
    z_pos = r*np.sin(w_c*t)*np.cos(theta) + start[2]
    #vel
    x_vel = r_dot*np.cos(w_c*t) - r*w_c*np.sin(w_c*t)
    y_vel = np.sin(theta)*(r_dot*np.sin(w_c*t) + r*w_c*np.cos(w_c*t))
    z_vel = np.cos(theta)*(r_dot*np.sin(w_c*t) + r*w_c*np.cos(w_c*t))
    #acc
    x_acc = r_ddot*np.cos(w_c*t) - 2*w_c*r_dot*np.sin(w_c*t) - w_c**2*r*np.cos(w_c*t)
    y_acc = np.sin(theta)*(r_ddot*np.sin(w_c*t) + 2*w_c*r_dot*np.cos(w_c*t) - w_c**2*r*np.sin(w_c*t))
    z_acc = np.cos(theta)*(r_ddot*np.sin(w_c*t) + 2*w_c*r_dot*np.cos(w_c*t) - w_c**2*r*np.sin(w_c*t))
    #jerk
    x_jerk = (r_dddot - 3*w_c**2*x_vel)*np.cos(w_c*t) - (3*w_c*r_ddot - w_c**3*r)*np.sin(w_c*t)
    y_jerk = np.sin(theta)*((r_dddot - 3*w_c**2*x_vel)*np.sin(w_c*t) + (3*w_c*r_ddot - w_c**3*r)*np.cos(w_c*t))
    z_jerk = np.cos(theta)*((r_dddot - 3*w_c**2*x_vel)*np.sin(w_c*t) + (3*w_c*r_ddot - w_c**3*r)*np.cos(w_c*t))

    pos_array = np.array([x_pos, y_pos, z_pos])
    vel_array = np.array([x_vel, y_vel, z_vel])
    acc_array = np.array([x_acc, y_acc, z_acc])
    jerk_array = np.array([x_jerk, y_jerk, z_jerk])

    return pos_array, vel_array, acc_array, jerk_array

def lemniscate(start, r, w_c, t):
    r_dot = 0
    r_ddot = 0
    r_dddot = 0
    #pos
    x_pos = r*np.cos(w_c*t)/(np.sin(w_c*t)**2 + 1) + start[0]-r
    y_pos = start[1]
    z_pos = r*np.cos(w_c*t)*np.sin(w_c*t)/(1+np.sin(w_c*t)**2) + start[2]

    #x_vel, y_vel, z_vel = 0, 0, 0
    #x_acc, y_acc, z_acc = 0, 0, 0
    #x_jerk, y_jerk, z_jerk = 0, 0, 0
    #vel
    x_vel = -w_c*r*np.sin(t*w_c)/(np.sin(t*w_c)**2 + 1) - 2*w_c*r*np.sin(t*w_c)*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1)**2 + r_dot*np.cos(t*w_c)/(np.sin(t*w_c)**2 + 1)
    y_vel = 0
    z_vel = -w_c*r*np.sin(t*w_c)**2/(np.sin(t*w_c)**2 + 1) + w_c*r*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1) - 2*w_c*r*np.sin(t*w_c)**2*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1)**2 + r_dot*np.sin(t*w_c)*np.cos(t*w_c)/(np.sin(t*w_c)**2 + 1)
    #acc
    x_acc = (-w_c**2*r*np.cos(t*w_c) + 2*w_c**2*(np.sin(t*w_c)**2 - np.cos(t*w_c)**2 + 4*np.sin(t*w_c)**2*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1))*r*np.cos(t*w_c)/(np.sin(t*w_c)**2 + 1) + 4*w_c**2*r*np.sin(t*w_c)**2*np.cos(t*w_c)/(np.sin(t*w_c)**2 + 1) - 2*w_c*r_dot*np.sin(t*w_c) - 4*w_c*r_dot*np.sin(t*w_c)*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1) + 2*r_ddot*np.cos(t*w_c))/(np.sin(t*w_c)**2 + 1)
    y_acc = 0
    z_acc = 2*(-2*w_c**2*r*np.sin(t*w_c)*np.cos(t*w_c) + w_c**2*(np.sin(t*w_c)**2 - np.cos(t*w_c)**2 + 4*np.sin(t*w_c)**2*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1))*r*np.sin(t*w_c)*np.cos(t*w_c)/(np.sin(t*w_c)**2 + 1) + 2*w_c**2*r*np.sin(t*w_c)**3*np.cos(t*w_c)/(np.sin(t*w_c)**2 + 1) - 2*w_c**2*r*np.sin(t*w_c)*np.cos(t*w_c)**3/(np.sin(t*w_c)**2 + 1) - w_c*r_dot*np.sin(t*w_c)**2 + w_c*r_dot*np.cos(t*w_c)**2 - 2*w_c*r_dot*np.sin(t*w_c)**2*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1) + r_ddot*np.sin(t*w_c)*np.cos(t*w_c))/(np.sin(t*w_c)**2 + 1)
    #jerk
    x_jerk = (w_c**3*r*np.sin(t*w_c) - 6*w_c**3*(np.sin(t*w_c)**2 - np.cos(t*w_c)**2 + 4*np.sin(t*w_c)**2*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1))*r*np.sin(t*w_c)/(np.sin(t*w_c)**2 + 1) + 8*w_c**3*(1-3*np.sin(t*w_c)**2/(np.sin(t*w_c)**2 + 1) + 3*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1) - 6*np.sin(t*w_c)**2*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1)**2)*r*np.sin(t*w_c)*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1) + 6*w_c**3*r*np.sin(t*w_c)*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1) - 3*w_c**2*r_dot*np.cos(t*w_c) + 6*w_c**2*(np.sin(t*w_c)**2 - np.cos(t*w_c)**2 + 4*np.sin(t*w_c)**2*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1))*r_dot*np.cos(t*w_c)/(np.sin(t*w_c)**2 + 1) + 12*w_c**2*r_dot*np.sin(t*w_c)**2*np.cos(t*w_c)/(np.sin(t*w_c)**2 + 1) - 6*w_c*r_ddot*np.sin(t*w_c) - 12*w_c*r_ddot*np.sin(t*w_c)*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1) + 6*r_dddot*np.cos(t*w_c))/(np.sin(t*w_c)**2 + 1)
    y_jerk = 0
    z_jerk = 2*(2*w_c**3*r*np.sin(t*w_c)**2 - 2*w_c**3*r*np.cos(t*w_c)**2 - 3*w_c**3*(np.sin(t*w_c)**2 - np.cos(t*w_c)**2 + 4*np.sin(t*w_c)**2*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1))*r*np.sin(t*w_c)**2/(np.sin(t*w_c)**2 + 1) + 3*w_c**3*(np.sin(t*w_c)**2 - np.cos(t*w_c)**2 + 4*np.sin(t*w_c)**2*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1))*r*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1) + 4*w_c**3*(1 - 3*np.sin(t*w_c)**2/(np.sin(t*w_c)**2 + 1) + 3*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1) - 6*np.sin(t*w_c)**2*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1)**2)*r*np.sin(t*w_c)**2*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1) + 12*w_c**3*r*np.sin(t*w_c)**2*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1) - 6*w_c**2*r_dot*np.sin(t*w_c)*np.cos(t*w_c) + 3*w_c**2*(np.sin(t*w_c)**2 - np.cos(t*w_c)**2 + 4*np.sin(t*w_c)**2*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1))*r_dot*np.sin(t*w_c)*np.cos(t*w_c)/(np.sin(t*w_c)**2 + 1) + 6*w_c**2*r_dot*np.sin(t*w_c)**3*np.cos(t*w_c)/(np.sin(t*w_c)**2 + 1) - 6*w_c**2*r_dot*np.sin(t*w_c)*np.cos(t*w_c)**3/(np.sin(t*w_c)**2 + 1) - 3*w_c*r_ddot*np.sin(t*w_c)**2 + 3*w_c*r_ddot*np.cos(t*w_c)**2 - 6*w_c*r_ddot*np.sin(t*w_c)**2*np.cos(t*w_c)**2/(np.sin(t*w_c)**2 + 1) + 3*r_ddot*np.sin(t*w_c)*np.cos(t*w_c))/(np.sin(t*w_c)**2 + 1)

    pos_array = np.array([x_pos, y_pos, z_pos])
    vel_array = np.array([x_vel, y_vel, z_vel])
    acc_array = np.array([x_acc, y_acc, z_acc])
    jerk_array = np.array([x_jerk, y_jerk, z_jerk])

    return pos_array, vel_array, acc_array, jerk_array
