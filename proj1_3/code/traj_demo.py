import inspect
import csv
import json
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import time

from flightsim.animate import animate
from flightsim.axes3ds import Axes3Ds
from flightsim.crazyflie_params import quad_params
from flightsim.simulate import Quadrotor, simulate, ExitStatus
from flightsim.world import World

from proj1_3.code.occupancy_map import OccupancyMap
from proj1_3.code.se3_control import SE3Control
from proj1_3.code.world_traj import WorldTraj
from proj1_3.code.hopf_fibration_control import HopfFibrationControl
from proj1_3.code.tray_waypoints import WaypointTraj
from proj1_3.code.world_traj_mod import WorldTrajMod

#world file
filename = '../util/test_maze.json'

# Load the test example.
file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / '..' / 'util' / filename
world = World.from_file(file)          # World boundary and obstacles.
start  = world.world['start']          # Start point, shape=(3,)
goal   = world.world['goal']           # Goal point, shape=(3,)

# This object defines the quadrotor dynamical model and should not be changed.
quadrotor = Quadrotor(quad_params)
robot_radius = 0.25

# Your SE3Control object (from project 1-1).
#my_se3_control = SE3Control(quad_params)
my_se3_control = HopfFibrationControl(quad_params)

# Map Traj Object
#my_world_traj = WorldTraj(world, start, goal) #for forced circle trajectories

# Waypoint Trajectory Objectz
my_waypoint_traj = WaypointTraj(start, 2, 1, 0, 8, 2, 'circle')
cons = my_waypoint_traj.get_contraints()

#narrow window constraints
des_thrust = quad_params['k_thrust']*4*(1*quad_params['rotor_speed_max'])**2
#pos = np.array([[0, -1, 0], [0, 0, 4], [0, 3.5, 4],  [0, 4, 4], [0, 4.5, 4], [0, 8, 4], [0, 9, 0]])
pos = np.array([[0, -1, -1], [0, 0, 4], [0, 4, 4], [0, 8, 4], [0, 9, -1]]) #'narrow window'
o = np.array([1,1,-1,-1,-1])

#straight line paths
#pos = np.array([[0, 0, 0], [0, 0, 2], [0, 0, 4], [0, 0, 8],[0, 0, 4], [0, 0, 2], [0, 0, -1]]) #verticle path
#pos = np.array([[0, 0, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0],[0, 4, 0], [0, 2, 0], [0, -1, 0]]) #y axis travel
#pos = np.array([[0, 0, 0], [1, 2, 0], [2, 4, 0], [4, 8, 0],[2, 4, 0], [1, 2, 0], [0, -1, 0]]) #horizontal travel
#o = np.array([1,1,1,-1,-1,-1,-1])

#circular path
#pos = np.array(([0, 0, -2],[0, np.sqrt(2), -np.sqrt(2)],[0, 2, 0], [0, np.sqrt(2), np.sqrt(2)], [0, 0, 2], [0, -np.sqrt(2), np.sqrt(2)],
#               [0, -2, 0], [0, -np.sqrt(2), -np.sqrt(2)], [0, 0, -2], [0, np.sqrt(2), -np.sqrt(2)])) #8 point circle traj
#o = np.array([1,1,1,1,-1,-1,-1,-1,-1,-1])

maze_points = [([1., 5., 1.5]), ([1.125, 4.625, 1.625]), ([1.375, 2.375, 1.625]), ([2.875, 1.125, 1.625]),
               ([8.625, 1.375, 1.625]), ([8.875, 1.875, 1.625]), ([8.625, 2.625, 1.625]),
               ([8.125, 2.875, 1.625]), ([6.375, 3.125, 1.625]), ([5.375, 3.875, 1.625]),
               ([4.625, 4.125, 1.625]), ([4.125, 4.875, 1.625]), ([4.375, 7.125, 1.625]),
               ([4.875, 7.375, 1.625]), ([7.625, 7.125, 1.625]), ([7.875, 7.125, 1.625]), ([9., 7., 1.5])]
#pos = np.array(maze_points)
#o = np.append(np.ones(7), -1*np.ones(10))
#acc = np.array([[0, 0, des_thrust/quad_params['mass']*(-0-9.81), 2]]) #[-1, -1, -g] - -1: not specified
#o = np.array([-1, -1, -1, -1, -1])

#box path
#pos = np.array([[0, 0, 0], [0, 0, 5], [5, 5, 5], [5, 5, 0], [0, 0, 0]])
#o = np.array([1, 1, -1, -1, -1])


#pos = np.array([[0, 0, 0],[0, 0, 0]])
#o = np.array([-1,-1,-1])

cons = (pos, o)
#start = [5, 0, 0]
start = pos[0]
goal = pos[-1]

my_world_traj = WorldTrajMod(world, start, cons) #for min snap w/body angle control

# Set simulation parameters.
t_final = 10

#a = 0
#b = 0
#c = -1
[a, b ,c] = [0, 0, o[0]]
[a, b, c] = [a, b, c]/np.linalg.norm([a, b, c])
#print('o', -1)
#print('abc', [a, b, c])

if c > 0:
    q = (1/np.sqrt(2*(1+c))) * np.array([-b,a,0, 1+c]) # [i,j,k,w]
else:
    q = 1/np.sqrt(2*(1-c)) * np.array([1-c, 0, a, -b]) # [i,j,k,w]
print('q', q)
initial_state = {'x': start,
                 'v': (0, 0, 0),
                 'q': q, #[i,j,k,w]
                 'w': (0, 0, 0)}

# Perform simulation
print('Simulate.')
(sim_time, state, control, flat, exit) = simulate(initial_state,
                                              quadrotor,
                                              my_se3_control,
                                              my_world_traj,
                                              t_final)
#print(exit.value)

flight_time = sim_time[-1]
flight_distance = np.sum(np.linalg.norm(np.diff(state['x'], axis=0),axis=1))

# Print results.
print(f"  Flight time:     {flight_time:.1f} seconds")
print(f"  Flight distance: {flight_distance:.1f} meters")

# Plot Results
# Position and Velocity vs. Time
(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Position vs Time')
x = state['x']
x_des = flat['x']
ax = axes[0]
ax.plot(sim_time, x_des[:,0], 'r', sim_time, x_des[:,1], 'g', sim_time, x_des[:,2], 'b')
ax.plot(sim_time, x[:,0], 'r.',    sim_time, x[:,1], 'g.',    sim_time, x[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('position, m')
ax.grid('major')
ax.set_title('Position')
v = state['v']
v_des = flat['x_dot']
ax = axes[1]
ax.plot(sim_time, v_des[:,0], 'r', sim_time, v_des[:,1], 'g', sim_time, v_des[:,2], 'b')
ax.plot(sim_time, v[:,0], 'r.',    sim_time, v[:,1], 'g.',    sim_time, v[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('velocity, m/s')
ax.set_xlabel('time, s')
ax.grid('major')

# Force
(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Force vs Time')
a_des = control['F_des']
ax = axes[0]
ax.plot(sim_time, a_des[:,0], 'r', sim_time, a_des[:,1], 'g', sim_time, a_des[:,2], 'b')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.grid('major')
ax.set_title('`f des`')
abc_des = control['abc']
ax = axes[1]
ax.plot(sim_time, abc_des[:,0], 'r', sim_time, abc_des[:,1], 'g', sim_time, abc_des[:,2], 'b')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.grid('major')
ax.set_title('b3 des')

'''
# Desired Body Rates
(fig, axes) = plt.subplots(nrows=1, ncols=1, sharex=True, num='Desired Body Rates vs Time')
a_des = control['w_des']
ax = axes
ax.plot(sim_time, a_des[:,0], 'r.', sim_time, a_des[:,1], 'g.', sim_time, a_des[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.grid('major')
ax.set_title('w des')
'''

# Acceleration and Jerk vs Time
(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Acceleration vs Time')
a_des = flat['x_ddot']
ax = axes[0]
ax.plot(sim_time, a_des[:,0], 'r', sim_time, a_des[:,1], 'g', sim_time, a_des[:,2], 'b')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('acceleration, m/s^2')
ax.grid('major')
ax.set_title('Acceleration')
j_des = flat['x_dddot']
ax = axes[1]
ax.plot(sim_time, j_des[:,0], 'r', sim_time, j_des[:,1], 'g', sim_time, j_des[:,2], 'b')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('jerk, m/s')
ax.set_xlabel('time, s')
ax.grid('major')

#b3 state and b3 des
s = np.array([0, 0, 0])
for i in state['q']:
    b3 = Rotation.from_quat(i).apply(np.array([0, 0, 1]))
    s = np.vstack([s, b3])
s = np.delete(s, 1, 0)
(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Unit Orientation vs Time')
ax = axes[0]
ax.plot(sim_time, s[:,0], 'r.', sim_time, s[:,1], 'g.', sim_time, s[:,2], 'b.')
ax.legend(('a', 'b', 'c'), loc='upper right')
ax.set_ylabel('b3 state, rad/s')
ax.set_xlabel('time, s')
ax.grid('major')
ax.set_title('b3')
abc_des = control['abc']
ax = axes[1]
ax.plot(sim_time, abc_des[:,0], 'r', sim_time, abc_des[:,1], 'g', sim_time, abc_des[:,2], 'b')
ax.legend(('a', 'b', 'c'), loc='upper right')
ax.grid('major')
ax.set_title('b3 des')

# Orientation and Angular Velocity vs. Time
(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Orientation vs Time')
q_des = control['cmd_q']
q = state['q']
ax = axes[0]
ax.plot(sim_time, q_des[:,0], 'r', sim_time, q_des[:,1], 'g', sim_time, q_des[:,2], 'b', sim_time, q_des[:,3], 'k')
ax.plot(sim_time, q[:,0], 'r.',    sim_time, q[:,1], 'g.',    sim_time, q[:,2], 'b.',    sim_time, q[:,3],     'k.')
ax.legend(('i', 'j', 'k', 'w'), loc='upper right')
ax.set_ylabel('quaternion')
ax.set_xlabel('time, s')
ax.grid('major')
w = state['w']
ax = axes[1]
ax.plot(sim_time, w[:,0], 'r.', sim_time, w[:,1], 'g.', sim_time, w[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('angular velocity, rad/s')
ax.set_xlabel('time, s')
ax.grid('major')

# Commands vs. Time
(fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Commands vs Time')
s = control['cmd_motor_speeds']
ax = axes[0]
ax.plot(sim_time, s[:,0], 'r.', sim_time, s[:,1], 'g.', sim_time, s[:,2], 'b.', sim_time, s[:,3], 'k.')
ax.legend(('1', '2', '3', '4'), loc='upper right')
ax.set_ylabel('motor speeds, rad/s')
ax.grid('major')
ax.set_title('Commands')
M = control['cmd_moment']
ax = axes[1]
ax.plot(sim_time, M[:,0], 'r.', sim_time, M[:,1], 'g.', sim_time, M[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('moment, N*m')
ax.grid('major')
T = control['cmd_thrust']
ax = axes[2]
ax.plot(sim_time, T, 'k.')
ax.set_ylabel('thrust, N')
ax.set_xlabel('time, s')
ax.grid('major')
'''
#Quad Orientation and Time
(fig, axes) = plt.subplots(nrows=1, ncols=1, sharex=True, num='Quad Orientation vs Time')
s = control['cmd_o']
ax = axes
ax.plot(sim_time, s)
#ax.legend(('orientation'), loc='upper right')
ax.set_ylabel('orientation')
ax.set_xlabel('time (s)')
ax.grid('major')
ax.set_title('Quad Orientation')
'''
#Sign Force and Time
(fig, axes) = plt.subplots(nrows=7, ncols=1, sharex=True, num='Quad Orientation vs Time')
s = control['mode']
ax = axes[0]
ax.plot(sim_time, s)
ax.set_ylabel('mode')
ax.grid('major')
ax.set_title('Mode')

s = control['sign_f']
ax = axes[1]
ax.plot(sim_time, s)
ax.set_ylabel('sign_f')
ax.grid('major')
ax.set_title('sign_f')

s = control['cmd_o']
ax = axes[2]
ax.plot(sim_time, s)
ax.set_ylabel('quad_o')
ax.grid('major')
ax.set_title('quad_o')

abc_des = control['abc']
ax = axes[3]
ax.plot(sim_time, abc_des[:,0], 'r.', sim_time, abc_des[:,1], 'g.', sim_time, abc_des[:,2], 'b.')
ax.legend(('a', 'b', 'c'), loc='upper right')
ax.grid('major')
ax.set_title('b3 des')

abc_des = control['F_des']
ax = axes[4]
ax.plot(sim_time, abc_des[:,0], 'r.', sim_time, abc_des[:,1], 'g.', sim_time, abc_des[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.grid('major')
ax.set_title('F des')

a_des = control['acc_des']
ax = axes[5]
ax.plot(sim_time, a_des[:,0], 'r.', sim_time, a_des[:,1], 'g.', sim_time, a_des[:,2], 'b.')
#ax.plot(sim_time, a_des)
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.grid('major')
ax.set_xlabel('time (s)')
ax.set_title('r_ddot_des')

s = control['F_mag']
ax = axes[6]
ax.plot(sim_time, s)
ax.set_ylabel('f_mag')
ax.grid('major')
ax.set_title('force magnitude')

#Ideal Trajectory (w/o position gains)
(fig, axes) = plt.subplots(nrows=7, ncols=1, sharex=True, num='Ideal Trajectory')

s = control['ideal_traj'][:, 0] #mode
ax = axes[0]
ax.plot(sim_time, s)
ax.set_ylabel('mode')
ax.grid('major')
ax.set_title('Mode')

s = control['ideal_traj'][:, 1] #F sign
ax = axes[1]
ax.plot(sim_time, s)
ax.set_ylabel('sign_f')
ax.grid('major')
ax.set_title('sign_f')

s = control['ideal_traj'][:, 2] #quad orientation
ax = axes[2]
ax.plot(sim_time, s)
ax.set_ylabel('quad_o')
ax.grid('major')
ax.set_title('quad_o')

abc_des = control['ideal_traj'][:,3] #abc
a = control['ideal_traj'][:,3]
b = control['ideal_traj'][:,4]
c = control['ideal_traj'][:,5]
ax = axes[3]
ax.plot(sim_time, a, 'r.', sim_time, b, 'g.', sim_time, c, 'b.')
ax.legend(('a', 'b', 'c'), loc='upper right')
ax.grid('major')
ax.set_title('b3 des')

abc_des = control['ideal_traj'][4] #F des
x = control['ideal_traj'][:,6]
y = control['ideal_traj'][:,7]
z = control['ideal_traj'][:,8]
ax = axes[4]
ax.plot(sim_time, x, 'r.', sim_time, y, 'g.', sim_time, z, 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.grid('major')
ax.set_title('F des')

a_des = control['ideal_traj'][5] #w des
x = control['ideal_traj'][:,9]
y = control['ideal_traj'][:,10]
z = control['ideal_traj'][:,11]
ax = axes[5]
ax.plot(sim_time, x, 'r.', sim_time, y, 'g.', sim_time, z, 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.grid('major')
ax.set_xlabel('time (s)')
ax.set_title('w_des')

s = control['ideal_traj'][:,12]
ax = axes[6]
ax.plot(sim_time, s)
ax.set_ylabel('f_mag')
ax.grid('major')
ax.set_title('force magnitude')

'''
#z_ddot vs y_ddot
(fig, axes) = plt.subplots(nrows=1, ncols=1, sharex=True, num='y_ddot vs z_ddot')
a_des = control['r_des']
ax = axes
ax.plot(a_des[:,1], a_des[:,2], 'r.')
ax.plot(a_des[0,1], a_des[0,2], 'go', markersize=8, markeredgewidth=3, markerfacecolor='none')
ax.plot(a_des[-1,2], a_des[-1,2], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
ax.plot(0, -9.81, 'bo', markersize=16, markeredgewidth=3, markerfacecolor='none')
ax.grid('major')
ax.set_title('y_ddot vs z_ddot des')
'''
#z_ddot vs y_ddot
# Desired Body Rates
(fig, axes) = plt.subplots(nrows=1, ncols=1, sharex=True, num='y_ddot vs z_ddot planned')
a_des = control['acc_plan']
ax = axes
ax.plot(a_des[:,1], a_des[:,2], 'r.')
ax.plot(a_des[0,1], a_des[0,2], 'go', markersize=8, markeredgewidth=3, markerfacecolor='none')
ax.plot(a_des[-1,1], a_des[-1,2], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
ax.plot(0, -9.81, 'bo', markersize=16, markeredgewidth=3, markerfacecolor='none')
ax.grid('major')
ax.set_title('y_ddot vs z_ddot des')

#z_ddot vs x_ddot
# Desired Body Rates
(fig, axes) = plt.subplots(nrows=1, ncols=1, sharex=True, num='x_ddot vs z_ddot planned')
a_des = control['acc_plan']
ax = axes
ax.plot(a_des[:,0], a_des[:,2], 'r.')
ax.plot(a_des[0,0], a_des[0,2], 'go', markersize=8, markeredgewidth=3, markerfacecolor='none')
ax.plot(a_des[-1,0], a_des[-1,2], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
ax.plot(0, -9.81, 'bo', markersize=16, markeredgewidth=3, markerfacecolor='none')
ax.grid('major')
ax.set_title('y_ddot vs z_ddot des')

#acceleration vs gain components
# Desired Body Rates
(fig, axes) = plt.subplots(nrows=1, ncols=1, sharex=True, num='planned acc vs gain component')
acc = control['a_mag']
gain = control['gain_mag']
ax = axes
ax.plot(sim_time, acc, 'r.')
ax.plot(sim_time, gain, 'b.')
ax.plot(sim_time, np.ones(len(sim_time))*(-9.81), 'g.')
ax.legend(('planned acc', 'gain component'), loc='upper right')
ax.grid('major')
ax.set_title('planned acc vs gain component')

# 3D Paths
fig = plt.figure('3D Path')
ax = Axes3Ds(fig)
world.draw_custom_world(ax)
ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
world.draw_line(ax, flat['x'], color='black', linewidth=2)
world.draw_points(ax, state['x'], color='blue', markersize=4)
world.draw_points(ax, pos, color='purple', markersize=8)
ax.legend(handles=[
    Line2D([], [], color='black', linewidth=2, label='Trajectory'),
    Line2D([], [], color='blue', linestyle='', marker='.', markersize=4, label='Flight')],
    loc='upper right')

# Animation (Slow)
# Instead of viewing the animation live, you may provide a .mp4 filename to save.
R = Rotation.from_quat(state['q']).as_matrix()
filename = "test.mp4"
filename = None
ani = animate(sim_time, state['x'], R, world=world, filename=filename)
plt.show()

#export to csv
#desired acc data
'''
with open('acc_cons.csv', 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    # writing the data rows
    a = np.hstack([cons[0], cons[1]])
    csvwriter.writerows(a)

with open('real_acc.csv', 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    # writing the data rows
    a = np.hstack([x, x_des, a_des])
    csvwriter.writerows(a)
with open('time.csv', 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    # writing the data rows
    a = np.hstack([x, a_des])
    csvwriter.writerow(sim_time)
'''