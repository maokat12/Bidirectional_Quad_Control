import csv

from flightsim.crazyflie_params import quad_params
from flightsim.simulate import Quadrotor

from proj1_3.code.alt_sandbox import Sandbox

# create CSV file w/header
file_name = 'brute_force_search.csv'
fields = ['Kp', 'Kd', 'maze time', 'maze success', 'window time', 'window success', 'success?']
with open(file_name, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

i = 0
k_r = 1
k_w = 1

for k_p in range(16, 25, 1):
    #k_p = k_p * 0.1 + 5
    for k_d in range(4, 11, 1):
        k_d = k_d+0.5

        ### MAZE
        filename = '../util/test_maze.json'
        mazeSandbox = Sandbox(filename, 'maze', k_p, k_d, 1, 1)
        maze_goal, maze_collision, maze_time = mazeSandbox.run_Sandbox(filename, 'maze', k_p, k_d, 1, 1)

        ### WINDOW
        filename = '../util/test_window.json'
        windowSandbox = Sandbox(filename, 'window', k_p, k_d, 1, 1)
        window_goal, window_collision, window_time = windowSandbox.run_Sandbox(filename, 'window', k_p, k_d, 1, 1)

        #save to file
        print('Kp: ', k_p)
        print('Kd: ', k_d)
        window_success = True
        maze_success = True
        search_success = 'fail'

        window_success = window_collision and window_goal
        maze_success = maze_collision and maze_goal
        if window_success and (window_time < 10.0) and maze_success and (maze_time < 10.0):
             search_success = 'pass'

        print('window time', window_time)
        print('maze time', maze_time)
        print('search success?', search_success)

        output = [str(k_p), str(k_d), maze_time, maze_success, window_time, window_success, search_success]
        with open(file_name, 'a') as csvfile:
             csvwriter = csv.writer(csvfile)
             csvwriter.writerow(output)