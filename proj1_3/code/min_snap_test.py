import numpy as np
import cvxpy as cp
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sympy import cos, sin, diff, symbols


#useful functions
def get_distance(p1, p2):  # return distance between two points
    p1 = np.array(p1)
    p2 = np.array(p2)
    dist = abs(np.linalg.norm(p2 - p1))
    return dist

# point lists to test on
maze_points = [([1., 5., 1.5]), ([1.125, 4.625, 1.625]), ([1.375, 2.375, 1.625]), ([2.875, 1.125, 1.625]),
               ([8.625, 1.375, 1.625]), ([8.875, 1.875, 1.625]), ([8.625, 2.625, 1.625]),
               ([8.125, 2.875, 1.625]), ([6.375, 3.125, 1.625]), ([5.375, 3.875, 1.625]),
               ([4.625, 4.125, 1.625]), ([4.125, 4.875, 1.625]), ([4.375, 7.125, 1.625]),
               ([4.875, 7.375, 1.625]), ([7.625, 7.125, 1.625]), ([7.875, 7.125, 1.625]), ([9., 7., 1.5])]

maze_points_short_removed = [[1,  5,  1.5], [1.125,   4.875 ,  1.625  ], [1.125,   3.875 ,  1.625  ], [1.125,   2.875 ,  1.625  ], [2.   ,   2.    ,  1.625  ], [2.875,   1.125 ,  1.625  ],
                             [3.53125, 1.125 ,  1.625  ], [4.1875,  1.125 ,  1.625  ], [4.84375, 1.125 ,  1.625  ], [5.5     ,1.125 ,  1.625  ], [6.15625, 1.125 ,  1.625  ], [6.8125 , 1.125 ,  1.625  ], [7.46875, 1.125 ,  1.625  ],
                             [8.125  , 1.125 ,  1.625  ], [8.875 ,  1.875 ,  1.625  ], [8.875  , 2.125 ,  1.625  ], [8.625  , 2.375 ,  1.625  ], [8.625  , 2.625 ,  1.625  ], [8.375 ,  2.875 ,  1.625  ],
                             [7.375  , 2.875 ,  1.625  ], [6.375 ,  2.875 ,  1.625  ], [5.875 ,  3.375 ,  1.625  ], [5.375 ,  3.875 ,  1.625  ], [5.125 ,  3.875 ,  1.625  ], [4.625 ,  4.375 ,  1.625  ], [4.125 ,  4.875,   1.625  ],
                             [4.125  , 5.75  ,  1.625  ], [4.125 ,  6.625,   1.625  ], [4.875 ,  7.375,   1.625  ], [6.    ,  7.375,   1.625  ], [7.125 ,  7.375,   1.625  ], [7.375 ,  7.125,   1.625  ], [8.125 ,  7.125,   1.625  ], [8.875 ,  7.125,   1.625  ],
                             [9.     , 7.     , 1.5    ]]

window_points = [([0.7, -4.3, 0.7]), ([0.625, -3.875, 0.625]), ([0.875, -1.375, 0.625]),([2.375, 0.125, 0.875]),
                 ([3.875, 1.875, 2.375]), ([4.125, 12.875, 2.625]), ([5.875, 14.875, 4.375]),([6.125, 16.125, 4.375]),
                 ([6.625, 16.625, 4.125]), ([7.875, 17.875, 3.125]), ([8., 18., 3.])]

over_n_under_points =[[0.5, 2.5, 5.5], [0.625, 2.625, 5.375], [1.625, 2.625, 4.375], [1.625, 2.625, 2.375], [2.625, 2.625, 1.375],
                      [4.375, 2.625, 1.375], [4.625, 2.625, 1.625], [4.625, 2.625, 3.625], [5.625, 2.625, 4.625], [7.375, 2.625, 4.625],
                      [7.625, 2.625, 4.375], [7.625, 2.625, 2.375], [8.625, 2.625, 1.375], [10.375, 2.625, 1.375], [10.625, 2.625, 1.625],
                      [10.625, 2.625, 3.625], [11.625, 2.625, 4.625], [13.375, 2.625, 4.625], [13.625, 2.625, 4.375], [13.625, 2.625, 2.375],
                      [14.625, 2.625, 1.375], [16.375, 2.625, 1.375], [16.625, 2.625, 1.625], [16.625, 2.625, 3.625], [17.625, 2.625, 4.625],
                      [18.125, 2.625, 4.625], [18.875, 2.625, 5.375], [19.0, 2.5, 5.5]]

points = maze_points
#points = points[0:4]
print(points)
plot = True

dist_vel = 1.5 #m/s
vel_max = 3 #m/s

#get distances
distances = []
for i in range(len(points)-1):
    dist = get_distance(points[i+1], points[i])
    distances.append(dist)
distances = np.array(distances)
time_segments = distances/dist_vel
#print('distances', distances)
#print('time segments', time_segments)

#add a little extra time to start and stop
time_segments[0] = time_segments[0]+0.3
time_segments[-1] = time_segments[-1]+0.3

#print('distances', distances)
#print('time segments', time_segments)

num_segments = len(time_segments)
#print('num segments', num_segments)

#set up QP Hessian
H = None
f = np.zeros((1, 8*num_segments))
for i in range(num_segments):
    T = time_segments[i]
    H_sub = 2*96*np.array([[1050*T**7,  525*T**6,   210*T**5,   52.5*T**4],
                          [525*T**6,  270*T**5,   112.5*T**4, 30*T**3],
                          [210*T**5,  112.5*T**4, 50*T**3,    15*T**2],
                          [52.5*T**4, 30*T**3,    15*T**2,    6*T]])
    H_sub = np.block([[H_sub, np.zeros((4, 4))],
                      [np.zeros((4, 8))]])
    if i == 0: #start of construction
        H = H_sub
    else:
        H = np.block([[H, np.zeros((8*i, 8))],
                      [np.zeros((8, 8*i)), H_sub]])
#print("H shape", H.shape)

#set up equality constraints
# start/end constraints
A_eq_start = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 2, 0, 0],
                       [0, 0, 0, 0, 6, 0, 0, 0]])
tf = time_segments[-1]
A_eq_end = np.array([[tf**7, tf**6, tf**5, tf**4, tf**3, tf**2, tf**1, 1],
                     [7*(tf**6), 6*(tf**5), 5*(tf**4), 4*(tf**3), 3*(tf**2), 2*tf, 1, 0],
                     [42*(tf**5), 30*(tf**4), 20*(tf**3), 12*(tf**2), 6*tf, 2, 0, 0],
                     [210*(tf**4), 120*(tf**3), 60*(tf**2), 24*tf, 6, 0, 0, 0]])
A_eq = np.block([[A_eq_start, np.zeros((4, 8*(num_segments-1)))],
                 [np.zeros((4, 8*(num_segments-1))), A_eq_end]])

beq_x = np.array([points[0][0], 0, 0, 0, points[-1][0], 0, 0, 0])
beq_y = np.array([points[0][1], 0, 0, 0, points[-1][1], 0, 0, 0])
beq_z = np.array([points[0][2], 0, 0, 0, points[-1][2], 0, 0, 0])

if num_segments > 1:
    #waypoints constraints
    for i in range(num_segments-1):
        T = time_segments[i]
        A_eq_sub = np.array([T**7, T**6, T**5, T**4, T**3, T**2, T, 1])
        RHS = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        A_eq_sub = np.block([[np.zeros((1, 8*i)), A_eq_sub, np.zeros((1, 8*(num_segments-(i+1))))],
                            [np.zeros((1, 8*(i+1))), RHS, np.zeros((1, max(0, 8*(num_segments-(i+2)))))]])
        A_eq = np.vstack((A_eq,
                          A_eq_sub))
        beq_x = np.hstack((beq_x, points[i+1][0], points[i+1][0]))
        beq_y = np.hstack((beq_y, points[i+1][1], points[i+1][1]))
        beq_z = np.hstack((beq_z, points[i+1][2], points[i+1][2]))

    #continuity constraints
    for i in range(num_segments-1):
        T = time_segments[i]
        A_eq_sub_LHS = np.block([[7*T**6, 6*T**5, 5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],
                                 [42*T**5, 30*T**4, 20*T**3, 12*T**2, 6*T, 2, 0, 0],
                                 [210*T**4, 120*T**3, 60*T**2, 24*T, 6, 0, 0, 0]])
        A_eq_sub_RHS = np.block([[0, 0, 0, 0, 0, 0, -1, 0],
                                 [0, 0, 0, 0, 0, -2, 0, 0],
                                 [0, 0, 0, 0, -6, 0, 0, 0]])
        A_eq_sub = np.block([np.zeros((3,8*i)), A_eq_sub_LHS, A_eq_sub_RHS, np.zeros((3, 8*(num_segments-(i+2))))])
        A_eq = np.vstack((A_eq,
                          A_eq_sub))
        beq_x = np.hstack((beq_x, np.zeros(3)))
        beq_y = np.hstack((beq_y, np.zeros(3)))
        beq_z = np.hstack((beq_z, np.zeros(3)))

#inequality constraints - position
i = 10 #number of intervals
A = np.array([])
m = 0.2 #distance deviance threshold
b_x = np.array([])
b_y = np.array([])
b_z = np.array([])
for k in range(num_segments):
    T = time_segments[k]
    start = points[k]
    stop = points[k+1]
    for j in range(i+1):
        A_sub = np.block([[(T*j/i)**7, (T*j/i)**6, (T*j/i)**5, (T*j/i)**4, (T*j/i)**3, (T*j/i)**2, (T*j/i), 1],
                          [-(T*j/i)**7, -(T*j/i)**6, -(T*j/i)**5, -(T*j/i)**4, -(T*j/i)**3, -(T*j/i)**2, -(T*j/i), -1]])
        if j == 0 and k == 0:
            A = np.block([A_sub, np.zeros((2, 8*(num_segments-(k+1))))])
        else:
            A = np.block([[A],
                          [np.zeros((2, 8*k)), A_sub, np.zeros((2, 8*(num_segments-(k+1))))]])
        b_x = np.hstack((b_x, max(start[0], stop[0])+m, -1*min(start[0], stop[0])+m))
        b_y = np.hstack((b_y, max(start[1], stop[1])+m, -1*min(start[1], stop[1])+m))
        b_z = np.hstack((b_z, max(start[2], stop[2])+m, -1*min(start[2], stop[2])+m))

#inequality constraints - velocity
for k in range(num_segments):
    T = time_segments[k]
    for j in range(i+1):
        A_sub = np.block([[7*(T*j/i)**6, 6*(T*j/i)**5, 5*(T*j/i)**4, 4*(T*j/i)**3, 3*(T*j/i)**2, 2*T*j/i, 1, 0],
                          [-7*(T*j/i)**6, -6*(T*j/i)**5, -5*(T*j/i)**4, -4*(T*j/i)**3, -3*(T*j/i)**2, -2*(T*j/i), -1, 0]])
        A = np.block([[A],
                      [np.zeros((2, 8*k)), A_sub, np.zeros((2, 8*(num_segments-(k+1))))]])
        b_x = np.hstack((b_x, vel_max, vel_max))
        b_y = np.hstack((b_y, vel_max, vel_max))
        b_z = np.hstack((b_z, vel_max, vel_max))

b_eq = np.vstack((beq_x, beq_y, beq_z))
b = np.vstack((b_x, b_y, b_z))

#print numpy arrays to csv
#np.savetxt("A_eq.csv", A_eq, delimiter=",")
#np.savetxt("A.csv", A, delimiter=",")
#np.savetxt("b_eq.csv", b_eq, delimiter=",")
#np.savetxt("b.csv", b, delimiter=",")
#np.savetxt("H.csv", H, delimiter = ',')
#exit()

#print('A_eq shape', A_eq.shape)
#print('bx eq shape', beq_x.shape)
#print('Aineq shape',A.shape)
#print('bx ineq shape', b_x.shape)
#print('bx', b_x)
#exit()
'''
#test QP w/cxvpy - no constraints yet - this failed at 3 segments
x = cp.Variable(8*num_segments)
prob_x = cp.Problem(cp.Minimize(cp.quad_form(x, H)), [])
prob_x.solve()

print("\nThe optimal value is", prob_x.value)
print("A solution x is")
print(x.value)
''' #cvxpy failed me

#test with scipy.optimize
#holy shit scipy works
#carried by https://stackoverflow.com/questions/17009774/quadratic-program-qp-solver-that-only-depends-on-numpy-scipy
def loss(w, sign=1.):
    return sign * (0.5 * np.dot(w.T, np.dot(H, w)))

def jac(w, sign=1.):
    return sign * (np.dot(w.T, H))
#solve x
x_eq_cons = {'type': 'eq',
             'fun':lambda x: beq_x - np.dot(A_eq, x),
             'jac':lambda x: -A_eq}
x_ineq_cons = {'type': 'ineq',
               'fun': lambda x: b_x - np.dot(A, x),
               'jac':lambda x: -A}
cons_x = [x_eq_cons, x_ineq_cons]
x0 = np.random.randn(len(H)) #declares variables for the QP sovler
res_cons_x = optimize.minimize(loss, x0, jac=jac,constraints=cons_x, method='SLSQP', options={'disp':False})
#print('constrained', res_cons_x)

#solve y
y_eq_cons = {'type': 'eq',
             'fun':lambda y: beq_y - np.dot(A_eq, y),
             'jac':lambda y: -A_eq}
y_ineq_cons = {'type': 'ineq',
               'fun': lambda y: b_y - np.dot(A, y),
               'jac':lambda y: -A}
cons_y = [y_eq_cons, y_ineq_cons]
y0 = np.random.randn(len(H)) #declares variables for the QP sovler
res_cons_y = optimize.minimize(loss, y0, jac=jac,constraints=cons_y, method='SLSQP', options={'disp':False})
#print('constrained', res_cons_y)

#solve z
z_eq_cons = {'type': 'eq',
             'fun':lambda z: beq_z - np.dot(A_eq, z),
             'jac':lambda z: -A_eq}
z_ineq_cons = {'type': 'ineq',
               'fun': lambda z: b_z - np.dot(A, z),
               'jac':lambda z: -A}
cons_z = [z_eq_cons, z_ineq_cons]
z0 = np.random.randn(len(H)) #declares variables for the QP sovler
res_cons_z = optimize.minimize(loss, z0, jac=jac,constraints=cons_z, method='SLSQP', options={'disp':False})
#print('constrained', res_cons_z)

#plot!
if plot:
    cont_time = 0
    plt.figure(5)
    ax = plt.axes(projection='3d')
    for i in range(num_segments):
        T = time_segments[i]
        time_math = np.linspace(0, T, 50)
        time_plot = np.linspace(cont_time, cont_time+T, 50)

        #constant
        w_c = 1
        #pos
        x = res_cons_x['x'][0+8*i:8+8*i]
        y = res_cons_y['x'][0+8*i:8+8*i]
        z = res_cons_z['x'][0+8*i:8+8*i]
        #velocity
        dx = [0, 7*x[0], 6*x[1], 5*x[2], 4*x[3], 3*x[4], 2*x[5], x[6]]
        dy = [0, 7*y[0], 6*y[1], 5*y[2], 4*y[3], 3*y[4], 2*y[5], y[6]]
        dz = [0, 7*z[0], 6*z[1], 5*z[2], 4*z[3], 3*z[4], 2*z[5], z[6]]
        #acceleration
        ddx = [0, 0, 42*x[0], 30*x[1], 20*x[2], 12*x[3], 6*x[4], 2*x[5]]
        ddy = [0, 0, 42*y[0], 30*y[1], 20*y[2], 12*y[3], 6*y[4], 2*y[5]]
        ddz = [0, 0, 42*z[0], 30*z[1], 20*z[2], 12*z[3], 6*z[4], 2*z[5]]
        #jerk
        dddx = [0, 0, 0, 210*x[0], 120*x[1], 60*x[2], 24*x[3], 6*x[4]]
        dddy = [0, 0, 0, 210*y[0], 120*y[1], 60*y[2], 24*y[3], 6*y[4]]
        dddz = [0, 0, 0, 210*z[0], 120*z[1], 60*z[2], 24*z[3], 6*z[4]]

        #a = symbols('a')
        #pos = x
        p = x[0]*time_math**7 + x[1]*time_math**6 + x[2]*time_math**5 + x[3]*time_math**4 + x[4]*time_math**3 + x[5]*time_math**2 + x[6]*time_math + x[7]
        #p = 2
        #p = np.ones(len(time_math))*2

        #x = p*cos(a)/(1+sin(a)**2)
        #y = p*cos(a)*sin(a)/(1+sin(a)**2)

        #x_pos = float(x.subs(a, i))
        #x_vel= float(diff(x, a).subs(a, i))
        #x_acc = float(diff(x, a, 2).subs(a, i))
        #x_jerk = float(diff(x, a, 3).subs(a, i))

        #y_pos = float(y.subs(a, i))
        #y_vel = float(diff(y, a).subs(a, i))
        #y_acc = float(diff(y, a, 2).subs(a, i))
        #y_jerk = float(diff(y, a, 3).subs(a, i))

        #z_pos = 0
        #z_vel = 0
        #z_acc = 0
        #z_jerk = 0

        #x_pos = np.polyval(np.array(x), time_math)*np.cos(time_plot)
        #x_pos = np.polyval(np.array(x), time_math)*np.cos(time_plot)/(1+np.sin(time_plot)**2)
        x_pos = p*np.cos(time_plot)/(1+np.sin(time_plot)**2)
        #y_pos = np.polyval(np.array(x), time_math)*np.sin(time_plot)
        y_pos = p*np.cos(time_plot)*np.sin(time_plot)/(1+np.sin(time_plot)**2)
        #x_pos = np.polyval(np.array(x), time_math)
        #x_vel = np.polyval(np.array(dx), time_math)
        #x_acc = np.polyval(np.array(ddx), time_math)
        #x_jerk = np.polyval(np.array(dddx), time_math)

        #y_pos = np.polyval(np.array(y), time_math)*np.sin(w_c*time_math)
        #y_pos = np.polyval(np.array(y), time_math)
        #y_vel = np.polyval(np.array(dy), time_math)
        #y_acc = np.polyval(np.array(ddy), time_math)
        #y_jerk = np.polyval(np.array(dddy), time_math)

        z_pos = np.polyval(np.array(z), time_math)
        #z_vel = np.polyval(np.array(dz), time_math)
        #z_acc = np.polyval(np.array(ddz), time_math)
        #z_jerk = np.polyval(np.array(dddz), time_math)

        #plt.figure(1)
        #plt.plot(time_plot, x_pos, 'r', label = 'x')
        #plt.plot(time_plot, y_pos, 'g', label = 'y')
        #plt.plot(time_plot, z_pos, 'b', label = 'z')
        #plt.legend
        #plt.title('position')
        '''
        plt.figure(2)
        plt.plot(time_plot, x_vel, 'r')
        plt.plot(time_plot, y_vel, 'g')
        plt.plot(time_plot, z_vel, 'b')
        plt.title('velocity')
        plt.figure(3)
        plt.plot(time_plot, x_acc, 'r')
        plt.plot(time_plot, y_acc, 'g')
        plt.plot(time_plot, z_acc, 'b')
        plt.title('acceleration')
        plt.figure(4)
        plt.plot(time_plot, x_jerk, 'r')
        plt.plot(time_plot, y_jerk, 'g')
        plt.plot(time_plot, z_jerk, 'b')
        plt.title('jerk')
        '''
        plt.figure(4)
        plt.plot(time_plot, p)

        plt.figure(5)
        ax.plot3D(x_pos, y_pos, z_pos)
        plt.title('trajectory')

        cont_time = cont_time + T
    plt.show()
