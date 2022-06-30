import numpy as np
from scipy.spatial.transform import Rotation as Rotation
from pyquaternion import Quaternion

g = 9.81 #m/s
sign_o = 1
m = 0.03 #kg
parse_csv = True

#input
r_ddot = np.array([[5], [2], [5]]) #acceleration
r_dddot = [[3],[1],[7]] #jerk
yaw = 0
yaw_n = np.radians(yaw)

#calculate force, abc
F = m*(r_ddot + np.array([[0], [0], [g]]))
F_dot = m*np.array(r_dddot)
sign_f = 1 if F[2][0] > 0 else -1

#abc
abc = F/np.linalg.norm(F) * sign_o*sign_f
[a, b, c] = abc
a = abc[0][0]
b = abc[1][0]
c = abc[2][0]

#abc dot
abc_dot = sign_f*sign_o * (F.T@F*np.identity(3) - F@F.T)/np.linalg.norm(F)**3 @ F_dot
[a_dot, b_dot, c_dot] = abc_dot
print('abc', abc)
print('abc_dot', [a_dot, b_dot, c_dot])

#north chart
w1_n = np.sin(yaw_n)*a_dot - np.cos(yaw_n)*b_dot - (a*np.sin(yaw_n) - b*np.cos(yaw_n)) * c_dot/(1+c)
w2_n = np.cos(yaw_n)*a_dot + np.sin(yaw_n)*b_dot - (a*np.cos(yaw_n) + b*np.sin(yaw_n)) * c_dot/(1+c)
w3_n = 0

#south chart
yaw_s = 2*np.arctan2(a, b) + yaw_n
w1_s = np.sin(yaw_s)*a_dot + np.cos(yaw_s)*b_dot - (a*np.sin(yaw_s) + b*np.cos(yaw_s)) * c_dot/(c-1)
w2_s = np.cos(yaw_s)*a_dot - np.sin(yaw_s)*b_dot - (a*np.cos(yaw_s) - b*np.sin(yaw_s)) * c_dot/(c-1)
w3_s = 0

print(np.sin(yaw_s)*a_dot + np.cos(yaw_s)*b_dot)
print(((a*np.sin(yaw_s) + b*np.cos(yaw_s))) * c_dot/(1-c))

#outputs
print("\nNorth Chart")
print('w1: ', w1_n)
print('w2: ', w2_n)
print('w3: ', w3_n)

print("\nSouth Chart")
print('w1: ', w1_s)
print('w2: ', w2_s)
print('w3: ', w3_s)


test = np.loadtxt("b3.csv",delimiter=',')
print(test.shape)

if parse_csv:
    count = 0
    failure = 0
    for row in test:
        r_ddot = row[0:3]#acceleration
        r_ddot = np.array([[r_ddot[0]], [r_ddot[1]], [r_ddot[2]]])
        r_dddot = row[3:6] #jerk
        r_dddot = np.array([[r_dddot[0]], [r_dddot[1]], [r_dddot[2]]])

        yaw = 0
        yaw_n = np.radians(yaw)

        F = m*(r_ddot + np.array([[0], [0], [g]]))
        F_dot = r_dddot
        sign_f = 1 if F[2][0] > 0 else -1

        #abc
        abc = F/np.linalg.norm(F) * sign_o*sign_f
        [a, b, c] = abc
        a = abc[0][0]
        b = abc[1][0]
        c = abc[2][0]

        #abc dot
        abc_dot = sign_f*sign_o * (F.T@F*np.identity(3) - F@F.T)/np.linalg.norm(F)**3 @ F_dot
        [a_dot, b_dot, c_dot] = abc_dot

        if (1 - np.abs(c)) > 0.001: #make sure c != 1, -1
            #north chart
            w1_n = np.sin(yaw_n)*a_dot - np.cos(yaw_n)*b_dot - (a*np.sin(yaw_n) - b*np.cos(yaw_n)) * c_dot/(c+1)
            w2_n = np.cos(yaw_n)*a_dot + np.sin(yaw_n)*b_dot - (a*np.cos(yaw_n) + b*np.sin(yaw_n)) * c_dot/(c+1)
            w3_n = 0

            #south chart
            yaw_s = 2*np.arctan2(a, b) + yaw_n
            w1_s = np.sin(yaw_s)*a_dot + np.cos(yaw_s)*b_dot - (a*np.sin(yaw_s) + b*np.cos(yaw_s)) * c_dot/(c-1)
            w2_s = np.cos(yaw_s)*a_dot - np.sin(yaw_s)*b_dot - (a*np.cos(yaw_s) - b*np.sin(yaw_s)) * c_dot/(c-1)
            w3_s = 0

            if (w1_n - w1_s) > 0.001 or (w2_n - w2_s) > 0.001:
                failure = failure+1
                print(count)
                print('c', c)
                print('abc', abc)
                print('abc_dot', abc_dot)
                print(w1_n)
                print(w1_s)
                print(w2_n)
                print(w2_s)

        count = count + 1

print(count)
print(failure)
