import numpy as np
from scipy.spatial.transform import Rotation as Rotation
from pyquaternion import Quaternion

abc = [0.27, 0.52, 0.29]
abc = abc/np.linalg.norm(abc)
print('abc', abc)
[a, b, c] = abc

e1 = [1, 0, 0]
e2 = [0, 1, 0]
e3 = [0, 0, 1]

yaw_n = 243
yaw_n = np.radians(yaw_n)
yaw_s = 2*np.arctan2(a, b) + yaw_n

print('yaw_N', yaw_n)
print('yaw_s', yaw_s)

q_abc = (1/np.sqrt(2*(1+c))) * np.array([1+c, -b, a, 0])
q_yaw_n = np.array([np.cos(yaw_n/2), 0, 0, np.sin(yaw_n/2)])
q_abc = Quaternion(np.array(q_abc))
q_yaw_n = Quaternion(np.array(q_yaw_n))

q_abc_bar = (1/np.sqrt(2*(1-c))) * np.array([-b, 1-c, 0, a])
q_yaw_s = np.array([np.cos(yaw_s/2), 0, 0, np.sin(yaw_s/2)])
q_abc_bar = Quaternion(np.array(q_abc_bar))
q_yaw_s = Quaternion(np.array(q_yaw_s))

q_des_n = q_abc * q_yaw_n
q_des_s = q_abc_bar * q_yaw_s

print('\nNorth Chart')
b3_int_n = q_abc.rotate(e3)
b3_fin_n = q_des_n.rotate(e3)
b2_int_n = q_abc.rotate(e2)
b2_fin_n = q_des_n.rotate(e2)
b1_int_n = q_abc.rotate(e1)
b1_fin_n = q_des_n.rotate(e1)

print('b1_int', b1_int_n)
print('b2_int', b2_int_n)
print('b3_int', b3_int_n)

print("\nb1_fin", b1_fin_n)
print("b2_fin", b2_fin_n)
print("b3_fin", b3_fin_n)

print('\nSouth Chart')
b3_int_s = q_abc_bar.rotate(e3)
b3_fin_s = q_des_s.rotate(e3)
b2_int_s = q_abc_bar.rotate(e2)
b2_fin_s = q_des_s.rotate(e2)
b1_int_s = q_abc_bar.rotate(e1)
b1_fin_s= q_des_s.rotate(e1)

print('b1_int', b1_int_s)
print("b2_int", b2_int_s)
print("b3_int", b3_int_s)

print("\nb1_fin", b1_fin_s)
print('b2_fin', b2_fin_s)
print('b3_fin', b3_fin_s)