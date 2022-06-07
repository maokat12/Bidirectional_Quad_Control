import numpy as np
#import sympy as sp

from sympy import cos, sin, symbols, diff

t, w_c = symbols('t, w_c')
p = t**7 + t**6 + t**5 + t**4 + t**3 + t**2 + t + 1
x_pos = p*cos(w_c*t)/(1+sin(w_c*t)**2)
x_vel= diff(x_pos, t)
x_acc = diff(x_pos, t, 2)
x_jerk = diff(x_pos, t, 3)
print("x_pos : {}\n".format(x_pos))
#print("x_vel : {}\n".format(x_vel))
#print("x_acc : {}\n".format(x_acc))
print("x_jerk : {}\n\n".format(x_jerk))

y_pos = p*cos(w_c*t)*sin(w_c*t)/(1+sin(w_c*t)**2)
y_vel= diff(y_pos, t)
y_acc = diff(y_pos, t, 2)
y_jerk = diff(y_pos, t, 3)
print("y_pos : {}\n".format(y_pos))
#print("y_vel : {}\n".format(y_vel))
#print("y_acc : {}\n".format(y_acc))
print("y_jerk : {}".format(y_jerk))

