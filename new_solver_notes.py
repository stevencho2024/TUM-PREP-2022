import os
import time

import numpy as np
import mpctools as mpc
import pdb
import cmath
from scipy import optimize as opt
from scipy.linalg import block_diag
from collections import deque
np.set_printoptions(precision=4, suppress=True)

def dynamics(x, u):
    # return next state starting from current state x if input u is applied
    # check what args_mpc_functions 'sys' (in the student's code) does, probably is what we want here
    # if you need other parameters, add them! For example the current curvature or so...
    

def costf(U, N, x0, xref, u_1, Q, R, S, m):
    U0 = U.reshape(-1,1)
    x = [x0]
    for k in range(N):
        x.append(dynamics(x[-1],U0[k*m:k*(m+1), :])) # state sequence
    u_last = u_1    # state applied during previous step
    cost = 0 # initialize overall cost with scalar value 0
    for k in range(N):
        dx = x[k]-xref
        u0 = U0[k*m:(k+1)*m, :]
        du = u0-u_last
        cost += (dx.T.dot(Q)).dot(dx)+(u0.T.dot(R)).dot(u0)+(du.T.dot(S)).dot(du)
        u_last = u0
    return cost[0][0] # return scalar value representing overall cost associated with input sequence U
                      # the [0][0] thing is only needed to avoid warnings and errors, but cost is a scalar thing

def cstrf(U, x0, N, qx, qy, qt):
    U0 = U.reshape(-1,1)
    x = [x0]
    for k in range(N):
        x.append(dynamics(x[-1],U0[k*m:k*(m+1), :]))
    cineq = []  # initialize list of constraints
    for k in range(N):
        cineq.append(qx[k]*x[k][0]+qy[k]*x[k][1]+qt[k])
    return np.asarray(cineq) # return constraints in array form (instead of list)


""""
    The following must be added where casadi is used
"""
cstr = [{'type': 'ineq', 'fun': lambda U: cstrf(U, x0, N, qx, qy, qt)}]
# U0 must be the guess of the input sequence (m*N x 1 array)
# x0 must be a 4 dim array with the initial condition of the EV (only the four features we are concerned with)
res = opt.minimize(costf, U0, args=(N, x0, xref, u_1, Q, R, S, m),
            method="SLSQP", constraints=cstr)
U = res.x.reshape(-1, 1)


# update U0 guess
...

# update u previous
...

# apply in Carla only first element of input sequence