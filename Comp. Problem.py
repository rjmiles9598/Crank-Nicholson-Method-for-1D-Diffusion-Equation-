# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 13:53:15 2020

@author: rjmil
"""

import numpy as np
import matplotlib.pyplot as plt

# Write a computer code to solve by the Crank-Nicolson method over the time 
# interval 0 ≤ t ≤ T the one-dimensional diffusion equation

#############################
# Discretizing our x-domain
# Variables:
# L = sixe of x-domain
# J = # of steps in x domain
# dx = step size in x domain

#############################

L = 1
J = 10
dx = float(L)/float(J-1)
x_grid = np.array([J*dx for j in range(J)])
print(x_grid)
#############################
# Discretizing our time-domain
#############################

T = 100
N = 1000
dt = float(T)/float(N-1)
t_grid = np.array([N*dt for n in range(N)])

#############################
# Set up our tri diagonal solver and calculator
#############################

def calc_matrix_tridiag(xs,K):
    N = len(xs)-2
    invdeltaxsq = 1./((xs[2]-xs[1])**2.)
    super_diag = np.ones([N-1])*invdeltaxsq
    sub_diag = np.ones([N-1])*invdeltaxsq
    diag = np.ones([N])*(-2*invdeltaxsq-K*K)
    return diag, super_diag, sub_diag

def tridiag_solver(diag, super_diag, sub_diag, rhs):
    N = len(diag)
    ds = np.zeros([N])
    xs = np.zeros([N])
    for i in range (1,N):
        w = sub_diag[i]/super_diag[i]
        diag[i] = diag[i]-w*super_diag[i]
        ds[i] = diag[i] - w*sub_diag[i]
    xs[N-1] = ds[N-1]/diag[N-1]
    for i in range (N-1,0,-1):
        xs[i] = (ds[i]-super_diag[i]*xs[i+1])/(diag[i])



