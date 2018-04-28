#! /usr/bin/env py -3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:41:32 2018

@author: Simon, Zyanya, Wessel
"""

import numpy as np
import matplotlib.pyplot as plt



"""
Parameters from BEM assignment
"""
N = 20
u_inf = 10.
N_blades = 3
hubrR = 0.2
R = 50.
pitch = 2*np.pi/180
a = 0.3 #starting value
aprime = 0.0 #starting value
rho = 1.225
a_w = 0.3
TSR = 6
omega = TSR * u_inf /R
t_steps = 100
t=np.linspace(0., 20., t_steps)
om_x = np.cos(omega*t)
om_y = np.sin(omega*t)

single_wake = np.zeros((t_steps, N, 3))


"""
Initialize coordlist matrix of rotor coordinates
"""
coordlist = np.zeros((1,3))


for tn in range(3):
    theta = 90+tn*120
    r=0
    for N_el in range(N):
        dr = R/N
        r = r+dr
        x = r*np.cos(np.radians(theta))
        y = r*np.sin(np.radians(theta))
        z = 0
        tempmat = [[x,y,z]]
        coordlist = np.vstack([coordlist, tempmat])

"""
Initialize matrix of ring coordinates
"""




def induced_velocity(point, point_1, point_2, circulation, r_vortex = 1e-10):
    """
    Calculates the induced velocity of a straight vortex element between point_1 and point_2 on point
    """
    point = np.asarray(point)
    point_1 = np.asarray(point_1)
    point_2 = np.asarray(point_2)
    
    r1 = point_1-point
    r2 = point_2-point
    r0 = r1-r2
    
    cross_r = np.cross(r1, r2)
    cross_squared = np.linalg.norm(cross_r)**2
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    if r1_norm<r_vortex or r2_norm<r_vortex or cross_squared<r_vortex:
        v_ind = np.array([0,0,0])
    else:
        vector_1 = circulation/(4*np.pi)*cross_r/cross_squared
        vector_2 = np.inner(r0, r1/r1_norm-r2/r2_norm)
        v_ind = vector_1*vector_2
    return v_ind
  
def middle_vals(data):
    return np.diff(data)/2+np.delete(data, -1)


def twist(section):
    """
    Generates a twist distrubution
    
    Input:
        section     = ndarray, the section(s) for which the twist should be
                      calculated, normalised to the radius of the rotor
                      
    Output:
        twist       = ndarray, the twist for the section(s). If sections is a
                      float, returns a float
    """
    return 14*(1-section)*np.pi/180.


def chord(section):
    """
    Generates a chord distribution
    
    Input:
        section     = ndarray, the section(s) for which the chord should be
                      calculated, normalised to the radius of the rotor
                      
    Output:
        twist       = ndarray, the chord for the section(s). If section is a
                      float, returns a float
    """
    return (3*(1-section)+1)


def map_values(data, x_start1, x_end1, x_start2, x_end2):
    """
    Maps data with boundaries x_start1 and x_end1 to x_start2 and x_start2
    """
    return x_start2 + (data-x_start1)*(x_end2-x_start2)/(x_end1-x_start1)

def unit_induced_velocity_calc(point, ring):
    """
    Calculates the induced velocity on a control point by a vortex ring
    """
    ind_vel = np.array([0,0,0], dtype = 'float64')
    for i in range(len(ring)):
        vel = induced_velocity(point, ring[i-1], ring[i], 1)
        ind_vel += vel
    return ind_vel

#reads airfoil polar data
polar = open('polar_DU95W180.csv', 'rb')
preader = np.genfromtxt(polar, delimiter = ',', skip_header = 2)
alphalist = preader[0,:]
cllist = preader[1,:]
cdlist = preader[2,:]
       


def polarreader(alpha, alphalist, cllist, cdlist):
    """
    instructions: call polarreader with the required alpha and the alphalist & cllist
    will return interpolated cl value
    """
    cl = np.interp(alpha, alphalist, cllist)
    cd = np.interp(alpha, alphalist, cdlist)
    return cl, cd
    

def circcalc(alpha, V, rho, c):
    """
    Very basic circulation calculator with lots of inputs and 1 output
    """
    circ = 0.5*c*V*polarreader(alpha, alphalist, cllist, cdlist)[0]
    return circ

   
"""
Initialising the blade
"""

#Cosine mapping
mapping = 0.5*(1-np.cos(np.linspace(0, np.pi, num=N)))

ends = map_values(mapping, 0,1, 0.2*R, R)
elements = middle_vals(ends)

#calculating the blade coordinates
controlpoints = np.zeros((N-1, 3))
controlpoints[:,0] = elements

#Constructing the matrix
single_wake[:,:,2] = np.transpose(np.broadcast_to(t*u_inf*(1-a_w), (N, t_steps)))
single_wake[:,:,0] = om_x.reshape((-1,1)) @ ends.reshape((1,-1))
single_wake[:,:,1] = om_y.reshape((-1,1)) @ ends.reshape((1,-1))
     
"""
Initialize u, v, w matrices with circulation/ring strength set to unity
"""

MatrixU = np.zeros((N-1, t_steps-1))
MatrixV = np.zeros((N-1, t_steps-1))
MatrixW = np.zeros((N-1, t_steps-1))
for icp in range(N-1):
    for jring in range(t_steps-1):
        ring = [single_wake[jring, icp,:], single_wake[jring+1, icp,:], single_wake[jring+1, icp+1,:], single_wake[jring, icp+1,:]]
        ind_vel = unit_induced_velocity_calc(controlpoints[icp], ring)
        MatrixU[icp][jring] = ind_vel[0]
        MatrixV[icp][jring] = ind_vel[1]
        MatrixW[icp][jring] = ind_vel[2]
