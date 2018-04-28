#! /usr/bin/env py -3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:41:32 2018

@author: Simon, Zyanya, Wessel
"""

import numpy as np


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
t=np.linspace(0., 20., 100)

def Disc_Geo_fixed(a_w, U_inf, R, omega, t):

    x_w=t*U_inf*(1-a_w)
    y_w=R*np.sin(omega*t)
    z_w=R*np.cos(omega*t)
    return x_w, y_w, z_w

def induced_velocity(point, point_1, point_2, circulation, r_vortex = 1e-10):
    """
    Calclates the induced velocity of a straight vortex element between point_1 and point_2 on point
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
       

"""
instructions: call polarreader with the required alpha and the alphalist & cllist
will return interpolated cl value
"""
def polarreader(alpha, alphalist, cllist, cdlist):
    cl = np.interp(alpha, alphalist, cllist)
    cd = np.interp(alpha, alphalist, cdlist)
    return cl, cd
    
"""
Very basic circulation calculator with lots of inputs and 1 output
"""
def circcalc(alpha, V, rho, c):
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

#single_vortex = Disc_Geo_fixed(a_w, u_inf, ends, omega, t)
     
"""
Initialize u, v, w matrices with circulation/ring strength set to unity
"""

  
#MatrixU = np.zeros(N, N)
#MatrixV = np.zeros(N, N)
#MatrixW = np.zeros(N, N)
#for icp in range(len(controlpoints)):
#    for jring in range(len(controlpoints)):
#        MatrixU[icp][jring] = induced_velocity(
#        MatrixV[icp][jring] = induced_velocity(
#        MatrixW[icp][jring] = induced_velocity(
        