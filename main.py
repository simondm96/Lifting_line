#! /usr/bin/env py -3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:41:32 2018

@author: Simon, Zyanya, Wessel
"""

import numpy as np
import csv

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
preader = csv.reader(polar, delimiter = ',')
alphalist = []
cllist = []
count = 0;
for row in preader:
    if count<2:
        #print 'header'
        count = count +1
    else:
        alphalist.append(float(row[0]))
        cllist.append(float(row[1]))

"""
instructions: call polarreader with the required alpha and the alphalist & cllist
will return interpolated cl value
"""
def polarreader(alpha, alphalist, cllist):
    cl = np.interp(alpha, alphalist, cllist)
    return cl
    
"""
Very basic circulation calculator with lots of inputs and 1 output
"""
def circcalc(alpha, V, rho, c):
    circ = 0.5*c*V*polarreader(alpha)
    return circ
    
    

    

    