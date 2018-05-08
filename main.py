#! /usr/bin/env py -3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:41:32 2018

@author: Simon, Zyanya, Wessel
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
start_time = time.time()


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
rho = 1.225
a_w = 0.2
TSR = 6
omega = TSR * u_inf /R

t_steps = 100
t=np.linspace(0., 30., t_steps)
single_wake = np.zeros((t_steps, (N+1), 3))

def induced_velocity(point, point_1, point_2, circulation, r_vortex = 1e-5):
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
    

def circcalc(alpha, V, c):
    """
    Very basic circulation calculator with lots of inputs and 1 output
    """
    circ = 0.5*c*V*polarreader(alpha, alphalist, cllist, cdlist)[0]
    return circ

   
"""
Initialising the blade
"""
#Cosine mapping
mapping = 0.5*(1-np.cos(np.linspace(0, np.pi, num=N+1)))


ends = map_values(mapping, 0,1, 0.2*R, R)
elements = middle_vals(ends)
mu = map_values(elements, 0.2*R, R, 0.2, 1)

#calculating the blade coordinates
controlpoints_single = np.zeros((N, 3))
controlpoints = [np.copy(controlpoints_single) for x in range(N_blades)]

"""
Initialize matrix of ring coordinates
"""
#Initilise the z component
single_wake[:,:,2] = np.transpose(np.broadcast_to(t*u_inf*(1-a_w), (N+1, t_steps)))
#making the list in which the wakes are initialised
Wake = [np.copy(single_wake) for x in range(N_blades)]


#Constructing the wake matrix for a single blade

for i in range(N_blades):
    rot = 2*np.pi/N_blades*i
    om_x = np.cos(omega*t+rot)
    om_y = np.sin(omega*t+rot)
    controlpoints[i][:,0] = elements*np.cos(rot) - np.sin(rot)*0.5*chord(mu)
    controlpoints[i][:,1] = elements*np.sin(rot)+np.cos(rot)*0.5*chord(mu)
    Wake[i][:,:,0] = np.matmul(om_x.reshape((-1,1)),ends.reshape((1,-1))) #x
    Wake[i][:,:,1] = np.matmul(om_y.reshape((-1,1)),ends.reshape((1,-1))) #y
    
     
"""
Initialize u, v, w matrices with circulation/ring strength set to unity
"""


MatrixU = np.zeros((N_blades*N, N_blades*N))
MatrixV = np.zeros((N_blades*N, N_blades*N))
MatrixW = np.zeros((N_blades*N, N_blades*N))

for icp in range(N_blades*N):
    icpn = icp%N
    for jring in range(N_blades*N):
        i = int(jring/N)
        jringn = jring%N
        ring = np.concatenate((Wake[i][:, jringn,:], Wake[i][::-1, jringn+1,:]))
        ind_vel = unit_induced_velocity_calc(controlpoints[i][icpn], ring)
        MatrixU[icp][jring] = ind_vel[0]
        MatrixV[icp][jring] = ind_vel[1]
        MatrixW[icp][jring] = ind_vel[2]
        
        
"""
Calculate circulations for U, V, W unit circulation matrices
"""
ulist = N_blades*N*[0.]
vlist = N_blades*N*[0.]
wlist = N_blades*N*[0.]
pitch = np.radians(2)

diff_u = 1
diff_v = 1
diff_w = 1

n = 0
precision = 1e-18
nmax = 50

print("--- Time to init is %s seconds ---" % (time.time() - start_time))

while (diff_u>precision and diff_v>precision and diff_w>precision and n<nmax):
    u_old = ulist
    v_old = vlist
    w_old = wlist
    
    gammalist = []
    gammalist_nondim = []
    for z in range(N_blades*N):
        zn = z%N
        i = int(z/N)
            
        c = chord(mu[zn])
        tw = twist(mu[zn])
        
        r1 = np.array([0,0,-1/elements[zn]])
        r2 = controlpoints[i][zn]
        n_azim = np.cross(r1, r2)

        
        V_ax = u_inf + wlist[z]
        V_tan = omega*elements[zn] + np.dot(np.array([ulist[z], vlist[z], V_ax]), n_azim)
        V_p = np.sqrt(V_tan**2 +V_ax**2)
        ratio = V_ax / V_tan
        
        alpha = np.arctan(ratio)-tw+pitch
        circ = circcalc(alpha*180./np.pi, V_p, c)
        gammalist.append(circ)
        gammalist_nondim.append(circ/((u_inf**2)/(1*np.pi*omega)))
    
    
    ulist = np.matmul(MatrixU,  np.array(gammalist)) 
    vlist = np.matmul(MatrixV,  np.array(gammalist)) 
    wlist = np.matmul(MatrixW,  np.array(gammalist))
    
    diff_u_list = np.abs(u_old - ulist)
    diff_v_list = np.abs(v_old - vlist)
    diff_w_list = np.abs(w_old - wlist)
    
    diff_u = np.amax(diff_u_list)
    diff_v = np.amax(diff_v_list)
    diff_w = np.amax(diff_w_list)
    
    n+=1
"""  
plot circulations, set nondim=True for a nondimensionalized version of the plot
"""
def circplot(nondim):
    if nondim == True:
        dimfactor = np.pi*u_inf**2 / (N_blades*omega)
        for i in range(len(gammalist)):
            gammalist[i] = gammalist[i]/dimfactor
        plt.plot(mu, gammalist[:20])
        plt.ylim(0, 1.1)
    else:
        plt.plot(mu, gammalist[:20])
        plt.ylim((0, 1.9))
    
    plt.xlabel("r/R")
    plt.ylabel("Circulation \gamma")
    plt.show()




"""
3D plot testr
"""
"""
fact = 1.0
fig = plt.figure(num=None, figsize=(50, 50), dpi=30, facecolor='w', edgecolor='k')

ax = plt.axes(projection='3d')
ax.set_zlim(-50, 50)
ax.set_xlim(-50, 50)
ax.set_ylim(0, 140*fact)
ax.view_init(0, -20)
for i in range(int(t_steps*fact)):
    x1 = x
    y1 = y
    z1 = z
    x = single_wake[i:i+2,:,0]
    y = single_wake[i:i+2,:,1]
    z = single_wake[i:i+2,:,2]
    ax.plot_wireframe(x, z, y)
    plt.pause(.00001)
 

plt.show()
"""
print("--- Total runtime is %s seconds ---" % (time.time() - start_time))
