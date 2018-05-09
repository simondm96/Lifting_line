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
rho = 1.225
a_w = 0.1
TSR = 6
omega = TSR * u_inf /R

t_steps = 150
t=np.linspace(0., 30, t_steps)
single_wake = np.zeros((t_steps, (N+1), 3))

polar = np.genfromtxt('polar_DU95W180.csv', delimiter = ',', skip_header = 2)

def induced_velocity(point, point_1, point_2, circulation, r_vortex = 0.5):
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
        twist       = ndarray, the twist for the section(s). If section is a
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



def polarreader(alpha, polar):
    """
    instructions: call polarreader with the required alpha and the alphalist & cllist
    will return interpolated cl value
    """
    cl = np.interp(alpha, polar[:,0], polar[:,1])
    cd = np.interp(alpha, polar[:,0], polar[:,2])
    return cl, cd
    

def circcalc(cl, V, c):
    """
    Very basic circulation calculator with lots of inputs and 1 output
    """
    circ = 0.5*c*V*cl
    return circ

   
"""
Initialising the blade
"""
#Cosine mapping
#mapping = 0.5*(1-np.cos(np.linspace(0, np.pi, num=N+1)))
#Normal mapping
mapping = np.linspace(0,1,num=N+1)


ends = map_values(mapping, 0,1, 0.2*R, R)
elements = middle_vals(ends)
mu = map_values(elements, 0.2*R, R, 0.2, 1)
mu_ends = map_values(ends, 0.2*R, R, 0.2, 1)

#calculating the blade coordinates
controlpoints = np.zeros((N, 3))
controlpoints[:,0] = elements
#controlpoints[:,1] = 0.5*chord(mu)*np.cos(twist(mu)+pitch)
#controlpoints[:,2] = 0.5*chord(mu)*np.sin(twist(mu)+pitch)

#calculate the starting locations of the vortex filaments in the flow
vortex_start_y = chord(mu_ends)*np.cos(twist(mu_ends)+pitch)
vortex_start_z = chord(mu_ends)*np.sin(twist(mu_ends)+pitch)

"""
Initialize matrix of ring coordinates
"""
#Initilise the z component
single_wake[1:,:,2] = vortex_start_z+np.transpose(np.broadcast_to(t[:-1]*u_inf*(1 - a_w), (N + 1, t_steps-1)))
single_wake[0,:,2] = 0
#making the list in which the wakes are initialised
Wake = [np.copy(single_wake) for x in range(N_blades)]


#Constructing the wake matrix for a single blade

for i in range(N_blades):
    rot = 2*np.pi/N_blades*i
    om_x = np.cos(omega*t[:-1] + rot)
    om_y = np.sin(omega*t[:-1] + rot)
    Wake[i][1:,:,0] = np.matmul(om_x.reshape((-1,1)),ends.reshape((1,-1))) - np.matmul(om_y.reshape((-1,1)),chord(mu_ends).reshape((1,-1)))#x
    Wake[i][1:,:,1] = np.matmul(om_y.reshape((-1,1)),ends.reshape((1,-1))) + np.matmul(om_x.reshape((-1,1)),vortex_start_y.reshape((1,-1)))#y
    Wake[i][0,:,0] = ends*np.cos(rot)
    Wake[i][0,:,1] = ends*np.sin(rot)
    
     
"""
Initialize u, v, w matrices with circulation/ring strength set to unity
"""


MatrixU = np.zeros((N, N_blades*N))
MatrixV = np.zeros((N, N_blades*N))
MatrixW = np.zeros((N, N_blades*N))

ringlist = []
for icp in range(N):
    for jring in range(N_blades*N):
        i = int(jring/N)
        jringn = jring%N
        ring = np.concatenate((Wake[i][:, jringn,:], Wake[i][::-1, jringn+1,:]))
        ringlist.append(ring)
        ind_vel = unit_induced_velocity_calc(controlpoints[icp], ring)
        MatrixU[icp,jring] = ind_vel[0]
        MatrixV[icp,jring] = ind_vel[1]
        MatrixW[icp,jring] = ind_vel[2]
        
        
"""
Calculate circulations for U, V, W unit circulation matrices
"""
ulist = N*[0.]
vlist = N*[0.]
wlist = N*[0.]


diff_u = 1
diff_v = 1
diff_w = 1

n = 0
precision = 1e-5
nmax = 1000

print("--- Time to init is %s seconds ---" % (time.time() - start_time))



while (diff_u>precision and diff_v>precision and diff_w>precision) and n<nmax:
    u_old = ulist
    v_old = vlist
    w_old = wlist
    
    fazlist = []
    faxlist = []
    gammalist = []
    alphalist = []
    vellist = []
    for z in range(N):            
        c = chord(mu[z])
        tw = twist(mu[z])
        
        r1 = np.array([0,0,-1/elements[z]])
        r2 = controlpoints[z]
        n_azim = np.cross(r1, r2)/np.linalg.norm(np.cross(r1, r2))

        
        V_ax = u_inf + wlist[z]
        V_tan = omega*elements[z] + np.dot(np.array([ulist[z], vlist[z], V_ax]), n_azim)
        V_p = np.sqrt(V_tan**2 + V_ax**2)
        ratio = V_ax / V_tan
        
        alpha = np.arctan(ratio)- tw + pitch
        
          
        polar_val = polarreader(alpha*180./np.pi, polar)
#        print(polar_val)
        circ = circcalc(polar_val[0], V_p, c)
        alphalist.append(alpha)
        gammalist.append(circ)      
        L = 0.5*c*rho*(V_p**2)*polar_val[0]
        D = 0.5*c*rho*(V_p**2)*polar_val[1]
        f_azim = L*(V_ax/V_p) - D*(V_tan/V_p)
        f_axial = L*(V_tan/V_p) + D*(V_ax/V_p)
        fazlist.append(f_azim)
        faxlist.append(f_axial)
        vellist.append(V_p)
        
    
    gammalist_mat = np.tile(np.array(gammalist), N_blades)
    
    ulist = np.matmul(MatrixU,  np.array(gammalist_mat)) 
    vlist = np.matmul(MatrixV,  np.array(gammalist_mat)) 
    wlist = np.matmul(MatrixW,  np.array(gammalist_mat))
    
    diff_u_list = np.abs(u_old - ulist)
    diff_v_list = np.abs(v_old - vlist)
    diff_w_list = np.abs(w_old - wlist)
    
    diff_u = np.amax(diff_u_list)
    diff_v = np.amax(diff_v_list)
    diff_w = np.amax(diff_w_list)
    
    n+=1

    
#calculate induction factors
a = -wlist/u_inf
aprime = np.sqrt(ulist**2+vlist**2)/omega/elements

"""  
plot circulations, set nondim=True for a nondimensionalized version of the plot
"""
def circplot(nondim):
    if nondim == True:
        dimfactor = np.pi*u_inf**2 / (N_blades*omega)
        for i in range(len(gammalist)):
            gammalist[i] = gammalist[i]/dimfactor
        plt.plot(mu, gammalist[:20])
    else:
        plt.plot(mu, gammalist[:20])
        
    plt.ylim((0, max(gammalist)*1.1))
    plt.xlabel("r/R")
    plt.ylabel(r"Circulation $\Gamma$")
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
