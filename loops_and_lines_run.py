"""
Created: October 2017
Authors: Tyler A. Engstrom (initiated code), Mahesh C. Gandikota, J. M. Schwarz
Description: Runs the loops_and_lines code generating and deforming a 2D semiflexible polymer network with area-preserving inclusions 
"""
"""
Nx, Ny must be even
orient must be zero
"""

import matplotlib.pyplot as plt  
import numpy as np
from scipy.optimize import minimize
from loops_and_lines import RandomSpringNetwork
import sys


### default parameters

orient       = 0
Nx           = 8
Ny           = 8
N            = Nx*Ny
K_hookean    = 10.0                #hookean spring constant
K_sfp        = 1.0                 #sfp angular spring constant
K_cross      = 0*pow(10,-1)        #crosslink angular spring constant
num_of_incl  = 35                #number of inclusions/cells
K_area       = pow(10,3)         #area inclusion spring constant
frac_array   = np.array([1.0]) #probability of spring occupation
no_frac      = len(frac_array)       
maxStrain    = 0.1
runs         = 1
maxIter      = 100
tolerance    = 1e-5
Nsteps       = np.int(maxStrain*100)
myMovieStart = 0
myMovieIncr  = 1
parameters   = "N = {:g} | orient = {:g} | K_hookean = {:g} | K_sfp = {:g} | K_cross = {:g} | no_of_inclusions = {:g} |runs = {:g} | maxIter = {:g} | tolerance = {:g}".format(N, orient, K_hookean, K_sfp, K_cross, num_of_incl, runs, maxIter, tolerance)


for h in range(no_frac):
    frac_springs     = frac_array[h]
    
    energy_runs           = np.zeros((runs, Nsteps))  #stores energy at each strain in each run
    success_runs          = np.zeros((runs, Nsteps))  #stores success at each strain in each run
    sfp_energy_runs       = np.zeros((runs, Nsteps))  #spf (theta0 = pi) angular springs
    crosslink_energy_runs = np.zeros((runs, Nsteps))  #crosslink (theta0 = pi/3) angular springs
    hookean_energy_runs   = np.zeros((runs, Nsteps))  #hookean springs
    inclusion_energy_runs = np.zeros((runs, Nsteps))  #area springs of inclusions
    
    for i in range(runs):
        ### create RSN objects and assign springs
        print("run = "+str(i), flush='True')
    
        s1 = RandomSpringNetwork(Nx, Ny, orient)  # s1 will be optimized using CG
        s1.set_initial_config()
        s1.set_springs(K_hookean, K_sfp, K_cross, K_area, frac_springs)
        s1.set_inclusions(num_of_incl)
                         
        ### for a series of strain values, optimize and write an image file
    
        strainStep          = maxStrain/Nsteps
        R1                  = [] # to store s1 solution
        E1_store            = [] #stores energy at each strain
        success1_store      = [] #stores minimization success
        sfp_store           = [] #stores angular spring energy at each strain   
        crosslink_store     = [] #stores cross_link energy (theta0 = pi/3) at each strain 
        hookean_store       = [] #stores hookean spring energy at each strain
        inclusion_store     = [] #stores area spring energy of inclusions
        
                                 # s1.draw_network(0, True, dots=True)
        for k in range(Nsteps):
        
            s1.apply_compression( (k+1)*strainStep ) 
    
            guess1          = s1.get_coords()  
         
            cons = ()
                        
            res             = minimize(s1.energy, guess1, 
                                       constraints = cons, bounds = s1.bounds(), 
                                       tol=tolerance, options={'maxiter':maxIter}, 
                                       method='SLSQP')
            E1_store        = np.append(E1_store, res.fun)
            success1_store  = np.append(success1_store, res.success)
            
            R1              = res.x
            s1.set_coords(R1) 
            
            sfpEnergy       = s1.sfp_energy(s1.get_coords())
            sfp_store       = np.append(sfp_store, sfpEnergy)
            crossEnergy     = s1.cross_link_energy(s1.get_coords())
            crosslink_store = np.append(crosslink_store, crossEnergy) 
            hookeanEnergy   = s1.hookean_energy(s1.get_coords())   
            hookean_store   = np.append(hookean_store, hookeanEnergy)   
            inclusionEnergy = s1.inclusion_energy(s1.get_coords())
            inclusion_store = np.append(inclusion_store, inclusionEnergy)
                             
            # s1.draw_network(k, res.success, dots=True)   
        
        energy_runs[i]           = E1_store
        success_runs[i]          = success1_store 
        sfp_energy_runs[i]       = sfp_store
        crosslink_energy_runs[i] = crosslink_store
        hookean_energy_runs[i]   = hookean_store   
        inclusion_energy_runs[i] = inclusion_store
        
    
    s1.store(runs, maxStrain, Nsteps, frac_array[h], energy_runs, sfp_energy_runs, 
             crosslink_energy_runs, hookean_energy_runs, inclusion_energy_runs, success_runs, parameters)
