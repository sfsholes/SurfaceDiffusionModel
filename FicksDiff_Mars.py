# FicksDiff_Mars.py
# By Steven F. Sholes, PhD
# sfsholes@uw.edu
# Updated: 3/4/2020
# IMPORTANT: Please cite Sholes et al. 2019 in Astrobiology
# (doi:10.1089/ast.2018.1835) if using this model or variations thereof
#
# This model is used to model the flux of atmospheric gases into the
# surface and should be abstractable to other rocky bodies. A full
# description of the model physics is included in the above paper.
#
# This model currently outputs plots for diffusion of CO and H2
# into the subsurface of Mars. A Monte Carlo simulation is performed
# as many regolith variables are uncertain.
#
# To run set the following primary components:
#       MC_runs - the number of simulations at each depth (1000 used)
#       zb_res  - resolution for testing where the microbial layer is (100 used)
#                 i.e., where the gas concentration is 0
#       res     - resolution for depth (1000 used)

## IMPORT MODULES ##
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import pylab
import pdb
import os

## DEFINE VARIABLES ##
MC_runs = 1000      #Number of runs per linspace for Monte Carlo simulation
zb_res = 100        #Resolution of zb (0 concentration layer) to test
res = 1000.         #Resolution of z linspace
cwd = os.getcwd()   #Current Working Directory for saving
#If the code is not working, try setting cwd to the actual path
#cwd = "C:\\Users\\steve\\Desktop\\GitHubTest"

## DEFINE CONSTANTS ##
Rgas = 8.31446e7    #erg/mol K          Gas Constant
T0 = 210.           #K                  Surface Temp
Na = 6.022E+23      #molecules/mol      Avogadro's Number
g = 371.            #cm/s2              Mars Surface Gravity
P0 = 6000.          #g/cm/s2            Mars Surface Pressure
cmkm = 100000.0     #                   cm in a km conversion
CO_mass = 28        #g/mol              CO molecular mass
H2_mass = 2         #g/mol              H2 molecular mass
r0 = 6.0e-4         #cm                 Mean pore size

## DEFINE MC TESTING BOUNDS ##
# Ranges for each variable used in the Monte Carlo simulation
eps0_l, eps0_h = 0.2, 0.6               # Surface porosity range (unitless)
tau0_l, tau0_h = 1.5, 2.5               # Surface tortuosity range (unitless)
r0_l, r0_h = 1e-4, 1e-3                 # Mean pore size range (cm)
k_l, k_h = 6, 26                        # Pore closure depth range (km)
rho_l, rho_h = 2.9, 3.1                 # Rock density range (g/cm3)
m_l, m_h = 10, 30                       # Temperature gradient range (K/km)
zb_l, zb_h = 0.001, 10                  # Microbial layer depth range (km), 0 gives errors

## DEFINE BOUNDARY CONDITIONS ##
c0_CO = 7.47e-4     #                   CO Surface mixing ratio
n0_CO = 1.56e14     #                   CO Surface number density
n0_H2 = 3.10e12     #                   H2 Surface number density
#nzb = 20.e-9       #                   Used for testing non-zero density at depth    
nzb = 0             #molecules/cm3      CO number density at depth zb

## SET MAXIMUM FLUXES ##
# Used for plotting, these come from the photochemical model
max_bio_1 = 2.1e8   #molecules/cm2/s    Maximum biological sink Met 1
max_bio_2 = 1.6e5   #molecules/cm2/s    Maximum biological sink Met 2
max_bio_3 = 1.7e5   #molecules/cm2/s    Maximum biological sink Met 3
max_bio_4 = 1.4e8   #molecules/cm2/s    Maximum biological sink Met 4
max_bio_5 = 1.3e8   #molecules/cm2/s    Maximum biological sink Met 5

def run(eps0,tau0,r0,k,rho,m,mass,zb):
    """Used to run the diffusion model under the given parameter space:
    eps:    Surface porosity        Unitless
    tau:    Surface tortuosity      Unitless
    r0:     Mean pore size          cm
    k:      Pore closure depth      km  (converted into cm here)
    rho:    Rock density            g/cm3
    m:      Temperature slope       K/km
    mass:   Molar mass of molecule  g/mol
    zb:     Microbial depth         km"""
    
    #Calculate the constant in diffusivity
    D_var = (eps0*np.sqrt(8*Rgas))/(tau0*3*np.sqrt(np.pi*mass))
    #Convert k,m,zb into cgs units
    k *= cmkm
    m /= cmkm
    zb *= cmkm
    
    #Calculate the r(z) linear slope
    r = (r0 - 0)/(0 - k)
    
    #Check if running H2 or CO
    if int(mass) == 28:
        n0 = n0_CO
    elif int(mass) == 2:
        n0 = n0_H2
    else:
        print("Choose either CO (28) or H2 (2)")
    
    def fun(z,n):
        #Diffusivity (D) function
        D_fun = D_var*np.sqrt(m*z+T0)*(r*z+r0)*np.exp(-z/k)/(np.exp(-z/k)**(-1./3.))
        #Derivative of Diffusivity (dD/dz)
        D1_fun = (D_var*r*(np.exp(-z/k))**(2./3.)*np.sqrt(m*z+T0))+((D_var*m*(np.exp(-z/k))**(2./3.)*(r*z+r0))/(2*np.sqrt(m*z+T0)))- \
                    ((2*D_var*(np.exp(-z/k))**(2./3.)*np.sqrt(m*z+T0)*(r*z+r0))/(3*k))
        #Return (dn/dz, d2n/dz2)
        return np.vstack((n[1],-(D1_fun/D_fun)*n[1]))
    
    def bc(na,nb):
        #Return (na[0] - surface concentration, nb[0] - zb concentration)
        return np.array([na[0]-n0,nb[0]-nzb])
    
    #Setup depth (z) grid and initial guess for number density profile (n)
    z = np.linspace(0,zb,res)
    n = np.ones((2,z.size))*n0
    
    #Run the numerical solver
    sol = solve_bvp(fun,bc,z,n)
    
    #Check that the solution worked
    #if sol.status != 0:
    #    print("WARNING: sol.status is %d" % sol.status)
    #print(sol.message)
    
    return {'sol':sol,
            'zb':zb,
            'T':m,
            'rho':rho,
            'r':r,
            'r0':r0,
            'tau':tau0,
            'eps':eps0,
            'k':k,
            'Dvar':D_var}

def plotter(result):
    """Plots the Mixing Ratio, Flux, Number Density, Diffusivity
    and dn/dz based on the output (result) of run()
    
    !!!Currently not tested for r(z) - pore space with depth"""
    
    ### --- PLOT THE MIXING RATIO --- ###
    plt.figure()
    plt.title("Mixing Ratio with Depth")
    mr = (result["sol"].y[0]*Rgas*(result["T"]*result["sol"].x+T0))/(Na*(result["rho"]*g*result["sol"].x+P0))
    plt.plot(result["sol"].x/cmkm,mr,label='T = '+str(result["T"]*cmkm)+' K/km\nzb = '+
             str(result["zb"]/cmkm)+' km\neps = '+str(result["eps"])+'\ntau = '+str(result["tau"]))
    #plt.plot(result["sol"].x/cmkm,(-(c0_CO/result["zb"])*result["sol"].x+c0_CO))
    plt.yscale('log')
    plt.ylabel('Mixing Ratio')
    plt.xlabel('Depth [km]')
    plt.legend()
    
    ### --- FIND AND PLOT THE FLUX --- ###
    dz = result["sol"].x[1] - result["sol"].x[0]
    flux = -(result["Dvar"]*np.sqrt(result["T"]*result["sol"].x+T0)*(result["r"]*result["sol"].x+result["r0"])*np.exp(-result["sol"].x/ \
                result["k"])/(np.exp(-result["sol"].x/result["k"])**(-1./3.)))*result["sol"].y[1]
    print('%.2e ' %flux[0])
    plt.figure()
    plt.title('Flux with Depth')
    plt.plot(result["sol"].x/cmkm,flux,label='T = '+str(result["T"]*cmkm)+' K/km\nzb = '+
             str(result["zb"]/cmkm)+' km\neps = '+str(result["eps"])+'\ntau = '+str(result["tau"]))
    plt.yscale('log')
    plt.ylabel('Flux [molecules/cm2/s]')
    plt.xlabel('Depth [km]')
    plt.legend()
    
    # ### --- PLOT THE NUMBER DENSITITES --- ###
    # plt.figure()
    # plt.title('n with Depth')
    # plt.plot(result["sol"].x/cmkm,result["sol"].y[0],label='T = '+`result["T"]*cmkm`+' K/km\nzb = '+
    #          `result["zb"]/cmkm`+' km\neps = '+`result["eps"]`+'\ntau = '+`result["tau"]`)
    # #plt.plot(result["sol"].x/cmkm,(-n0/result["zb"])*result["sol"].x+n0,label='Weiss et al. 2000')
    # #plt.yscale('log')
    # #plt.ylim(1e-8,1e-5)
    # plt.ylabel('Number Density')
    # plt.xlabel('Depth [km]')
    # plt.title('n with Depth')
    # plt.legend()
    
    plt.show()

def montecarlo(numruns,mass):
    """Runs a Monte Carlo system for a variety of set ranges
    
    numruns:    int     Number of runs to run for each zb
    mass:       int     Either 2 (H2) or 28 (CO)"""
    
    #Set up lists
    zb_range = np.linspace(zb_l,zb_h,zb_res)
    flux_list = []
    med_flux_list = []
    
    #Set up main loop
    i = 0
    #Test over all zb defined in range 
    while i < zb_range.size:
        file_count = 0
        z_med = []
        #Run the diffusion model with random sample of variables within range for the given zb
        while file_count <= numruns:
            result = run(np.random.uniform(eps0_l, eps0_h),np.random.uniform(tau0_l, tau0_h),np.random.uniform(r0_l, r0_h), \
                         np.random.uniform(k_l, k_h),np.random.uniform(rho_l, rho_h),np.random.uniform(m_l, m_h),mass,zb_range[i])
            #result = run(0.6,2,6e-4,10,2.9,20,28,zb_range[i])
            #Make sure to only keep solutions that worked
            if result["sol"].status != 0:
                file_count += 1
            flux = -(result["Dvar"]*np.sqrt(result["T"]*result["sol"].x+T0)*(result["r"]*result["sol"].x+result["r0"])*np.exp(-result["sol"].x/result["k"])/ \
                     (np.exp(-result["sol"].x/result["k"])**(-1./3.)))*result["sol"].y[1]
            flux_list.append([zb_range[i],flux[0]])
            z_med.append(flux[0])
            file_count += 1
        med = np.median(z_med)
        #ADJUST this number as necessary. Needed to prevent weird plots in case of boundary condition errors
        #I.E. cannot plot negative fluxes on log scale
        if med < 0:
            med = 1e9
        med_flux_list.append(med)
        i += 1
    
    new_flux = np.array(flux_list)
    np.save(os.path.join(cwd,'FD_code_flux'),new_flux)
    np.save(os.path.join(cwd,'FD_code_med'),med_flux_list)
    #pdb.set_trace()
    
    bins = np.logspace(5,11,200)
    #pdb.set_trace()
    
    #Pass the results onto the plotting function
    hist,xedges,yedges = np.histogram2d(new_flux[:,1],new_flux[:,0],[bins,np.linspace(0,10,zb_res+1)])#,range=[[1e6,1e10],[0,10]])
    return [hist,xedges,yedges,med_flux_list,zb_range]

def mcplot(result, result2):
    """Plots the results of montecarlo(). Takes in an array with new_flux, med_list."""
    
    #REDFINE individual variables based on passed matrix from montecarlo function
    hist = result[0]
    xedges = result[1]
    yedges = result[2]
    x,y = np.meshgrid(xedges,yedges)
    
    hist2 = result2[0]
    xedges2 = result2[1]
    yedges2 = result2[2]
    x2,y2 = np.meshgrid(xedges2,yedges2)
    
    diff = (result[4][2]-result[4][1])/2.
    
    #plt.figure()
    f, axarr = plt.subplots(2,sharex=True)
    axarr[0].plot(result[3],result[4]+diff,color='k')
    axarr[0].pcolormesh(x,np.flipud(y),np.flipud(np.transpose(hist)),cmap='Blues')
    axarr[0].axvline(x=max_bio_4,linewidth=3,color='k',linestyle='--',label='Met. 4')
    axarr[0].axvline(x=max_bio_1,linewidth=3,color='crimson',linestyle='-',label='Met. 1')
    axarr[0].axvline(x=max_bio_3,linewidth=3,color='b',linestyle='-',label='Met. 3')
    axarr[0].set_xscale('log')
    axarr[0].set_ylim(0,10)
    axarr[0].set_xlim(1e5,1e11)
    #axarr[0].set_xlabel('Flux [molecules cm-2 s-1]')
    axarr[0].set_ylabel('Depth [km]')
    axarr[0].invert_yaxis()
    axarr[1].pcolormesh(x2,np.flipud(y2),np.flipud(np.transpose(hist2)),cmap='Purples')
    axarr[1].plot(result2[3],result2[4]+diff,color='k')
    axarr[1].axvline(x=max_bio_5,linewidth=3,color='g',linestyle='--',label='Met. 5')
    axarr[1].axvline(x=max_bio_2,linewidth=3,color='darkviolet',linestyle='-',label='Met. 2')
    axarr[1].set_xscale('log')
    axarr[1].set_ylim(0,10)
    axarr[1].set_xlim(1e5,1e11)
    axarr[1].set_xlabel('Flux [molecules cm$^2$ s$^{-1}$]')
    axarr[1].set_ylabel('Depth [km]')
    axarr[1].invert_yaxis()
    f.subplots_adjust(hspace=0)
    #plt.savefig('WithoutLegend.png')
    axarr[0].legend(loc=2,prop={'size': 10})
    axarr[1].legend(loc=1,prop={'size': 10})
    #plt.savefig('WithLegend.png')
    plt.show()

mcplot(montecarlo(MC_runs,CO_mass),montecarlo(MC_runs,H2_mass))