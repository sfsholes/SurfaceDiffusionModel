# SurfaceDiffusionModel
Fick's Law atmospheric surface diffusion model from Sholes et al. 2019 (in Astrobiology)

This model is used to calculate at what depth microbes could plausibly exploit atmospheric free energy on Mars. To calculate the diffusion flux of gases (CO and H2 in the paper), we use a modified form of Fick's laws which assume mean free paths of diffusing gases are greater than the typical pore size such that Knudsen diffusion dominates. 

Given the uncertainites of the physical surface conditions and gradients, we employ a Monte Carlo simulation to test the potential flux under both a range of microbial layer depth and physical parameter values (e.g., porosity, tortuosity, pore size, pore closure depth, and temperature). For each depth we assume each parameter has a uniform distribution bounded by the parameter ranges set in the main file. 

Thus, this model can both run gas subsurface diffusion calculations for a given set of regolith parameters as well as provide a likelihood distribution of the diffusion flux as a function of the biological layer depth. 

To test atmospheric diffusion without a biotic layer at depth, this code can be modified to use the pore closure depth as a lower bound where concentration of gases is set to 0. 

Built for Python 3.7
Can be run with the base parameters used in the paper from the terminal


