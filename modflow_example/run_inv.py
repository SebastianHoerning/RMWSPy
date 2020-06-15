#-------------------------------------------------------------------------------
# Purpose:     Groundwater inversion
#
# Author:      Dr.-Ing. Sebastian Hoerning
#
# Created:     07.01.2020, Centre for Natural Gas, EAIT, 
#			   The University of Queensland, Brisbane, QLD, Australia
#-------------------------------------------------------------------------------
import os
import sys
import datetime
import IPython
import numpy as np
import scipy.stats as st
import scipy.spatial as sp
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.interpolate as interpolate
from gwmod import *
import gcopula_sparaest as sparest


# use random seed if you want to ensure reproducibility
np.random.seed(123)

# create temp folder in which the individual model runs are saved
if not os.path.exists('temp'):
	os.mkdir('temp')

# example name
nm = 'gw_inv'

'''
	define executable names and path to file
add path to MODFLOW executable
MODFLOW reference:
Langevin, C.D., Hughes, J.D., Banta, E.R., Provost, A.M., 
Niswonger, R.G., and Panday, Sorab, 2019, 
MODFLOW 6 Modular Hydrologic Model version 6.1.0: 
U.S. Geological Survey Software Release, 12 December 2019, 
https://doi.org/10.5066/F76Q1VQV
'''

exe_loc = r'C:\Users\uqshoern\Documents\transport_random_mixing\bin'
mfexe = 'mf6.exe'
mfexe = os.path.join(exe_loc, mfexe)
	
# defines the number of points on the circle
# HAS TO BE AN EVEN NUMBER
p_on_circle = 8

# define workspaces and flowpy objects
MF = {}
MF['ws'] = {}
MF['sim'] = {}
MF['gwf'] = {}
MF['npf'] = {}
for c in range(p_on_circle):
	MF['ws'][c] = os.path.join('temp', '%s_%i'%(nm,c))

nper, nstp, perlen, tsmult = 1, 1, 1., 1.

nlay, nrow, ncol = 1, 80, 50
delr = delc = 10.
top = 10.
botm = 0.

strt = 0.

laytyp = np.zeros((nrow, ncol))
laytyp = laytyp.astype(int)

kh = np.ones((nrow, ncol))
kv = np.ones((nrow, ncol))

wel_loc = []
for w in range(ncol):
	wel_loc.append(((0,0,w),1))

bound_loc = []
for w in range(ncol):
	bound_loc.append(((0,79,w),0))

for c in range(p_on_circle):

	# Create the Flopy simulation object
	MF['sim'][c] = flopy.mf6.MFSimulation(sim_name=nm, exe_name=mfexe,
								 version='mf6', sim_ws=MF['ws'][c])

	# Create the Flopy temporal discretization object
	pd = (perlen, nstp, tsmult)
	tdis = flopy.mf6.modflow.mftdis.ModflowTdis(MF['sim'][c], pname='tdis',
												time_units='DAYS', nper=nper,
												perioddata=[pd])

	# Create the Flopy groundwater flow (gwf) model object
	model_nam_file = '{}.nam'.format(nm)
	MF['gwf'][c] = flopy.mf6.ModflowGwf(MF['sim'][c], modelname=nm,
							   model_nam_file=model_nam_file, save_flows=True)

	# Create the Flopy iterative model solver (ims) Package object
	ims = flopy.mf6.modflow.mfims.ModflowIms(MF['sim'][c], pname='ims', 
											 complexity='SIMPLE',
											 outer_hclose=1e-6,
											 inner_hclose=1e-6,
											 rcloserecord=1e-6)

	# create gwf file
	dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(MF['gwf'][c], pname='dis', nlay=nlay,
												   nrow=nrow, ncol=ncol,
												   length_units='METERS',
												   delr=delr, delc=delc,
												   top=top,
												   botm=botm)

	# Create the initial conditions package
	ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(MF['gwf'][c], pname='ic', strt=strt)

	# Create the node property flow package
	MF['npf'][c] = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(MF['gwf'][c], pname='npf',
												   icelltype=laytyp, k=kh,
												   k33=kv)

	# wel
	flopy.mf6.modflow.mfgwfwel.ModflowGwfwel(MF['gwf'][c], maxbound=len(wel_loc),
											 stress_period_data={0: wel_loc})

	# boundary
	flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(MF['gwf'][c], stress_period_data={0: bound_loc})

	# Create the output control package
	headfile = '{}.hds'.format(nm)
	head_record = [headfile]
	budgetfile = '{}.cbb'.format(nm)
	budget_record = [budgetfile]
	saverecord = [('HEAD', 'ALL'),
				  ('BUDGET', 'ALL')]

	oc = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(MF['gwf'][c], pname='oc',
												saverecord=saverecord,
												head_filerecord=head_record,
												budget_filerecord=budget_record)

	# Write the datasets
	MF['sim'][c].write_simulation()
	# Run the simulation
	success, buff = MF['sim'][c].run_simulation()
	assert success, 'mf6 model did not run'

# read input for random mixing
obswell = np.load(r'well_loc.npy')
h_obswell = np.load(r'well_h.npy')
kh_obswell = np.load(r'well_kh.npy')

# fit marginal
dist_par = st.lognorm.fit(kh_obswell)

# transform observations to standard normal using the fitted cdf;
cv = st.norm.ppf(st.lognorm.cdf(kh_obswell, dist_par[0], loc=dist_par[1], scale=dist_par[2]))
vcdfinv = None

# fit a Gaussian copula -> spatial model
outputfile = None               			# if you want to specify an outputfile -> os.path.join(savefolder, 'MLM.sparaest')   
u = (st.rankdata(kh_obswell)-0.5)/kh_obswell.shape[0]   # observations in copula (rank) space
covmods = ['Mat', 'Exp', 'Sph']     # covariance function that will be tried for the fitting
ntries = 6                                  # number of tries per covariance function with random subsets

cmods = sparest.paraest_multiple_tries(np.copy(obswell),
									   u,
									   ntries = [ntries,ntries],
									   n_in_subset = 5,               # number of values in subsets
									   neighbourhood = 'nearest',     # subset search algorithm
									   covmods = covmods,             # covariance functions
									   outputfile = outputfile)       # store all fitted models in an output file

# take the copula model with the highest likelihood
# reconstruct from parameter array
likelihood = -666
for model in range(len(cmods)):
	for tries in range(ntries):
		if cmods[model][tries][1]*-1. > likelihood:
			likelihood = cmods[model][tries][1]*-1.
			cmod = '0.01 Nug(0.0) + 0.99 %s(%1.3f)'%(covmods[model], cmods[model][tries][0][0])
			if covmods[model] == 'Mat':
				cmod += '^%1.3f'%(cmods[model][tries][0][1])

print (cmod)


# initialize the groundwater model
my_model = GWModel(h_obswell, obswell, MF, dist_par, laytyp, kv, headfile, threading=True, nthreads=4)

# number of conditional fields to be simulated
nfields = 2
# initialize Random Mixing Whittaker-Shannon
CS = RMWS(my_model,
		 domainsize = (nrow, ncol),
		 covmod = cmod,
		 nFields = nfields,
		 cp = obswell,
		 cv = cv,
		 optmethod = 'circleopt',
		 minObj = 0.2,    
		 maxiter = 3,
		 p_on_circle=p_on_circle
		 )

# run RMWS
CS()

# POST-PROCESSING
#----------------
# backtransform simulated fields to original data space
kh_fields = st.lognorm.ppf(st.norm.cdf(CS.finalFields), dist_par[0], loc=dist_par[1], scale=dist_par[2])

# save simulated kh fields
np.save('RandomFields.npy', kh_fields)

# run MF again using these kh fields to obtain final heads
h_fields = []
h_at_wells = []
for i in range(nfields):
	# Create the node property flow package      
	MF['gwf'][0].remove_package('npf')
	MF['npf'][0] = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(MF['gwf'][0], pname='npf',
												   icelltype=laytyp, k=kh_fields[i],
												   k33=kv)

	# Write the datasets
	MF['sim'][0].write_simulation()

	# Run the simulation
	success, buff = MF['sim'][0].run_simulation(silent=True)
	assert success, 'mf6 model did not run'

	hds = flopy.utils.HeadFile(os.path.join(MF['ws'][0], headfile))
	nlfield_at_x = hds.get_data()[0]
	h_fields.append(nlfield_at_x)
	nlvals_at_x = CS.get_at_cond_locations(nlfield_at_x, obswell)
	h_at_wells.append(nlvals_at_x)

h_fields = np.array(h_fields)
h_at_wells = np.array(h_at_wells)


# create box plot for simulated MWL values
hdict = {}
for i in range(h_obswell.shape[0]):
	hdict[i] = []
	for j in range(nfields):
		hdict[i].append(h_at_wells[j][i])

boxlist = []
for i in range(h_obswell.shape[0]):
	boxlist.append(hdict[i])
x = np.arange(1, h_obswell.shape[0]+1)

plt.figure(figsize=(12,5))
plt.boxplot(boxlist)
plt.plot(x, h_obswell, 'x', c='red')
plt.savefig( 'boxplot_h.png', dpi=150)
plt.clf()
plt.close()


# plot mean field and standard deviation field
meanfield = np.mean(kh_fields, axis=0)
stdfield = np.std(kh_fields, axis=0)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
img1 = axs[0].imshow(meanfield,
					 origin='lower',
					 interpolation='nearest',
					 cmap='jet')
axs[0].plot(obswell[:,1],obswell[:,0],'x',c='black')
axs[0].set_title("mean field")
divider1 = make_axes_locatable(axs[0])
cax1 = divider1.append_axes("right", size="10%", pad=0.1)
cbar1 = plt.colorbar(img1, cax=cax1) 

img2 = axs[1].imshow(stdfield,
					 origin='lower',
					 interpolation='nearest',
					 cmap='Reds'
					 )
axs[1].plot(obswell[:,1], obswell[:,0], 'x', c='black')
axs[1].set_title("standard deviation field")
divider2 = make_axes_locatable(axs[1])
cax2 = divider2.append_axes("right", size="10%", pad=0.1)
cbar2 = plt.colorbar(img2, cax=cax2)
plt.savefig('meanf_stdf_kh.png', dpi=150)
plt.clf()
plt.close()

# plot mean field and standard deviation field
meanfield = np.mean(h_fields, axis=0)
stdfield = np.std(h_fields, axis=0)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
img1 = axs[0].imshow(meanfield,
					 origin='lower',
					 interpolation='nearest',
					 cmap='jet')
axs[0].plot(obswell[:,1], obswell[:,0], 'x', c='black')
axs[0].set_title("mean field")
divider1 = make_axes_locatable(axs[0])
cax1 = divider1.append_axes("right", size="10%", pad=0.1)
cbar1 = plt.colorbar(img1, cax=cax1) 

img2 = axs[1].imshow(stdfield,
					 origin='lower',
					 interpolation='nearest',
					 cmap='Reds'
					 )
axs[1].plot(obswell[:,1], obswell[:,0], 'x', c='black')
axs[1].set_title("standard deviation field")
divider2 = make_axes_locatable(axs[1])
cax2 = divider2.append_axes("right", size="10%", pad=0.1)
cbar2 = plt.colorbar(img2, cax=cax2)
plt.savefig('meanf_stdf_h.png', dpi=150)
plt.clf()
plt.close()

# plot reality
khrel = np.load(r'reality_kh.npy')
hrel = np.load(r'reality_h.npy')

plt.figure()
plt.imshow(khrel, interpolation='nearest', origin='lower', cmap='jet')
plt.plot(obswell[:,1], obswell[:,0], 'x', c='black')
plt.colorbar()
plt.savefig(r'reality_kh.png')
plt.clf()
plt.close()

plt.figure()
plt.imshow(hrel, interpolation='nearest', origin='lower', cmap='jet')
plt.plot(obswell[:,1], obswell[:,0], 'x', c='black')
plt.colorbar()
plt.savefig(r'reality_h.png')
plt.clf()
plt.close()




