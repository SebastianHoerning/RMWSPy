import os
import sys
import datetime
import numpy as np
import scipy.stats as st
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import bresenhamline as bresenhamline
import scipy.interpolate as interpolate
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from cml import *
import RMWSPy.rmwspy.gcopula_sparaest as sparest

# start time
start = datetime.datetime.now()

# use random seed if you want to ensure reproducibility
np.random.seed(121)

# paperstyle_plot
# if True -> same plot style as in paper
# which requires the installation of some
# non-standard modules and the download
# of an additional nc file
paperstyle_plot = False 

# PREPROCESSING
#--------------
# define start and end time step
# 543 is the time step used in the paper
start_time_idx = 541
end_time_idx = 542

# start_time_idx = 543
# end_time_idx = 544

# get path to input_data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datafolder = os.path.join(project_root, r'cml_example\input_data')

# load input data
# rain gauge data
p_data     = np.genfromtxt(os.path.join(datafolder,'syn_obs_RG_2015_08.csv'), delimiter='\t')
# rain gauge coordinates
p_lat_lon = p_data[:,(0,1)]

# CML data
mwl_data   = np.genfromtxt(os.path.join(datafolder,'syn_obs_CML_2015_08.csv'), delimiter='\t')
# CML coordinates
mwl_lat_lon = mwl_data[:,0:4]	

# define grid
xs = 7.94736
xsize = 165
xinc = 0.014798
ys = 47.7972403
ysize = 183
yinc = 0.010011303
xg = np.arange(xs,(xs+xinc*xsize),xinc)
yg = np.arange(ys,(ys+yinc*ysize),yinc)
grid = np.array(np.meshgrid(yg,xg)).reshape(2,-1).T

# transform to standard xy grid
grid_ = np.copy(grid)
grid_[:,0] = (grid_[:,0]-ys)/yinc
grid_[:,1] = (grid_[:,1]-xs)/xinc

# transform p_lat_lon to same grid
p_xy = np.copy(p_lat_lon)
p_xy[:,0] = (p_xy[:,0] - ys)/yinc
p_xy[:,1] = (p_xy[:,1] - xs)/xinc
p_xy = p_xy.astype(int)

# transform mwl_lat_lon to same grid
mwl_xy = np.copy(mwl_lat_lon)
mwl_xy[:,0] = (mwl_xy[:,0] - ys)/yinc
mwl_xy[:,2] = (mwl_xy[:,2] - ys)/yinc
mwl_xy[:,1] = (mwl_xy[:,1] - xs)/xinc
mwl_xy[:,3] = (mwl_xy[:,3] - xs)/xinc
mwl_xy = mwl_xy.astype(int)

# define line integrals between the two coordinates of mwl using Bresenham's Line Algorithm
mwl_integrals = []
for integ in range(mwl_xy.shape[0]):
	mwl_integrals.append(np.array(
	bresenhamline.get_line(mwl_xy[integ,:2], mwl_xy[integ,2:])))

# loop over time steps
for tstep in range(start_time_idx, end_time_idx):

	# rain gauge values
	prec = p_data[:, (2+tstep)]

	# CML integral values
	mwl_prec = mwl_data[:, (5+tstep)]

	# transform mm to 1/10mm (not necessary but better to handle with KDE)
	prec = prec * 10.

	# fit a non-parametric marginal distribution using KDE with Gaussian kernel
	# this assumes that there are wet observations
	p0 = 1. - np.float(prec[prec > 0].shape[0]) / prec.shape[0]   

	# optimize the kernelwidth
	x = np.log(prec[prec > 0])
	bandwidths = 10 ** np.linspace(-1, 1, 100)
	grid = GridSearchCV(KernelDensity(kernel='gaussian'),
					{'bandwidth': bandwidths},
					cv=5)
	grid.fit(x[:, None])

	# use optimized kernel for kde
	kde = KernelDensity(bandwidth=grid.best_params_['bandwidth'], kernel='gaussian')
	kde.fit(x[:, None])

	# build cdf and invcdf from pdf
	xx = np.arange(x.min() - 1., x.max() + 1., 0.001)
	logprob = np.exp(kde.score_samples(xx[:, None]))
	cdf_ = np.cumsum(logprob) * 0.001
	cdf_ = np.concatenate(([0.0], cdf_))
	cdf_ = np.concatenate((cdf_, [1.0]))
	xx = np.concatenate((xx, [x.max() + 1.]))
	xx = np.concatenate(([x.min() - 1.], xx))
	cdf = interpolate.interp1d(xx, cdf_)
	invcdf = interpolate.interp1d(cdf_, xx)

	# transform observations to standard normal using the fitted cdf;
	# zero (dry) observations 
	mp0 = prec == 0.0   
	lecp = p_xy[mp0]
	lecv = np.ones(lecp.shape[0]) * st.norm.ppf(p0)

	# non-zero (wet) observations
	cp = p_xy[~mp0]
	cv = st.norm.ppf((1.-p0)*cdf(np.log(prec[~mp0]))+p0)

	# fit a Gaussian copula -> spatial model
	outputfile = None               			# if you want to specify an outputfile -> os.path.join(savefolder, 'MLM.sparaest')   
	u = (st.rankdata(prec) - 0.5) / prec.shape[0]   # observations in copula (rank) space
	covmods = ['Mat', 'Exp', 'Sph',]     # covariance function that will be tried for the fitting
	ntries = 6                                  # number of tries per covariance function with random subsets

	# cmods = sparest.paraest_multiple_tries(np.copy(p_xy),
	# 									   u,
	# 									   ntries = [ntries, ntries],
	# 									   n_in_subset = 5,               # number of values in subsets
	# 									   neighbourhood = 'nearest',     # subset search algorithm
	# 									   covmods = covmods,             # covariance functions
	# 									   outputfile = outputfile)       # store all fitted models in an output file

	# # take the copula model with the highest likelihood
	# # reconstruct from parameter array
	# likelihood = -666
	# for model in range(len(cmods)):
	# 	for tries in range(ntries):
	# 		if cmods[model][tries][1]*-1. > likelihood:
	# 			likelihood = cmods[model][tries][1]*-1.
	# 			cmod = '0.01 Nug(0.0) + 0.99 %s(%1.3f)'%(covmods[model], cmods[model][tries][0][0])
	# 			if covmods[model] == 'Mat':
	# 				cmod += '^%1.3f'%(cmods[model][tries][0][1])

	cmod = '1.0 Mat(60)^1.0'
	# print (cmod)
 
	# SIMULATION USING RMWSPy
	#------------------------
	# number of conditional fields to be simulated
	nfields = 4

	# marginal distribution variables
	marginal = {}
	marginal['p0'] = p0
	marginal['invcdf'] = invcdf

	# initialize CMLModel
	my_CMLModel = CMLModel(mwl_prec, marginal, mwl_integrals)

	# initialize Random Mixing Whittaker-Shannon
	CS = RMWS(my_CMLModel,
			 domainsize = (ysize, xsize),
			 covmod = cmod,
			 nFields = nfields,
			 cp = cp,
			 cv = cv,
			 le_cp = lecp,
			 le_cv = lecv,
			 optmethod = 'circleopt',
			 minObj = 0.4,
			 maxbadcount= 20,    
			 maxiter = 4,
			 )

	# run RMWS
	CS()

	# POST-PROCESSING
	#----------------
	# backtransform simulated fields to original data space
	f_prec_fields = st.norm.cdf(CS.finalFields)

	mp0f = f_prec_fields <= p0
	f_prec_fields[mp0f] = 0.0

	f_prec_fields[~mp0f] = (f_prec_fields[~mp0f]-p0)/(1.-p0)
	f_prec_fields[~mp0f] = invcdf(f_prec_fields[~mp0f])
	f_prec_fields[~mp0f] = np.exp(f_prec_fields[~mp0f])
	f_prec_fields[~mp0f] /= 10.         # to get back to mm

	# save simulated precipitation fields
	np.save('sim_precfields_tstp=%i.npy'%tstep, f_prec_fields)

	# create box plot for simulated MWL values
	mwldict = {}
	for i in range(mwl_prec.shape[0]):
		mwldict[i] = []
		coords = mwl_integrals[i]

		for j in range(nfields):
			mwldict[i].append(f_prec_fields[j, coords[:,0], coords[:,1]].mean())

	boxlist = []
	for i in range(mwl_prec.shape[0]):
		boxlist.append(mwldict[i])
	x = np.arange(1, mwl_prec.shape[0]+1)


	# random index for plotting single realization
	rix = np.random.randint(0, f_prec_fields.shape[0], 1)

	if paperstyle_plot:
		import plot_paper
		plot_paper.plot_pp(datafolder, p_data, mwl_data, f_prec_fields, mwl_prec, boxlist, x, f_prec_fields.shape[0], tstep, rix)
	else:
		# basic plots
		# box plot
		plt.figure(figsize=(12,5))
		plt.boxplot(boxlist)
		plt.plot(x, mwl_prec, 'x', c='red')
		plt.savefig(r'boxplot_mwl_tstp=%i.png'%tstep, dpi=150)
		plt.clf()
		plt.close()

		# plot single realization
		plt.figure()
		plt.imshow(f_prec_fields[rix[0]], origin='lower', interpolation='nearest', cmap='jet')
		plt.plot(p_xy[:,1],p_xy[:,0],'x',c='black')
		for i in range(len(mwl_integrals)):
			plt.plot(mwl_integrals[i][:,1], mwl_integrals[i][:,0], '.', c='green')
		plt.plot(mwl_xy[:,1], mwl_xy[:,0], '.', c='red')
		plt.plot(mwl_xy[:,3], mwl_xy[:,2], '.', c='blue')
		plt.colorbar()
		plt.savefig(r'prec_field_tstp=%i.png'%tstep)
		plt.clf()
		plt.close()

		# plot mean field and standard deviation field
		meanfield = np.mean(f_prec_fields, axis=0)
		stdfield = np.std(f_prec_fields, axis=0)
		fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
		img1 = axs[0].imshow(meanfield,
							 origin='lower',
							 interpolation='nearest',
							 cmap='Blues')
		axs[0].plot(p_xy[:,1],p_xy[:,0],'x',c='black')
		for i in range(len(mwl_integrals)):
			axs[0].plot(mwl_integrals[i][:,1], mwl_integrals[i][:,0], '.', c='green')
		axs[0].plot(mwl_xy[:,1], mwl_xy[:,0], '.', c='red')
		axs[0].plot(mwl_xy[:,3], mwl_xy[:,2], '.', c='blue')
		axs[0].set_title("mean field")
		divider1 = make_axes_locatable(axs[0])
		cax1 = divider1.append_axes("right", size="10%", pad=0.1)
		cbar1 = plt.colorbar(img1, cax=cax1) 

		img2 = axs[1].imshow(stdfield,
							 origin='lower',
							 interpolation='nearest',
							 cmap='Reds'
							 )
		axs[1].plot(p_xy[:,1], p_xy[:,0], 'x', c='black')
		for i in range(len(mwl_integrals)):
			axs[1].plot(mwl_integrals[i][:,1], mwl_integrals[i][:,0], '.', c='green')
		axs[1].plot(mwl_xy[:,1], mwl_xy[:,0], '.', c='red')
		axs[1].plot(mwl_xy[:,3], mwl_xy[:,2], '.', c='blue')
		axs[1].set_title("standard deviation field")
		divider2 = make_axes_locatable(axs[1])
		cax2 = divider2.append_axes("right", size="10%", pad=0.1)
		cbar2 = plt.colorbar(img2, cax=cax2)
		plt.savefig(r'meanf_stdf_tstp=%i.png'%tstep, dpi=150)
		plt.clf()
		plt.close()

	end = datetime.datetime.now()

	print('time needed:', end - start)