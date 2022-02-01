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
import numpy as np
import scipy.stats as st
from RMWSPy.rmwspy import *
from concurrent.futures import ThreadPoolExecutor

try:
	import flopy
	'''
	flopy references:

	Bakker, M., Post, V., Langevin, C. D., Hughes, J. D.,
	White, J. T., Starn, J. J. and Fienen, M. N., 2016, 
	Scripting MODFLOW Model Development Using Python 
	and FloPy: Groundwater, v. 54, p. 733–739, doi:10.1111/gwat.12413.

	Bakker, M., Post, V., Langevin, C. D., Hughes, J. D., White, J. T., 
	Leaf, A. T., Paulinski, S. R., Larsen, J. D., Toews, M. W., Morway, 
	E. D., Bellino, J. C., Starn, J. J., and Fienen, M. N., 2019, 
	FloPy v3.3.1 — release candidate: U.S. Geological Survey Software 
	Release, 15 December 2019, http://dx.doi.org/10.5066/F7BK19FH
	'''
except:
	# CHANGE THIS PATH TO YOUR FLOPY INSTALLATION IF IT IS NOT IN YOUR PYTHONPATH
	fpth = os.path.abspath(os.path.join(r'C:\Users\uqshoern\Documents\transport_random_mixing'))
	sys.path.append(fpth)
	import flopy

class GWModel(NonLinearProblemTemplate):
	def __init__(self, data, data_cp, MF, marginal, laytyp, kv, headfile, threading=False, nthreads=6):
		self.data = data        
		self.data_cp = data_cp
		self.MF = MF
		self.marginal = marginal
		self.laytyp = laytyp
		self.kv = kv
		self.headfile = headfile
		self.threading = threading 
		self.nthreads = nthreads      
		
	def objective_function(self, prediction):
		if prediction.ndim == 1:
			return np.mean((self.data - prediction)**2)**0.5
		elif prediction.ndim == 2:
			return np.mean((self.data - prediction)**2, axis=1)**0.5
	
	def allforwards(self, fields):	
		out = np.empty((fields.shape[0], self.data.shape[0]))	
		if self.threading:
			out_ = self.threading_forward(fields)
			for i, o in enumerate(out_):
				out[i] = o			
		else:			
			for k in range(fields.shape[0]):
				out[k] = self.forward(k, fields[k])
		return out

	def threading_forward(self, fields):
		i = np.arange(fields.shape[0])
		with ThreadPoolExecutor(max_workers = self.nthreads) as executor:
			results = executor.map(self.forward, i, fields)
		return results
		
	def forward(self, i, kfield):
		kh = st.lognorm.ppf(st.norm.cdf(kfield), self.marginal[0], loc=self.marginal[1], scale=self.marginal[2])
		# Create the node property flow package      
		self.MF['gwf'][i].remove_package('npf')
		self.MF['npf'][i] = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(self.MF['gwf'][i], pname='npf',
													   icelltype=self.laytyp, k=kh,
													   k33=self.kv)

		# Write the datasets
		self.MF['sim'][i].write_simulation()
	
		# Run the simulation
		success, buff = self.MF['sim'][i].run_simulation(silent=True)
		assert success, 'mf6 model did not run. check folder {}'.format(i)

		hds = flopy.utils.HeadFile(os.path.join(self.MF['ws'][i], self.headfile))
		nlfield_at_x = hds.get_data()[0]
		nlvals_at_x = self.get_at_location(nlfield_at_x, self.data_cp)

		return nlvals_at_x

	def get_at_location(self, data, cp):
		assert cp.ndim > 1
		dimensions = list(map(lambda x: cp[:,x], range(cp.ndim)))
		fullslice = [slice(None,None)] 
		if data.ndim > cp.ndim:
			return data[ tuple(fullslice + dimensions) ].T
		else:
			return data[ tuple(dimensions) ].T


