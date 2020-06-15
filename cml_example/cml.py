import os
import sys
import numpy as np
import scipy.stats as st
fpth = os.path.abspath(os.path.join('..', 'rmwspy'))
sys.path.append(fpth)
from random_mixing_whittaker_shannon import *

class CMLModel(NonLinearProblemTemplate):
	def __init__(self, data, marginal, cmllinks):
		self.data = data        
		self.marginal = marginal
		self.cmllinks = cmllinks     
		
	def objective_function(self, prediction):
		if prediction.ndim == 1:
			return np.mean((self.data - prediction)**2)**0.5
		elif prediction.ndim == 2:
			return np.mean((self.data - prediction)**2, axis=1)**0.5
	
	def allforwards(self, fields):	
		out = np.empty((fields.shape[0], self.data.shape[0]))
		for k in range(fields.shape[0]):
			out[k] = self.forward(fields[k])
		return out
		
	def forward(self, field):        
		rain = st.norm.cdf(field)
		mp0 = rain <= self.marginal['p0']
		rain[mp0] = 0.0
		rain[~mp0] = (rain[~mp0] - self.marginal['p0']) / (1. - self.marginal['p0'])
		rain[~mp0] = self.marginal['invcdf'](rain[~mp0])
		rain[~mp0] = np.exp(rain[~mp0])/10.

		# get CML integrals from simulated rain field
		CML = []
		for link in range(self.data.shape[0]):
			nlvals_at_x = self.get_cml_on_path(rain, self.cmllinks[link])
			CML.append(np.mean(nlvals_at_x))
		CML = np.array(CML)
		return CML

	def get_cml_on_path(self, data, cp):
		assert cp.ndim > 1
		dimensions = list(map(lambda x: cp[:,x], range(cp.ndim)))
		fullslice = [slice(None,None)] 
		if data.ndim > cp.ndim:
			return data[ tuple(fullslice + dimensions) ].T
		else:
			return data[ tuple(dimensions) ].T


