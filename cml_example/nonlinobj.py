import numpy as np
import scipy.stats as st

class NonLinearClass:

	def objFunc(self, sim, obs):
		if sim.ndim == 1:
			return np.mean((obs - sim)**2)**0.5
		elif sim.ndim ==2:
			return np.mean((obs - sim)**2, axis=1)**0.5
		
	def forwardmodel(self, i, kfield):       
		# calculate the precipitation field
		rain = st.norm.cdf(kfield)
		mp0 = rain <= self.nl_variables['p0']
		rain[mp0] = 0.0
		rain[~mp0] = (rain[~mp0] - self.nl_variables['p0']) / (1. - self.nl_variables['p0'])
		rain[~mp0] = self.nl_variables['invcdf'](rain[~mp0])
		rain[~mp0] = np.exp(rain[~mp0])/10.

		# get CML integrals from rain
		CML = []
		for link in range(self.NL_cp.shape[0]):
			nlvals_at_x = self.get_at_cond_locations(rain, self.NL_cp[link])
			CML.append(np.mean(nlvals_at_x))

		return CML