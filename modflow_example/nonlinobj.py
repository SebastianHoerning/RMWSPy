import os
import numpy as np
import scipy.stats as st

class NonLinearClass:

	def objFunc(self, sim, obs):
		if sim.ndim == 1:
			return np.mean((obs - sim)**2)**0.5
		elif sim.ndim ==2:
			return np.mean((obs - sim)**2, axis=1)**0.5
		

	def forwardmodel(self, i, kfield):
		kh = st.lognorm.ppf(st.norm.cdf(kfield), self.nl_variables['dist_par'][0], loc=self.nl_variables['dist_par'][1], scale=self.nl_variables['dist_par'][2])
		# Create the node property flow package      
		self.nl_variables['MF']['gwf'][i].remove_package('npf')
		self.nl_variables['MF']['npf'][i] = self.nl_variables['flopy'].mf6.modflow.mfgwfnpf.ModflowGwfnpf(self.nl_variables['MF']['gwf'][i], pname='npf',
													   icelltype=self.nl_variables['laytyp'], k=kh,
													   k33=self.nl_variables['kv'])

		# Write the datasets
		self.nl_variables['MF']['sim'][i].write_simulation()
	
		# Run the simulation
		success, buff = self.nl_variables['MF']['sim'][i].run_simulation(silent=True)
		assert success, 'mf6 model did not run'

		hds = self.nl_variables['flopy'].utils.HeadFile(os.path.join(self.nl_variables['MF']['ws'][i], self.nl_variables['headfile']))
		nlfield_at_x = hds.get_data()[0]
		nlvals_at_x = self.get_at_cond_locations(nlfield_at_x, self.NL_cp)

		return nlvals_at_x

	  