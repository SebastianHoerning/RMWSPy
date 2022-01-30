#-------------------------------------------------------------------------------
# Name:        FFT Moving Average (FFT-MA)
# Purpose:     Simulation of standard normal random fields
#
# Author:      Dr.-Ing. S. Hoerning
#
# Created:     19/11/2021, Centre for Natural Gas, EAIT,
#                          The University of Queensland, Brisbane, QLD, Australia
#-------------------------------------------------------------------------------

import numpy as np
import sys
from . import covariancefunction as covfun
import scipy
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

class FFTMA(object):
	def __init__(self,
				 domainsize = (100,100),
				 covmod     = '1.0 Exp(2.)',
				 anisotropy = False, 	# requires tuple (scale 0, scale 1,...., scale n, rotate 0, rotate 1,..., rotate n-1)
				 						# note that scale is relative to range defined in covmod
				 periodic   = False,
				 ):

		self.counter = 0
		self.anisotropy = anisotropy
		self.periodic = periodic
		if len(domainsize) == 3:
			self.xyz = np.mgrid[[slice(0,n,1) for n in domainsize]].reshape(3,-1).T
		# adjust domainsize by cutoff for non-perjodic output
		self.cutoff = 0
		if not self.periodic:
			cutoff = covfun.find_maximum_range(covmod)
			cutoffs = []
			for dim in domainsize:
				tsize = dim + cutoff
				# find closest multiple of 8 that is larger than tsize
				m8 = np.int(np.ceil(tsize/8.)*8.)
				cutoffs.append(m8 - dim)
			self.cutoff = np.array(cutoffs)

		self.domainsize = np.array(domainsize)+self.cutoff
		self.covmod     = covmod
		self.ndim       = len(self.domainsize)
		self.npoints    = np.prod(self.domainsize)

		self.grid = np.mgrid[[slice(0,n,1) for n in self.domainsize]]

		if self.anisotropy == False:
			# ensure periodicity of domain
			for i in range(self.ndim):
				self.domainsize = self.domainsize[:,np.newaxis]
			self.grid = np.min((self.grid, np.array(self.domainsize)-self.grid), axis=0)

			# compute distances from origin (--> wavenumbers in fourier space)
			h = ((self.grid**2).sum(axis=0))**0.5
			# covariances (in fourier space!!!)
			Q = covfun.Covariogram(h, self.covmod)
			FFTQ = np.abs(np.fft.fftn(Q))
			self.sqrtFFTQ = np.sqrt(FFTQ)
		else:
			self.apply_anisotropy()

		# create ones array for local opt
		self.ones = np.ones(self.sqrtFFTQ.shape)
		self.Y = self.simnew()


	def simnew(self):
		self.counter += 1
		# normal random numbers
		u = np.random.standard_normal(size=self.sqrtFFTQ.shape)
		# fft of normal random numbers
		U = np.fft.fftn(u)
		# combine with covariance 
		GU = self.sqrtFFTQ * U
		# create field using inverse fft
		self.Y = np.real(np.fft.ifftn(GU)) 

		if not self.periodic:
			# readjust domainsize to correct size (--> no boundary effects...)
			gridslice = [slice(0,(self.domainsize.squeeze()-self.cutoff)[i],1)
													  for i in range(self.ndim)]
			self.Y = self.Y[tuple(gridslice)]
			self.Y = self.Y.reshape(self.domainsize.squeeze()-self.cutoff)

		return self.Y


	def apply_anisotropy_old(self, aniparameter):
		if self.ndim == 2:			
			angle = aniparameter[-1]
			stretch = np.array([[aniparameter[0], 0], [0, aniparameter[1]]])
			new_grid = self.grid.reshape(2, -1).T
			new_grid = np.dot(stretch, new_grid.T)
			new_grid = new_grid.reshape(self.grid.shape)

			# ensure periodicity of domain	
			for i in range(self.ndim):
				new_grid[i] = np.min((new_grid[i], np.max(new_grid[i]) + 1 - new_grid[i]), axis=0)	

			# compute distances from origin (--> wavenumbers in fourier space)
			h = ((new_grid**2).sum(axis=0))**0.5

			# covariances (in fourier space!!!)
			Q = covfun.Covariogram(h, self.covmod)

			# FFT of covariances	 
			nQ = np.fft.fftshift(scipy.ndimage.rotate(np.fft.fftshift(Q), angle, reshape=False))
			FFTQ = np.abs(np.fft.fftn(nQ))
			self.sqrtFFTQ = np.sqrt(FFTQ)
		elif self.ndim == 3:
			angle = aniparameter[-2]
			angle2 = aniparameter[-1]
			stretch = np.array([[aniparameter[0], 0, 0], [0, aniparameter[1], 0], [0, 0, aniparameter[2]]])
			new_grid = self.grid.reshape(3, -1).T
			new_grid = np.dot(stretch, new_grid.T)
			new_grid = new_grid.reshape(self.grid.shape)

			# ensure periodicity of domain	
			for i in range(self.ndim):
				new_grid[i] = np.min((new_grid[i], np.max(new_grid[i]) + 1 - new_grid[i]), axis=0)

			# compute distances from origin (--> wavenumbers in fourier space)
			h = ((new_grid**2).sum(axis=0))**0.5

			# covariances (in fourier space!!!)
			Q = covfun.Covariogram(h, self.covmod)

			# FFT of covariances
			# DO WE NEED TO ROTATE 2 TIMES????	 		
			nQ = np.fft.fftshift(Q)
			nQ = scipy.ndimage.rotate(nQ, angle, axes=(0,2), reshape=False)
			nQ = scipy.ndimage.rotate(nQ, angle2, axes=(1,2), reshape=False)
			nQ = np.fft.fftshift(nQ)
			# OR JUST ONCE???
			# nQ = np.fft.fftshift(scipy.ndimage.rotate(np.fft.fftshift(Q), angle, axes=(2,1), reshape=False))
			FFTQ = np.abs(np.fft.fftn(nQ))
			self.sqrtFFTQ = np.sqrt(FFTQ)

	def apply_anisotropy(self):
		# Create an array to stretch the distances
		stretchlist =[]
		for d in range(self.ndim):
			stretchdim = [0]*self.ndim
			stretchdim[d] = 1/self.anisotropy[d]
			stretchlist.append(stretchdim)
		stretch = np.array(stretchlist)
		new_grid = self.grid.reshape(self.ndim, -1).T
		new_grid = np.dot(stretch, new_grid.T)
		new_grid = new_grid.reshape(self.grid.shape)

		# ensure periodicity of domain	
		for i in range(self.ndim):
			new_grid[i] = np.min((new_grid[i], np.max(new_grid[i]) + 1 - new_grid[i]), axis=0)

		# compute distances from origin (--> wavenumbers in fourier space)
		h = ((new_grid**2).sum(axis=0))**0.5

		# covariances (in fourier space!!!)
		Q = covfun.Covariogram(h, self.covmod)

		# FFT of covariances and rotation
		nQ = np.fft.fftshift(Q)
		
		# I can't figure out how to make this more general...
		axeslist=[]
		for d in range(self.ndim-1):
			axeslist.append((d,self.ndim-1))

		for d in range(self.ndim-1):
			angle = self.anisotropy[self.ndim+d]
			nQ = scipy.ndimage.rotate(nQ, angle, axes=axeslist[d], reshape=False)
		nQ = np.fft.fftshift(nQ)
		FFTQ = np.abs(np.fft.fftn(nQ))
		self.sqrtFFTQ = np.sqrt(FFTQ)


			

	def simnew_with_given_rns(self, u):
		# fft of normal random numbers
		U = np.fft.fftn(u)
		# combine with covariance 
		GU = self.sqrtFFTQ * U
		# create field using inverse fft
		self.Y = np.real(np.fft.ifftn(GU)) 

		if not self.periodic:
			# readjust domainsize to correct size (--> no boundary effects...)
			gridslice = [slice(0,(self.domainsize.squeeze()-self.cutoff)[i],1)
													  for i in range(self.ndim)]
			self.Y = self.Y[tuple(gridslice)]
			self.Y = self.Y.reshape(self.domainsize.squeeze()-self.cutoff)

		return self.Y

	def simnew_fullsize(self):
		self.counter += 1
		# normal random numbers
		u = np.random.standard_normal(size=self.sqrtFFTQ.shape)
		# fft of normal random numbers
		U = np.fft.fftn(u)
		# combine with covariance 
		GU = self.sqrtFFTQ * U
		# create field using inverse fft
		self.Y = np.real(np.fft.ifftn(GU)) 
		return self.Y

	def sim_on_circle(self, weights, rns, H=None):
				
		if H is not None:
			u = (self.ones - H) * rns[0] + (H * np.tensordot(weights, rns, axes=1))
		else:
			u = np.tensordot(weights, rns, axes=1)

		U = np.fft.fftn(u)
		GU = self.sqrtFFTQ * U

		self.Y = np.real(np.fft.ifftn(GU)) 

		if not self.periodic:
			# readjust domainsize to correct size (--> no boundary effects...)
			gridslice = [slice(0,(self.domainsize.squeeze()-self.cutoff)[i],1)
													  for i in range(self.ndim)]
			self.Y = self.Y[tuple(gridslice)]
			self.Y = self.Y.reshape(self.domainsize.squeeze()-self.cutoff)

		return self.Y

	def sim_on_circle_vtrans(self, weights, rns, m=1.0, k=2.0, inv=False, H=None):
		
		if H is not None:
			u = (self.ones - H) * rns[0] + (H * np.tensordot(weights, rns, axes=1))
		else:
			u = np.tensordot(weights, rns, axes=1)

		U = np.fft.fftn(u)
		GU = self.sqrtFFTQ * U

		self.Y = np.real(np.fft.ifftn(GU)) 

		if not self.periodic:
			# readjust domainsize to correct size (--> no boundary effects...)
			gridslice = [slice(0,(self.domainsize.squeeze()-self.cutoff)[i],1)
													  for i in range(self.ndim)]
			self.Y = self.Y[tuple(gridslice)]
			self.Y = self.Y.reshape(self.domainsize.squeeze()-self.cutoff)

		self.Y = self.vtrans(self.Y, m, k)
		self.Y = self.vcdf(self.Y, m, k)
		# self.Y = (st.mstats.rankdata(self.Y) - 0.5)/np.prod(self.Y.shape)
		if inv:
			self.Y = 1. - self.Y

		return self.Y

	def vtrans(self, x, m, k):
		return np.where(x > m, k*(x-m), m-x)

	def vcdf(self, V, m, k):
		return st.norm.cdf(V/k + m) - st.norm.cdf(-V + m)

	def invert_for_rn(self, field):
		u = np.real(np.fft.ifftn(np.fft.fftn(field)/self.sqrtFFTQ))
		return u

	def sim_with_correlation(self, corr_u, c=0.7):
		self.counter += 1
		# normal random numbers
		u = np.random.standard_normal(size=self.sqrtFFTQ.shape)
		# random numbers correlated with c to corr_u
		# be careful with different shapes!!
		u = c * corr_u[:u.shape[0], :u.shape[1]] + np.sqrt(1 - c**2)*u
		# fft of normal random numbers
		U = np.fft.fftn(u)
		# combine with covariance 
		GU = self.sqrtFFTQ * U
		# create field using inverse fft
		self.Y = np.real(np.fft.ifftn(GU)) 

		if not self.periodic:
			# readjust domainsize to correct size (--> no boundary effects...)
			gridslice = [slice(0,(self.domainsize.squeeze()-self.cutoff)[i],1)
													  for i in range(self.ndim)]
			self.Y = self.Y[tuple(gridslice)]
			self.Y = self.Y.reshape(self.domainsize.squeeze()-self.cutoff)

		return self.Y

	def sim_on_circle_with_correlation(self, weights, rns, corr_u, c=0.7, H=None):		
		if H is not None:
			u = (self.ones - H) * rns[0] + (H * np.tensordot(weights, rns, axes=1))
		else:
			u = np.tensordot(weights, rns, axes=1)

		# introduce the correlation c to corr_u
		u = c * corr_u[:u.shape[0], :u.shape[1]] + np.sqrt(1 - c**2)*u

		print(np.corrcoef(corr_u[:u.shape[0], :u.shape[1]].flatten(), u.flatten()))

		U = np.fft.fftn(u)
		GU = self.sqrtFFTQ * U

		self.Y = np.real(np.fft.ifftn(GU)) 

		if not self.periodic:
			# readjust domainsize to correct size (--> no boundary effects...)
			gridslice = [slice(0,(self.domainsize.squeeze()-self.cutoff)[i],1)
													  for i in range(self.ndim)]
			self.Y = self.Y[tuple(gridslice)]
			self.Y = self.Y.reshape(self.domainsize.squeeze()-self.cutoff)

		return self.Y


# # EXAMPLES
# # 1: simple example
# domainsize = (200, 150)
# covmod = '1.0 Exp(13.5)'

# fftma = FFTMA(domainsize=domainsize, covmod=covmod)

# plt.figure()
# plt.imshow(fftma.simnew(), origin='lower', interpolation='nearest', cmap='jet')
# plt.show()


# # 2: example using linear combinations of normal random numbers
# t_s = np.linspace(0, np.pi*2, 12) 
# xsample = np.array((np.cos(t_s),np.sin(t_s))).T
# u1 = np.random.standard_normal(size=fftma.sqrtFFTQ.shape)
# u2 = np.random.standard_normal(size=fftma.sqrtFFTQ.shape)
# rns = np.concatenate((u1[np.newaxis,...], u2[np.newaxis, ...]))

# for x in xsample:

#     field = fftma.sim_on_circle(x, rns)
	
#     plt.figure()
#     plt.imshow(field, origin='lower', interpolation='nearest', cmap='jet', vmin=-3.3, vmax=3.3)
#     plt.colorbar()
# plt.show()

# # 3: example using linear combinations of normal random numbers and binary mask for local changes  
# H = np.ones(fftma.sqrtFFTQ.shape)
# H[:75, :75] *= 0

# for x in xsample:

#     field = fftma.sim_on_circle(x, rns, H=H)
	
#     fig, ax = plt.subplots()
#     ax.imshow(field, origin='lower', interpolation='nearest', cmap='jet', vmin=-3.3, vmax=3.3)
#     ax.add_patch(patches.Rectangle((0,0), 75, 75, edgecolor='black', linewidth=2, fill=False))
# plt.show()



# # 4: example using linear combinations of normal random numbers and v-transformation
# m = 0.9
# k = 1.9
# vinv = False

# for x in xsample:

#     field = fftma.sim_on_circle_vtrans(x, rns, m=m, k=k, inv=vinv)
	
#     plt.figure()
#     plt.imshow(field, origin='lower', interpolation='nearest', cmap='jet', vmin=0, vmax=1)
#     plt.colorbar()
# plt.show()


# # 5: example using linear combinations of normal random numbers, v-transformation and binary mask
# m = 0.9
# k = 1.9
# vinv = False

# for x in xsample:

#     field = fftma.sim_on_circle_vtrans(x, rns, m=m, k=k, inv=vinv, H=H)
	
#     fig, ax = plt.subplots()
#     ax.imshow(field, origin='lower', interpolation='nearest', cmap='jet', vmin=0, vmax=1)
#     ax.add_patch(patches.Rectangle((0,0), 75, 75, edgecolor='black', linewidth=2, fill=False))
# plt.show()


# # 6: 3D test
# domain = (30, 30, 30)
# covmod = '1.0 Exp(4.)'
# fftma = FFTMA(domainsize=domain, covmod=covmod)
# field3d = fftma.simnew()

# xyz = np.mgrid[[slice(0 , n, 1) for n in domain]].reshape(3,-1).T

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=field3d.flatten(), cmap='jet')
# plt.show()

# # 7: field with correlation to eg a DEM 
# # the DEM has to be inverted first to get the underlying
# # random numbers to which the correlation can be determined,
# # simulate a 'DEM' field first as an example
# np.random.seed(123)
# domainsize = (200, 150)
# covmod = '0.01 Nug(0.0) + 0.99 Sph(57.5)'
# fftma = FFTMA(domainsize=domainsize, covmod=covmod)
# dem = fftma.simnew_fullsize()

# # invert dem for random numbers u
# fftma = FFTMA(domainsize=domainsize, covmod=covmod)
# corr_u = fftma.invert_for_rn(dem)

# # simulate new fields (like rain) with a certain correlation to DEM
# covmod = '1.0 Sph(27.5)'
# fftma = FFTMA(domainsize=domainsize, covmod=covmod)

# cf = fftma.sim_with_correlation(corr_u, c=0.65)
# plt.figure()
# plt.imshow(dem[:200, :150], cmap='jet')
# plt.figure()
# plt.imshow(cf, cmap='jet')
# plt.show()

# # 8: example using linear combinations of normal random numbers
# # with correlation to other field
# c = 0.65
# t_s = np.linspace(0, np.pi*2, 8) 
# xsample = np.array((np.cos(t_s),np.sin(t_s))).T
# u1 = np.random.standard_normal(size=fftma.sqrtFFTQ.shape)
# u2 = np.random.standard_normal(size=fftma.sqrtFFTQ.shape)
# rns = np.concatenate((u1[np.newaxis,...], u2[np.newaxis, ...]))

# for x in xsample:

# 	field = fftma.sim_on_circle_with_correlation(x, rns, corr_u, c=c)
# 	print(np.corrcoef(dem[:200,:150].flatten(), field.flatten()))
# 	plt.figure()
# 	plt.imshow(field, origin='lower', interpolation='nearest', cmap='jet', vmin=-3.3, vmax=3.3)
# 	plt.colorbar()
# plt.show()

#9: example with anisotropy 2D
# domainsize = (200, 200)
# covmod = '1.0 Sph(20.5)'
# fftma = FFTMA(domainsize=domainsize, covmod=covmod, anisotropy=(1, 0.2, 27)) # that means the first dimension has the range defined in covmod, second dim has 0.2 times that range
# field = fftma.simnew()

# plt.figure()
# plt.imshow(field, interpolation='nearest', cmap='jet', vmin=-3.5, vmax=3.5)
# plt.show()

# # 10: example with anisotropy 3D
# domainsize = (80, 60, 50)
# covmod = '1.0 Sph(11.5)'
# fftma = FFTMA(domainsize=domainsize, covmod=covmod, anisotropy=(0.1, 0.5, 1, 45,25)) # variogram range will be smaller in z and y directions (and then rotated)
# field = fftma.simnew()

# plt.figure()
# plt.imshow(field[20, : , :], interpolation='nearest', cmap='jet', vmin=-3.5, vmax=3.5)
# plt.figure()
# plt.imshow(field[:, 20 , :], interpolation='nearest', cmap='jet', vmin=-3.5, vmax=3.5)
# plt.figure()
# plt.imshow(field[: , :, 20], interpolation='nearest', cmap='jet', vmin=-3.5, vmax=3.5)
# plt.show()
# xyz = np.mgrid[[slice(0,n,1) for n in domainsize]].reshape(3,-1).T
# plt.figure()
# axfield = plt.subplot2grid((1,1), (0,0), projection='3d', rowspan=1)
# axfield.scatter(xyz[:,0],xyz[:,1],xyz[:,2], c=field.flatten(), cmap='jet', vmin=-3.5, vmax=3.5, marker='o',linewidth=0)
# plt.show()