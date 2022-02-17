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

from pandas import cut
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
			for i, dim in enumerate(domainsize):
				if self.anisotropy == False:
					tsize = dim + cutoff
				else:
					tsize = dim + cutoff * self.anisotropy[i]
				# find closest multiple of 8 that is larger than tsize
				m8 = int(np.ceil(tsize/8.)*8.)
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
		axeslist = []
		for d in range(self.ndim-1):
			axeslist.append((d, self.ndim-1))

		for d in range(self.ndim-1):
			angle = self.anisotropy[self.ndim+d]
			nQ = scipy.ndimage.rotate(nQ, angle, axes=axeslist[d], reshape=False)
		nQ = np.fft.fftshift(nQ)
		FFTQ = np.abs(np.fft.fftn(nQ))
		self.sqrtFFTQ = np.sqrt(FFTQ)






