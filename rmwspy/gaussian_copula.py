#-------------------------------------------------------------------------------
# Name:        Theoretical gaussian copula
#
# Author:      Dr.-Ing. S. Hoerning
#
# Created:     02.05.2018, Centre for Natural Gas, EAIT,
#              The University of Queensland, Brisbane, QLD, Australia
#-------------------------------------------------------------------------------

import numpy as np
import scipy.stats as st
import numexpr

def stdnormpdf(x):
    return 1./(2*np.pi)**0.5 * np.exp(-x**2/2.)

def stdnormcdf_approx(x):
    return numexpr.evaluate('1./(1. + exp(-0.07056*x**3 - 1.5976*x))')

def multivariate_normal_copula_pdf(u, Cov):
    '''
    u... u-space point coordinates
    [ [u11,u12,u13,...],
      [u21,u22,u23,...],
      ...
    ]
    Cov... covariances between any two coordinates
    '''
    u    = np.array(u)                  
    Cov  = np.array(Cov)               
    dim  = Cov.shape[-1]              

    if np.ndim(u) == 1:
        u.shape = (1, dim)

    x = st.norm.ppf(u)

    vorfaktor = 1.0 / ((2*np.pi)**(dim/2.0) * np.linalg.det(Cov)**0.5)
    exponent = -0.5 * (((np.tensordot(x, np.linalg.inv(Cov), axes=1)) * x).sum(axis=1))

    # avoid numerical errors
    maxexp = 500
    exponent = np.where(exponent<-maxexp, -maxexp, exponent)
    exponent = np.where(exponent> maxexp,  maxexp, exponent)

    fn = vorfaktor * np.exp(exponent)

    f = stdnormpdf(x)
    c =  fn / np.prod(f, axis=1)

    return c



