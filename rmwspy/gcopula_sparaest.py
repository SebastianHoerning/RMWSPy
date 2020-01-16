#-------------------------------------------------------------------------------
# Name:        Copula fitting using maximum likelihood
#
# Author:      Dr.-Ing. P. Guthke, Dr.-Ing. S. Hoerning
#
# Created:     02.05.2018, University Stuttgart, Stuttgart, Germany
#                          The University of Queensland, Brisbane, QLD, Australia
# Copyright:   (c) Hoerning 2018
#-------------------------------------------------------------------------------
import numpy as np
import os
import matplotlib.pylab as plt
import scipy.stats as st
import scipy.spatial as sp
import scipy.optimize as opt
import datetime
import covariancefunction as variogram
import gaussian_copula as thcopula
import IPython

def paraest_multiple_tries(
                x,                          # coordinates [n x d]
                u,                          # values in copula space  [n x 1]
                ntries=[6,6],               # [no. of tries with same subsets (diff starting parameters), no. of tries with different subsets]
                n_in_subset=8,              # number of values in subset
                neighbourhood='nearest',    # subset search algorithm
                covmods=['Mat', 'Exp', 'Sph',] ,
                outputfile=None,            # if valid path+fname string --> write output file
                talk_to_me=True,
                ):

    if outputfile != None:
        if os.path.isdir(os.path.dirname(os.path.abspath(outputfile))) == True:
            fout = open(outputfile, 'a')
            fout.write('# ------------------------------------------------------------- #\n')
            fout.write('# ------------------------------------------------------------- #\n')
            fout.write('# OPTIMIZED PARAMETERS: %s - nested --- #\n'%str(datetime.datetime.now()))
            fout.write('# neighbourhood:      "%i" values with "%s"\n'%(n_in_subset, neighbourhood))
            fout.write('# number of different subsets built: %i\n'%(ntries[1]))
            fout.write('# number of tries for each subset selection: %i, but only the one with best likelihood is displayed\n'%(ntries[0]))
            fout.write('# covariance model \n')
            fout.flush()

    # number of spatial models
    nspamods = len(covmods)

    # draw ntries random states for the setting of starting parameters
    randstates_startpar = []
    for i in range(ntries[0]):
        np.random.rand(np.random.randint(10000, 100000))
        state = np.random.get_state()
        randstates_startpar.append(state)

    # draw ntries random states for the building of subsets
    randstates_subsets = []
    for i in range(ntries[0]):
        np.random.rand(np.random.randint(10000, 100000))
        state = np.random.get_state()
        randstates_subsets.append(state)

    # loop over different spatial models
    out = []
    for mod in range(nspamods):
        covmods0    = covmods[mod]

        # lop over different subsets
        out0 = []
        for subset in range(ntries[1]):

            # loop for different starting parameters for one model
            for startings in range(ntries[0]):
                np.random.set_state(randstates_startpar[startings])

                out000 = paraest_g(
                        x,
                        u,
                        n_in_subset=n_in_subset,
                        neighbourhood=neighbourhood,
                        seed = randstates_subsets[subset],
                        covmods = covmods0,
                        outputfile=None,
                        talk_to_me=True,
                        )

                # only take best of optimizations of same subsets
                if startings == 0:
                    out00 = out000
                else:
                    if out000[1] < out00[1]:
                        out00 = out000

            if outputfile != None:
                if os.path.isdir(os.path.dirname(os.path.abspath(outputfile))) == True:
                    fout.write('#\n')
                    # reconstruct parameters
                    p           = out00[0]
                    Likelihood  = out00[1] * -1
                    message     = out00[2]
                    
                    cov_model = reconstruct_parameters(p,covmods0)
                    
                    fout.write('%s \n'%(cov_model))
                    fout.write('# Likelihood: %1.3f\n'%Likelihood)
                    fout.write('# message: %s\n'%message)
                    fout.flush()        

            out0.append(out00)
        out.append(out0)
    if outputfile != None:
        fout.write('# ------------------------------------------------------------- #\n')
        fout.close()
    return out



def paraest_g(
                x,                          # coordinates [n x d]
                u,                          # values in copula space  [n x 1]
                n_in_subset=8,              # number of values in subset
                neighbourhood='nearest',    # subset search algorithm
                seed = None,               
                covmods = ['Mat'],    
                outputfile=None,            
                talk_to_me=True,
                ):
   
    # BUILD SUBSETS  
    if seed != None:
        curstate = np.random.get_state()
        np.random.set_state(seed)

    ind = build_subsets(
                        x,
                        n_in_subset=n_in_subset,
                        how=neighbourhood,
                        talk_to_me=talk_to_me,
                        plot_me=0)

    x0 = x[ind]
    u0 = u[ind]

    if seed != None:
         np.random.set_state(curstate)

    # calc distance matrices in each subset
    d0 = np.zeros((x0.shape[0], x0.shape[1], x0.shape[1]))
    for i in range(x0.shape[0]):
        d0[i] = sp.distance_matrix(x0[i], x0[i])

    
    args = (d0,                 
            u0,                 
            covmods,
            talk_to_me)

    p_bounds=[]   
    Rangebounds = [[(d0[np.where(d0>0)]).min()*2.,  d0.max()*2]]
    p_bounds += Rangebounds

    if covmods == 'Mat':
        Extrabounds = [[0.05, 50.0]]
        p_bounds += Extrabounds
  
    p_bounds = tuple(p_bounds)

    # random starting parameter set
    p_start = []
    for i in range(len(p_bounds)):
        p0 = np.random.rand() * 0.6 + 0.2
        p0 = p0 * (p_bounds[i][1] - p_bounds[i][0]) + p_bounds[i][0]
        p_start.append(p0)

    # start optimization
    out = opt.fmin_l_bfgs_b(Likelihood,
                           p_start,
                           bounds=list(p_bounds),
                           args=args,
                           approx_grad=True,
                           )

    # reconstruct parameters 
    p = out[0]
    cov_models = reconstruct_parameters(p,covmods)
    if outputfile != None:
        if os.path.isdir(os.path.dirname(os.path.abspath(outputfile))) == True:
            Like = -1*out[1]
            fout = open(outputfile, 'a')
            fout.write('# OPTIMIZED PARAMETERS: %s - nested --- #\n'%str(datetime.datetime.now()))
            fout.write('# neighbourhood:      "%i" values with "%s"\n'%(n_in_subset, neighbourhood))
            fout.write('# Likelihood: \n%i\n'%Like)
            fout.write('# message: %s'%out[2])
            fout.write('# covariance model \n')   
            fout.write('%s \n'%(cov_models))
            fout.write('# ------------------------------------------------------------- #\n')
            fout.close()

    return out

def reconstruct_parameters(p, covmods):
    
    counter = 0
   
    cov_model = ''
    Range = p[counter]
    counter += 1
    cov_model += '1.0 %s(%1.12f)'%(covmods, Range)

    if covmods=='Mat':
        Param = p[counter]
        counter += 1
        cov_model += '^%1.12f'%Param    
    cov_models = cov_model

    return cov_models


def Likelihood(p, Ds, us, covmods, talk_to_me=True):

    cov_models = reconstruct_parameters(p, covmods)

    Qs = np.array(variogram.Covariogram(Ds, cov_models))

    # copula densities
    cs = []
    for i in range(us.shape[0]):
        cs.append(thcopula.multivariate_normal_copula_pdf(us[i], Qs[i]))
    cs = np.array(cs)

    # avoid numerical errors
    cs[np.where(cs==0)] = 0.000000000000001

    # log Likelihood
    L = (np.log(cs)).sum()

    return -L   # negative likelihood --> minimization


def build_subsets(  coords,             # coordinate vector [x,y,z]
                    n_in_subset=6,      # number of points in subset
                    how='nearest',      # subset building routine
                    talk_to_me=False,
                    plot_me=False
                    ):

    n_coords  = coords.shape[0]
    n_subsets = int(np.floor(float(n_coords)/n_in_subset))
    n_used    = int(n_subsets*n_in_subset)

    if how == 'random':
        ind = np.arange(n_used)
        np.random.shuffle(ind)
        ind = (ind.reshape((n_subsets, n_in_subset))).astype('int')

    if how == 'nearest':
        ind = [] 
        not_taken = np.ones(coords.shape[0]).astype('bool')

        for subset in range(n_subsets):
            # take one point randomly
            i = np.where(not_taken==True)[0]
            np.random.shuffle(i)
            i_1 = i[0]
            # mark it
            not_taken[i_1] = False

            # calc distances to other coordinates
            d = sp.distance_matrix(coords[i_1][np.newaxis], coords[not_taken])[0]
            i_closest = np.argsort(d)[:n_in_subset-1]

            # retransform indices to coords array
            i_closest = np.arange(coords.shape[0])[not_taken][i_closest]
            not_taken[i_closest] = False

            i_subset = np.concatenate(([i_1], i_closest))
            i_subset = np.sort(i_subset)

            ind.append(i_subset)
        ind = np.array(ind)

    if plot_me == True:
        x = coords[ind]
        plt.figure()
        for i,xy in enumerate(x):
            xy = xy[np.argsort(xy[:,0])]
            plt.plot(xy[:,0], xy[:,1], '.-', alpha=0.5)
        plt.show()

    return ind


