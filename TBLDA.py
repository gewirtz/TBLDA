""" Telescoping bimodal latent Dirichlet allocation model"""

import numpy as np
import pickle
import math
import pandas as pd
import scipy as sp
import scanpy as sc
import scanpy.external as sce
import anndata as ad
import h5py as h5
import copy
from os import path
import sys

import torch
import torch.distributions as tdist
from torch.autograd import grad
import torch.optim as optim
from torch.distributions import constraints

import pyro
import pyro.optim as poptim
import pyro.distributions as dist
from pyro.ops.indexing import Vindex
from pyro.infer import SVI, Trace_ELBO

sc.settings.verbosity = 2
sc.logging.print_versions()
pyro.enable_validation(True)
pyro.set_rng_seed(1)


@pyro.poutine.scale(scale=1.0e-6)
def TBLDA_model(hps, mps, x, y, anc_portion, cell_ind_matrix):
    """
    Model for TBLDA

    Args:
        hps: Hyperparams object with model hyperparameters

        mps: Modelparams object containing overall model parameters

          x: [samples x genes] pytorch tensor of expression counts

          y: [snps x individuals] pytorch tensor of minor allele counts [0,1,2]

        anc_portion: Estimated ancestral structure (genotype-specific space; product of zeta and gamma)

        cell_ind_matrix: [samples x individuals] pytorch indicator tensor where each row has a single
			 1 coded at the position of the donor individual

    Returns:
        NA    
    """

    # declare plates
    snp_plt = pyro.plate('snps', mps.n_snps, dim=-2)
    ind_plt = pyro.plate('inds', mps.n_inds)
    k_b_plt = pyro.plate('k_b', mps.k_b)
    cell_plt = pyro.plate('cells', mps.n_samples)
    gene_plt = pyro.plate('genes', mps.n_genes)

    # global
    with k_b_plt:
        lambda_g = pyro.sample("lambda_g", dist.Dirichlet(hps.xi)) # [k_b, n_genes]

    with snp_plt, k_b_plt:
        lambda_s = pyro.sample("lambda_s", dist.Beta(hps.zeta, hps.gamma)) # [n_snps, k_b]

    # local - cell level. 
    with cell_plt:
        phi = pyro.sample("phi", dist.Dirichlet(hps.sigma))
        pi_g = torch.mm(phi, lambda_g) # [n_samples, n_genes]
        pyro.sample('x', dist.Multinomial(probs=pi_g, validate_args=False), obs=x)

    alpha = pyro.sample('alpha', dist.Uniform(hps.delta, hps.mu))

    phi_ind = torch.mm(phi.t(), cell_ind_matrix) #[k_b, n_inds]
    phi_ind = phi_ind / torch.sum(phi_ind, dim=0).view([1, mps.n_inds])
    pi_s = (alpha * anc_portion) + ((1 - alpha) * torch.mm(lambda_s, phi_ind)) # [n_snps, n_inds]

    with snp_plt:
        pyro.sample('y', dist.Binomial(2, pi_s), obs=y.float()) # [n_snps, n_inds]


@pyro.poutine.scale(scale=1.0e-6)
def TBLDA_guide(hps, mps, x, y, anc_portion, cell_ind_matrix):
    """
    Guide Function for TBLDA

    Args:
        hps: Hyperparams object with model hyperparameters

        mps: Modelparams object containing overall model parameters

          x: [samples x genes] pytorch tensor of expression counts

          y: [snps x individuals] pytorch tensor of minor allele counts [0,1,2]

        anc_portion: Estimated ancestral structure (genotype-specific space; product of zeta and gamma)

        cell_ind_matrix: [samples x individuals] pytorch indicator tensor where each row has a single
                         1 coded at the position of the donor individual

    Returns:
        NA    
    """

    # declare plates
    snp_plt = pyro.plate('snps', mps.n_snps, dim=-2)
    ind_plt = pyro.plate('inds', mps.n_inds)
    k_b_plt = pyro.plate('k_b', mps.k_b)
    cell_plt = pyro.plate('cells', mps.n_samples)
    gene_plt = pyro.plate('genes', mps.n_genes)

    xi = pyro.param("xi", torch.ones([mps.k_b, mps.n_genes]), constraint=constraints.positive)
    with k_b_plt:
        lambda_g = pyro.sample("lambda_g", dist.Dirichlet(xi)) # [k_b, n_genes]


    # local - cell level. 
    sigma = pyro.param("sigma", torch.ones([mps.n_samples, mps.k_b]), constraint=constraints.positive)
    with cell_plt:
        phi = pyro.sample("phi", dist.Dirichlet(sigma)) # [n_samples, k_b]

    alpha_p = pyro.param("alpha_p", torch.tensor([0.1]), constraint=constraints.interval(hps.delta, hps.mu))
    alpha = pyro.sample("alpha", dist.Delta(alpha_p))

    zeta = pyro.param("zeta", torch.ones([mps.n_snps, mps.k_b]), constraint=constraints.positive)
    gamma = pyro.param("gamma", torch.ones([mps.n_snps, mps.k_b]), constraint=constraints.positive)
    with snp_plt, k_b_plt:
        lambda_s = pyro.sample("lambda_s", dist.Beta(zeta, gamma)) # [n_snps, k_b]


def run_vi(hps, mps, lr, max_epochs, seed, write_its, verbose, check_conv_its=25, epsilon=1e-4):
    """

    """
    
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    opt1 = poptim.Adam({"lr": lr})
    svi = SVI(TBLDA_model, TBLDA_guide, opt1, loss=Trace_ELBO())
    losses = []

    for epoch in range(max_epochs):
        if verbose:
            print('EPOCH ' + str(epoch),flush=True)
        elbo = svi.step(hps, mps, x, y, anc_portion, cell_ind_matrix)
        losses.append(elbo)
        # only start checking for convergence after 5000 epochs
        if (epoch % check_conv_its == 0) and (epoch > 5000):
            converge = check_convergence(losses, epoch, epsilon)
            if converge:
                break
        # write intermediate output
        if((epoch>0) and (epoch%write_its==0)):
                pyro.get_param_store().save(('results_' + str(epoch) + '_epochs.save'))
                with open('results_' + str(epoch) + '_epochs_loss.data'), 'wb') as filehandle:
                    pickle.dump(losses, filehandle)
                # remove old files
                if epoch > write_its:
                    os.remove('results_' + str(epoch - write_its) + '_epochs.save')
                    os.remove('results_' + str(epoch - write_its) + '_epochs_loss.data')




