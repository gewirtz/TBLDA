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

class TBLDA():

    """
    Define TBLDA model parameters and hyperparameters

    k_b : Number of latent topics
    
    anc_portion: Estimated ancestral structure (genotype-specific space; product of zeta and gamma)

    sample_ind_matrix: [samples x individuals] pytorch indicator tensor where each row has a single
                         1 coded at the position of the donor individual

    """
    def __init__(self, n_inds, n_genes, n_snps, k_b, n_samples, \
                 anc_portion, sample_ind_matrix, \
                 sigma=1.0, xi=1.0, psi=1.0, delta=0.05, \
                 mu=0.85, gamma=1.0):
        super().__init__()
        self.n_inds = n_inds
        self.n_genes = n_genes
        self.n_snps = n_snps
        self.k_b = k_b
        self.n_samples = n_samples
        self.anc_portion = anc_portion
        self.alpha_lim = alpha_lim
        self.sample_ind_matrix = sample_ind_matrix
        self.psi = psi
        self.mu = mu
        self.delta = delta
        self.zeta = torch.tensor([zeta])
        self.sigma = torch.ones([k_b]) * sigma
        self.xi = torch.ones([n_genes]) * xi
        self.gamma = torch.tensor([gamma])

    
    """
    Model for TBLDA

    Args:

          x: [samples x genes] pytorch tensor of expression counts

          y: [snps x individuals] pytorch tensor of minor allele counts [0,1,2]

    """
    @pyro.poutine.scale(scale=1.0e-6)
    def model(self, x, y):
        # declare plates
        snp_plt = pyro.plate('snps', self.n_snps, dim=-2)
        ind_plt = pyro.plate('inds', self.n_inds)
        k_b_plt = pyro.plate('k_b', self.k_b)
        sample_plt = pyro.plate('samples', self.n_samples)
        gene_plt = pyro.plate('genes', self.n_genes)

        # global
        with k_b_plt:
            lambda_g = pyro.sample("lambda_g", dist.Dirichlet(self.xi)) # [k_b, n_genes]

        with snp_plt, k_b_plt:
            lambda_s = pyro.sample("lambda_s", dist.Beta(self.zeta, self.gamma)) # [n_snps, k_b]

        # local - cell level. 
        with sample_plt:
            phi = pyro.sample("phi", dist.Dirichlet(self.sigma))
            pi_g = torch.mm(phi, lambda_g) # [n_samples, n_genes]
            pyro.sample('x', dist.Multinomial(probs=pi_g, validate_args=False), obs=x)

        alpha = pyro.sample('alpha', dist.Uniform(self.delta, self.mu))

        phi_ind = torch.mm(phi.t(), self.sample_ind_matrix) #[k_b, n_inds]
        phi_ind = phi_ind / torch.sum(phi_ind, dim=0).view([1, self.n_inds])
        pi_s = (alpha * self.anc_portion) + ((1 - alpha) * torch.mm(lambda_s, phi_ind)) # [n_snps, n_inds]

        with snp_plt:
            pyro.sample('y', dist.Binomial(2, pi_s), obs=y.float()) # [n_snps, n_inds]


    """
    Guide Function for TBLDA

    Args:

          x: [samples x genes] pytorch tensor of expression counts

          y: [snps x individuals] pytorch tensor of minor allele counts [0,1,2]


    """
    @pyro.poutine.scale(scale=1.0e-6)
    def guide(self, x, y):
        # declare plates
        snp_plt = pyro.plate('snps', self.n_snps, dim=-2)
        ind_plt = pyro.plate('inds', self.n_inds)
        k_b_plt = pyro.plate('k_b', self.k_b)
        sample_plt = pyro.plate('samples', self.n_samples)
        gene_plt = pyro.plate('genes', self.n_genes)

        xi = pyro.param("xi", torch.ones([self.k_b, self.n_genes]), constraint=constraints.positive)
        with k_b_plt:
            lambda_g = pyro.sample("lambda_g", dist.Dirichlet(xi)) # [k_b, n_genes]


        # local - sample level. 
        sigma = pyro.param("sigma", torch.ones([self.n_samples, self.k_b]), constraint=constraints.positive)
        with sample_plt:
            phi = pyro.sample("phi", dist.Dirichlet(sigma)) # [n_samples, k_b]

        alpha_p = pyro.param("alpha_p", torch.tensor([0.1]), constraint=constraints.interval(self.delta, self.mu))
        alpha = pyro.sample("alpha", dist.Delta(alpha_p))

        zeta = pyro.param("zeta", torch.ones([self.n_snps, self.k_b]), constraint=constraints.positive)
        gamma = pyro.param("gamma", torch.ones([self.n_snps, self.k_b]), constraint=constraints.positive)
        with snp_plt, k_b_plt:
            lambda_s = pyro.sample("lambda_s", dist.Beta(zeta, gamma)) # [n_snps, k_b]

