# TBLDA
Telescoping Bimodal Latent Dirichlet Allocation

# Overview

# Usage

## Requirements

TBLDA is implemented in Python 3.8. It requires the following packages (used versions are in parentheses):

1. pandas 1.0.5
2. numpy 1.18.5
3. torch 1.9.0+cu102
4. pyro 1.4.0
5. h5py 2.10.0

## Running TBLDA

**Required Input:**

1. `expr_f`: Path to the file containing expression count data with samples as rows and genes as columns
2. `geno_f`: Path to the file containing genotype data in dosage format [0,1,2] with SNPs as rows and individuals as columns
3. `beta_f`: Path to the file containing estimated ancestry topics with SNPs as rows and topics as columns
4. `tau_f`: Path to the file containing ancestry proportions with SNPs as rows and individuals as columns
5. `samp_map_f`: Path to the file containing an L-length vector of individual IDs, where L is the total number of samples and individuals are given an ID in [0,N] where N is the total number of individuals
6. `K`: The number of shared latent factors in the model. We recommend starting with K in between 5 and 150.

**Optional Input:**

1. Seed (`--seed`): Value to seed the random number generator. Defaults to 21.
2. File delimiter (`--file_delim`): Character that separates columns across all files. Defaults to tab.
3. Learning rate (`--lr`): Learning rate for inference. Defaults to 0.05.
4. Maximum number of epochs (`--max_epochs`)
5. Write iterations (`--write_its`): Specifies how often intermediate results are saved (write output every <X> iterations).

  **Output**: 
  
  1. results_<X>_epochs.save: This contains the pyro parameter store with the estimates of xi, sigma, zeta, and gamma.
  2. results_<X>_epochs_loss.data: This contains the loss estimates at every epoch.
