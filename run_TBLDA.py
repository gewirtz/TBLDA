""" Reads in data and runs inference for the TBLDA model """

from def_params import *
from TBLDA import *
import argparse

parser = argparse.ArgumentParser(description='Run TBLDA')
parser.add_argument("expr_f", \
                    help="Expression count file with samples as rows")
parser.add_argument("geno_f", \
                    help="Genotype file in dosage format with SNPs as rows")
parser.add_argument("beta_f", \
                    help="File of estimated ancestry topics with SNPs as rows")
parser.add_argument("tau_f", \
                    help="File with ancestry proportions per individual with donors as rows")
parser.add_argument("samp_map_f", \
                    help="File of individual IDs [0,...,N] for each sample")
parser.add_argument("--seed", type=int, default=21,\
                    help="Random seed")
parser.add_argument("K", type=int, default=50,\
                    help="Number of latent shared topics")
parser.add_argument("--file_delim", default='tab', choices=['tab','space','comma'],\
                    help="Delimiter for all files")
args = parser.parse_args()

if args.file_delim == 'tab':
    f_delim = '\t'
elif args.file_delim == 'space':
    f_delim = ' '
else:
    f_delim = ','

# Read in data
x, y, anc_portion, cell_ind_matrix = import_data(args.expr_f, args.geno_f, args.beta_f, \
                                                 args.tau_f, args.samp_map_f, f_delim)






