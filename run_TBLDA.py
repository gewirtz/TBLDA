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
parser.add_argument("K", type=int, default=50, \
                    help="Number of latent shared topics")
parser.add_argument("--file_delim", default='tab', choices=['tab','space','comma'],\
                    help="Delimiter for all files")
parser.add_argument("--lr", default=0.05, \
                    help="Learning rate")
parser.add_argument("--max_epochs", default=65100, \
                    help='Maximum number of epochs to run')
parser.add_argument("--write_its", default=50, \
                    help='Write intermediate output every <X> iterations')
args = parser.parse_args()

if args.file_delim == 'tab':
    f_delim = '\t'
elif args.file_delim == 'space':
    f_delim = ' '
else:
    f_delim = ','

# check argument validity
if args.K < 2:
    raise argparse.ArgumentTypeError('Value must be at least 2 (minimum recommended 5)')
if args.lr <= 0 or args.lr >= 1:
    raise argparse.ArgumentTypeError('Learning rate must be between 0 and 1')

# Read in data
x, y, anc_portion, sample_ind_matrix = import_data(args.expr_f, args.geno_f, args.beta_f, \
                                                 args.tau_f, args.samp_map_f, f_delim)


m_params = Modelparams(n_inds = y.shape[1], n_snps = y.shape[0], n_samples = x.shape[0], \
                       n_genes = x.shape[1])
h_params = Hyperparams(m_params)

run_vi(hps=h_params, mps=m_params, lr=args.lr, seed=args.seed, max_epochs=args.max_epochs, \
       write_its=args.write_its)



