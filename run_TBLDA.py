""" Reads in data and runs inference for the TBLDA model """

from def_params import *
from TBLDA import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("expr_f")
parser.add_argument("geno_f")
parser.add_argument("zeta_f")
parser.add_argument("gamma_f")
parser.add_argument("samp_map_f")
parser.add_argument("seed")
parser.add_argument("K")
args = parser.parse_args()

if __name__ == "__main__":




