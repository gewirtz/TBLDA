""" Define TBLDA model parameters and hyperparameters """


class Hyperparams():
    """
    Model hyperparameters for symmetrical priors
    """
    def __init__(self, mps,
                 sigma=1.0,
                 xi=1.0,
                 psi=1.0,
                 delta=0.05,
                 mu=0.85,
                 gamma=1.0,):

        self.sigma = torch.ones([mps.k_b]) * sigma
        self.xi = torch.ones([mps.n_genes]) * xi
        self.psi = psi
        self.delta = delta
        self.mu = mu
        self.gamma = gamma


class Modelparams():
    """
    Various relevant model parameters
    """
    def __init__(self, n_inds, n_genes, n_snps, k_b, n_cells):
        self.n_inds = n_inds
        self.n_genes = n_genes
        self.n_snps = n_snps
        self.k_b = k_b
        self.n_samples = n_samples





