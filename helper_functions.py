""" Auxilliary functions that help run the model """

from TBLDA import *

def check_convergence(losses, epoch, epsilon):

    """
    Evaluates convergence criteria for the model run

    Args:
        losses: ELBO values from each epoch

        epoch: current epoch value

        epsilon: the upper bound for the proportional ELBO change over the previous 1000 iterations

    Returns:
        Boolean value whether convergence criteria has been reached    
    """

    elbo_prev_1000 =  np.mean(losses[(epoch - 1000):epoch])
    elbo_prev_2000 =  np.mean(losses[(epoch - 2000):epoch])
    elbo_penultimate_1000 = np.mean(losses[(epoch - 2000):(epoch - 1000)])
    delta_loss = (elbo_penultimate_1000 - elbo_prev_1000)) / elbo_prev_2000

    if delta_loss <= epsilon:
        pyro.get_param_store().save(( str(len(losses)) + '_epochs.save'))
        with open((str(len(losses)) + '_epochs_loss.data'), 'wb') as filehandle:
            pickle.dump(losses, filehandle)

        return(True)

    return(False)


def import_data(expr_f, geno_f, beta_f, tau_f, samp_map_f, f_delim):

    """
    Loads data matrices

    Args:
        expr_f: File containing expression count data

        geno_f: File containing minor allele counts [0,1,2]

        beta_f: File containing estimated beta matrix (ancestry topics)

        tau_f: File containing estimated tau matrix (individual ancestry proportions)

        samp_map_f: File containing [samples x individuals] matrix where each row has a single
                         1 coded at the position of the donor individual
   
        f_delim: Delimiter character for reading in all files
 
    Returns:
        x: [samples x genes] pytorch tensor of expression counts

        y: [snps x individuals] pytorch tensor of minor allele counts [0,1,2]

        anc_portion: Estimated ancestral structure (genotype-specific space; product of zeta and gamma)

        sample_ind_matrix: [samples x individuals] pytorch indicator tensor where each row has a single
                         1 coded at the position of the donor individual

    """

    x = torch.from_numpy(pd.read_csv(expr_f, delimiter=f_delim).to_numpy(dtype='float32'))
    y = pd.read_csv(geno_f, delimiter=f_delim)
    y = torch.from_numpy(y.to_numpy(dtype='int8'))
    anc_loadings = torch.from_numpy(np.genfromtxt(beta_f, delimiter=f_delim))
    anc_facs = torch.from_numpy(np.genfromtxt(tau_f, delimiter=f_delim)).t()
    sample_inds = torch.from_numpy(np.loadtxt(samp_map_f, skiprows=1, dtype='int64'))
    sample_ind_matrix = torch.zeros([x.shape[0], y.shape[1]])
    
    for sample in range(x.shape[0]):
        sample_ind_matrix[cell, sample_inds[sample].item()] = 1

    anc_portion = torch.mm(anc_loadings, anc_facs)
    
    return(x, y, anc_portion, sample_ind_matrix) 
    
