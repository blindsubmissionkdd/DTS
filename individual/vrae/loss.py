import torch
import torch.nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import numpy as np
import math
EPS = 1e-08



def total_kld(q_z, p_z):
    return torch.sum(kl_divergence(q_z, p_z))


def flow_kld(q_z, p_z, z, z0, sum_log_j):
    batch_size = z.size(0)
    e_log_pz = -torch.sum(p_z.entropy()) / batch_size
    return total_kld(q_z, p_z) / batch_size - torch.mean(sum_log_j)


def compute_nll(q_z, p_z, z, z0, sum_log_j, re_loss):
    batch_size = z.size(0)
    z_dim = z.size(1)
    e_log_pz = p_z.log_prob(z).sum(-1).mean()
    e_log_px_z = -re_loss / batch_size
    e_log_qz = -torch.sum(q_z.entropy()) / batch_size
    return -(e_log_px_z + e_log_pz - e_log_qz + torch.mean(sum_log_j))


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
    return m + torch.log(sum_exp)


def mutual_info(q_z, p_z, z):
    batch_size = z.size(0)
    z_dim = z.size(1)
    # log_qz = q_z.log_prob(
    # z.unsqueeze(1).expand(-1, batch_size, -1))
    log_qz = q_z.log_prob(z).sum(-1).unsqueeze(1).expand(batch_size, -1)
    e_log_q_zx = -torch.sum(q_z.entropy()) / batch_size
    e_log_qz = (log_sum_exp(log_qz, dim=1) - np.log(batch_size)).mean()

    return e_log_q_zx - e_log_qz


def mutual_info_flow(q_z, p_z, z, z0, sum_log_j):
    batch_size = z.size(0)
    z_dim = z.size(1)
    # log_flow_qz = q_z.log_prob(z).sum(-1) + sum_log_j
    log_flow_qz = q_z.log_prob(z0).sum(-1)
    # compute E_xE_{q(z'|x)}log(q(z'|x))
    e_log_q_flow_zx = torch.sum(log_flow_qz) / (batch_size)
    e_log_flow_qz = (log_sum_exp(log_flow_qz.unsqueeze(1).expand(batch_size, -1), dim=1) - np.log(batch_size)).mean()

    return e_log_q_flow_zx - e_log_flow_qz + sum_log_j.mean()


def gaussian_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-10 * kernel)


def compute_mmd(p, q, kernel='g'):
    if kernel == 'g':
        # use gaussian kernel
        x = q.rsample()
        y = p.sample()
        x_kernel = gaussian_kernel(x, x)
        y_kernel = gaussian_kernel(y, y)
        xy_kernel = gaussian_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    elif kernel == 'im':
        # use im kernel
        mmd = im_kernel(q, p)
    else:
        raise NotImplementedError
    return mmd


def im_kernel(q_z, p_z, z_var=1):
    sample_qz = q_z.rsample()
    sample_pz = p_z.sample()
    batch_size = sample_pz.size(0)
    z_dim = sample_qz.size(1)
    Cbase = 2 * z_dim * z_var

    norms_pz = torch.sum(sample_pz.pow(2), dim=1, keepdim=True)
    dotprobs_pz = torch.matmul(sample_pz, sample_pz.t())
    distances_pz = norms_pz + norms_pz.t() - 2. * dotprobs_pz

    norms_qz = torch.sum(sample_qz.pow(2), dim=1, keepdim=True)
    dotprobs_qz = torch.matmul(sample_qz, sample_qz.t())
    distances_qz = norms_qz + norms_qz.t() - 2. * dotprobs_qz

    dotprobs = torch.matmul(sample_qz, sample_pz.t())
    distances = norms_qz + norms_pz.t() - 2. * dotprobs

    stat = 0.
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = Cbase * scale
        res1 = C / (C + distances_qz)
        res1 += C / (C + distances_pz)
        res1 = res1 * (1 - torch.eye(batch_size, device=sample_pz.device))
        res1 = torch.sum(res1) / (batch_size * batch_size - batch_size)
        res2 = C / (C + distances)
        res2 = torch.sum(res2) * 2. / (batch_size * batch_size)
        stat += res1 - res2
    return stat

def gaussian_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel)

def compute_mmd(x, y):
    x_kernel = gaussian_kernel(x, x)
    y_kernel = gaussian_kernel(y, y)
    xy_kernel = gaussian_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd


def e_log_p(posterior, prior):
    """
    Analytically calculate the expected value of prior under the posterior.
    D(q||p) = - H(q) - e_log_p(q, p)
    """
    return - torch.sum(kl_divergence(posterior, prior) + posterior.entropy())


def logsumexp(inputs, dim):
    s, _ = inputs.max(dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    return outputs


def kld_decomp(posterior, prior, z):
    batch_size = z.size(0)
    code_size = z.size(1)
    log_probs = posterior.log_prob(
        z.unsqueeze(1).expand(-1, batch_size, -1))
    e_log_qzx = - torch.sum(posterior.entropy()) / batch_size
    e_log_qz = torch.sum(
        logsumexp(log_probs.sum(2), dim=1)
    ) / batch_size - np.log(batch_size)
    e_log_qzi = torch.sum(
        logsumexp(log_probs, dim=1)
    ) / batch_size - code_size * np.log(batch_size)
    e_log_pz = e_log_p(posterior, prior) / batch_size
    mutual_info = e_log_qzx - e_log_qz
    total_corr = e_log_qz - e_log_qzi
    dimwise_kl = e_log_qzi - e_log_pz
    return mutual_info, total_corr, dimwise_kl

def bow_recon_loss(outputs, targets):
    """
    Note that outputs is the bag-of-words log likelihood predictions.
    targets is the target counts.

    """
    return - torch.sum(targets * outputs)


def total_kld(posterior, prior):
    """
    Use pytorch built-in distributions to calculate kl divergence.

    """
    return torch.sum(kl_divergence(posterior, prior))

def get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, is_mss=True):
    batch_size, hidden_dim = latent_sample.shape
    # calculate log q(z|x)
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)
    # calculate log p(z)
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)
    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)
    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)
    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)
    return log_pz, log_qz, log_prod_qzi, log_q_zCx



def matrix_log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian for all combination of bacth pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.

    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).

    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).

    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).

    batch_size: int
        number of training images in the batch
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian.

    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.

    mu: torch.Tensor or np.ndarray or float
        Mean.

    logvar: torch.Tensor or np.ndarray or float
        Log variance.
    """
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density

def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Calculates a log importance weight matrix

    Parameters
    ----------
    batch_size: int
        number of training images in the batch

    dataset_size: int
    number of training images in the dataset
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()

def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed

def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size,1,dim).repeat(1, y_size,1)
    tiled_y = y.view(1,y_size,dim).repeat(x_size, 1,1)

    return torch.exp(-torch.mean((tiled_x - tiled_y)**2,dim=2)/dim*1.0)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)