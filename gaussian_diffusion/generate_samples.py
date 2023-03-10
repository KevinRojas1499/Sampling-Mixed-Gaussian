import numpy as np
from bisect import bisect_left
from torch.distributions.multivariate_normal import MultivariateNormal
import torch

def get_samples_from_mixed_gaussian(c,means,variances,num_samples):
    n = len(c)
    d = means[0].shape[0]
    accum = np.zeros(n)
    accum[0] = c[0]
    for i in range(1,n):
        accum[i] = accum[i-1]+c[i]

    gaussians = [MultivariateNormal(means[i],variances[i]) for i in range(n)]

    samples = torch.zeros(num_samples,d)
    for i in range(num_samples):
        idx = bisect_left(accum,np.random.rand(1)[0])
        samples[i] = gaussians[idx].sample()
    return samples

