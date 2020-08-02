r"""This code applies the moments accountant (MA), Dual and Central Limit 
Theorem (CLT) to estimate privacy budget of an iterated subsampled 
Gaussian Mechanism (either uniformly or by Poisson subsampling). 
The mechanism's parameters are controlled by flags.

Example:
  compute_muP
    --N=60000 \
    --batch_size=256 \
    --noise_multiplier=1.3 \
    --epochs=15

The output states that DP-optimizer satisfies 0.227-GDP.
"""

import numpy as np
from scipy.stats import norm
from scipy import optimize

# Total number of examples:N
# batch size:batch_size
# Noise multiplier for DP-SGD/DP-Adam:noise_multiplier
# current epoch:epoch
# Target delta:delta

# Compute mu from uniform subsampling
def compute_muU(epoch,noise_multi,N,batch_size):
    T=epoch*N/batch_size
    c=batch_size*np.sqrt(T)/N
    return(np.sqrt(2)*c*np.sqrt(np.exp(noise_multi**(-2))*norm.cdf(1.5/noise_multi)+3*norm.cdf(-0.5/noise_multi)-2))

# Compute mu from Poisson subsampling
def compute_muP(epoch,noise_multi,N,batch_size):
    T=epoch*N/batch_size
    return(np.sqrt(np.exp(noise_multi**(-2))-1)*np.sqrt(T)*batch_size/N)
    
# Dual between mu-GDP and (epsilon,delta)-DP
def delta_eps_mu(eps,mu):
    return norm.cdf(-eps/mu+mu/2)-np.exp(eps)*norm.cdf(-eps/mu-mu/2)

# inverse Dual
def eps_from_mu(mu,delta):
    def f(x):
        return delta_eps_mu(x,mu)-delta    
    return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root

# inverse Dual of uniform subsampling
def compute_epsU(epoch,noise_multi,N,batch_size,delta):
    return(eps_from_mu(compute_muU(epoch,noise_multi,N,batch_size),delta))

# inverse Dual of Poisson subsampling
def compute_epsP(epoch,noise_multi,N,batch_size,delta):
    return(eps_from_mu(compute_muP(epoch,noise_multi,N,batch_size),delta))

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

# Compute epsilon by MA
def compute_epsilon(epoch,noise_multi,N,batch_size,delta):
  """Computes epsilon value for given hyperparameters."""
  orders = [1 + x / 10. for x in range(1, 100)] + list(np.arange(12, 60,0.2))+list(np.arange(60,100,1))
  sampling_probability = batch_size / N
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=noise_multi,
                    steps=epoch*N/batch_size,
                    orders=orders)
  return get_privacy_spent(orders, rdp, target_delta=delta)[0]