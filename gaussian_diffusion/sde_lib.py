import abc
import torch
import numpy as np


class SDE(abc.ABC):
  def __init__(self):
    super().__init__()

  @abc.abstractmethod
  def marginal_prob_mean(self, x, t):
    """Parameters to determine the mean of distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def marginal_prob_var(self, x, t):
    """Parameters to determine the variance of the distribution of the SDE, $p_t(x)$."""
    pass

class LinearSDE(SDE):
  
  def __init__(self,beta):
    self.f = lambda x,t : -beta*t*x/2
    self.g = lambda t : (beta*t)**.5
    self.T = 1
    self.beta = beta


  def marginal_prob_mean(self, x0, t):
    var = torch.exp(-self.beta*t*t/4)
    return var * x0

  def marginal_prob_var(self, t):
    var = torch.exp(-self.beta*t*t/4)
    return 1-var


  def generate_samples_reverse(self, score_network: torch.nn.Module, shape, nsamples: int, ode=False) -> torch.Tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    x_t = torch.randn([nsamples] + shape, device=device)
    print(x_t.shape)
    time_pts = torch.linspace(1, 0, 1000).to(device)

    num_trajectories = 10
    trajectories = x_t[:num_trajectories,:]
    for i in range(len(time_pts) - 1):
        t = time_pts[i]
        dt = time_pts[i + 1] - t
        score = score_network(x_t,t.expand(x_t.shape[0], 1)).detach()

        prob_flow_constant = .5 if ode else 1
        tot_drift = self.f(x_t,t) - self.g(t)**2 * score*prob_flow_constant
        tot_diffusion = self.g(t)

        # euler-maruyama step
        x_t = x_t + tot_drift * dt 
        if ode == False:
          randomness = tot_diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5
          x_t += randomness
        trajectories = torch.cat((trajectories, x_t[:num_trajectories,:]),dim=1)
    x_t = x_t.cpu()
    return x_t, trajectories
  
  def generate_samples_reverse_fft(self, score_network: torch.nn.Module, shape, nsamples: int, ode=False) -> torch.Tensor:
    # This score function is in the data space
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_t = torch.randn([nsamples] + shape,device=device)
    time_pts = torch.linspace(1, 0, 100).to(device)

    num_trajectories = 10
    trajectories = x_t[:num_trajectories,:]
    for i in range(len(time_pts) - 1):
        t = time_pts[i]
        dt = time_pts[i + 1] - t
        
        x_data_t = torch.zeros_like(x_t,device=device)
        for i, el in enumerate(x_t):
          x_data_t[i] = torch.fft.irfft(getInverseTransform(el,dimension),norm="forward",n=dimension)  
        # A^-1 y

        score = score_network(x_data_t,t.expand(x_data_t.shape[0], 1)).detach() 
        score = getTransform(torch.fft.rfft(score,norm="forward"),dimension) #A*score

        tot_drift = self.f(x_t,t) - self.g(t)**2 * score
        tot_diffusion = self.g(t)

        # euler-maruyama step
        x_t = x_t + tot_drift * dt
        if ode == False:
          x_t*=2
          randomness = tot_diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5
          x_t += randomness
        
        
        trajectories = torch.cat((trajectories, x_t[:num_trajectories,:]),dim=1)

    return x_t, trajectories

def getTransform(ft,dim):
  if dim == 2:
    return ft.real
  else:
    n = ft.shape[0]
    beg = ft[:,0:1].real
    if dim%2 != 0: 
      middle = torch.view_as_real(ft[:,1:]).reshape(n-1,dim)
      end = ft[:,-1:].real
      return torch.cat((beg,middle,end),dim=1)
    else:
      middle = torch.view_as_real(ft[:,1:-1]).reshape(n-2,dim)
      return torch.cat((beg,middle),dim=1)

def getInverseTransform(ft,dim):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  newT = torch.zeros(dim,dtype=torch.cfloat,device=device)
  k = 0
  for i,val in enumerate(ft):
      if i == 0:
          newT[k] = val
          k+=1
          continue
      if(i%2 == 1):
          newT[k] += val 
      else :
          newT[k] += torch.complex(torch.tensor(0.,device=device),val)
          k+=1
  return newT
