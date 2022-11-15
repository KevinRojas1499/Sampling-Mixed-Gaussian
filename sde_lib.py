import abc
import torch

class SDE(abc.ABC):
  def __init__(self, N,T, beta):
    # Right now I just assume thhe function is beta*t
    super().__init__()
    self.N = N
    self.T = T

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
    self.beta = beta


  def marginal_prob_mean(self, x0, t):
    var = torch.exp(-self.beta*t*t/4)
    return torch.mul(var,x0)

  def marginal_prob_var(self, t):
    var = torch.exp(-self.beta*t*t/4)
    return 1-var
    

  def generate_samples_reverse(self, score_network: torch.nn.Module, nsamples: int) -> torch.Tensor:
      x_t = torch.randn((nsamples, 2))
      time_pts = torch.linspace(1, 0, 1000)
      beta = lambda t: beta(t)
      for i in range(len(time_pts) - 1):
          t = time_pts[i]
          dt = time_pts[i + 1] - t
          score = score_network(x_t,t.expand(x_t.shape[0], 1)).detach()
          tot_drift = self.f(x_t,t) - self.g(t)**2 * score
          tot_diffusion = self.g(t)

          # euler-maruyama step
          x_t = x_t + tot_drift * dt + tot_diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5
      return x_t
