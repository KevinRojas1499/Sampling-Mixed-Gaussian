import abc
import torch

class SDE(abc.ABC):
  #In reality im doing this for a specific SDE
  def __init__(self, N, T, beta):
    # Right now I just assume thhe function is beta*t
    super().__init__()
    self.N = N
    self.T = T
    self.beta = beta

  def marginal_prob_mean(self, x0, t):
    alpha = self.beta
    var = torch.exp(-alpha*t*t/4).squeeze()
    return var.view(x0.shape[0], 1, 1, 1) * x0 #torch.mul(var, x0)

  def marginal_prob_var(self, t):
    alpha = 20
    var = torch.exp(-alpha*t*t/4)
    return 1-var


class ReverseSDE():
  def __init__(self, sde,score):
    self.N = sde.N
    self.T = sde.T
    self.f = lambda x,t : sde.f(x,t) - sde.g(t) * sde.g(t) * score(x, self.T - t)
    self.g = sde.g