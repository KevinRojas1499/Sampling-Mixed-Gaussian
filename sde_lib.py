"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import numpy as np
import torch

class SDE(abc.ABC):
  #In reality im doing this for a specific SDE
  def __init__(self, N,T, f, g):
    super().__init__()
    self.N = N
    self.T = T
    self.f = f
    self.g = g
  
  def discretize(self,x0):
    xn = x0
    deltaT = self.T/self.N
    for k in range(self.N):
      t = k*deltaT
      xn = xn+self.f(xn,t) * deltaT + self.g(t) * np.random.randn(2)*deltaT**.5
    return xn

  def marginal_prob_mean(self, x0, t):
    alpha = 3
    var = torch.exp(-alpha*t)
    return torch.mul(var,x0)

  def marginal_prob_var(self, t):
    alpha = 3
    return 1-torch.exp(-2*alpha*t)
    

class ReverseSDE():
  def __init__(self, sde,score):
    self.N = sde.N
    self.T = sde.T
    self.f = lambda x,t : sde.f(x,t) - sde.g(t) * sde.g(t) * score(x, self.T - t)
    self.g = sde.g
  
  def discretize(self,x0):
    xn = torch.tensor(x0)
    deltaT = self.T/self.N
    for k in range(self.N):
      t = torch.Tensor([k*deltaT])
      xn = xn+self.f(xn,t) * deltaT + self.g(t) * torch.randn(2)*deltaT**.5
    return xn