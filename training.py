import torch
import time


def train(sde, score_model, number_of_steps, data):
  # optimizer = torch.optim.SGD(score_model.parameters(),lr=0.01)
  optimizer = torch.optim.Adam(score_model.parameters(), lr=3e-4)
  errors = []
  t0 = time.time()
  for i in range(number_of_steps):
    optimizer.zero_grad()
    loss = loss_function(sde,data, score_model)
    loss.backward()
    optimizer.step()

    if(i%10000 == 0):
      print(f"Step number {i} ({time.time() - t0}s) \nError : {loss}")
      errors.append(loss)
      torch.save(score_model.state_dict(), 'coefficients.pth')
  return errors


def loss_function(sde, data,score_function, eps = .0001):
  random_t = torch.rand((data.shape[0],1), device=data.device) * (sde.T - eps) + eps  
  z = torch.randn_like(data)
  mean = sde.marginal_prob_mean(data,random_t)
  std = sde.marginal_prob_var(random_t)**.5
  perturbed_data = mean+z*std

  loss = torch.mean(std*(std*score_function(perturbed_data,random_t)+z)**2)
  return loss