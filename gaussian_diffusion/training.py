import torch
import time
from torch.utils.data import DataLoader


def train(sde, score_model, number_of_steps, data, file_to_save, device):
  # optimizer = torch.optim.SGD(score_model.parameters(),lr=0.01)
  optimizer = torch.optim.Adam(score_model.parameters(), lr=3e-4)
  dataloader = DataLoader(data, batch_size=64, shuffle=True)
  errors = []
  t0 = time.time()
  epochs = 100
  step = 0
  for _ in range(epochs):
    for data in iter(dataloader):
      if step > number_of_steps:
        break
      optimizer.zero_grad()
      loss = loss_function(sde,data, score_model, device)
      loss.backward()
      optimizer.step()

      if(step%10000 == 0):
        print(f"Step number {step} ({time.time() - t0}s) \nError : {loss}")
        errors.append(loss)
        torch.save(score_model.state_dict(), file_to_save) if file_to_save != "" else None
      step += 1
  return errors


def loss_function(sde, data,score_function, device, eps = .0001):
  random_t = torch.rand((data.shape[0],1), device=data.device) * (sde.T - eps) + eps  
  shaped_random_t = torch.repeat_interleave(random_t, data.shape[-1], dim=1)[:, None, :]
  z = torch.randn_like(data).to(device)
  mean = sde.marginal_prob_mean(data,shaped_random_t)
  std = sde.marginal_prob_var(shaped_random_t)**.5
  perturbed_data = mean+z*std

  loss = torch.mean(std*(std*score_function(perturbed_data,random_t)+z)**2)
  return loss