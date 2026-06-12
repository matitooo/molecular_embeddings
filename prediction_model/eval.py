import torch 
from utils import masked_mse

def eval(model,test_loader):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model.eval()
  total_loss, total_n = 0.0, 0
  for batch in test_loader:
            batch = batch.to(device)
            pred, mask = model(batch)           
            target = batch.y.view(pred.shape)   
            loss = masked_mse(pred, target, mask)
            n = mask.sum().item()
            total_loss += loss.item() * n
            total_n    += n
  return total_loss / total_n