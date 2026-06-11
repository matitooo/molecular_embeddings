from utils import masked_mse
import torch

def run_epoch(model,optimizer,device,loader, train=True):
    model.train() if train else model.eval()
    total_loss, total_n = 0.0, 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device)

            pred, mask = model(batch)           
            target = batch.y.view(pred.shape)   

            loss = masked_mse(pred, target, mask)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n = mask.sum().item()
            total_loss += loss.item() * n
            total_n    += n

    return total_loss / total_n

def train_loop(model,optimizer,device,train_loader,test_loader,n_epochs):
    best_val_loss = float("inf")
    for epoch in range(1, n_epochs + 1):
      train_loss = run_epoch(model,optimizer,device,train_loader,train = True)
      val_loss   = run_epoch(model,optimizer,device,test_loader, train=False)

      print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

      if val_loss < best_val_loss:
          best_val_loss = val_loss
          torch.save(model.state_dict(), "best_model.pt")
          print(f"           ↳ saved (val {val_loss:.4f})")

    return None