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

def train_loop(model, optimizer, device, train_loader, test_loader,
               n_epochs, patience=20, min_delta=1e-5):

    best_train_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = run_epoch(model, optimizer, device, train_loader, True)
        val_loss = run_epoch(model, optimizer, device, test_loader, False)

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if best_train_loss - train_loss > min_delta:
            best_train_loss = train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs with no loss decrease.")
            break

    return model, val_loss

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