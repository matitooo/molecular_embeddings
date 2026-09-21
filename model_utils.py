from utils import masked_mse
import torch
import os

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
               n_epochs, patience=20, min_delta=1e-5,base_path = ""):

    best_train_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = run_epoch(model, optimizer, device, train_loader, True)
        if epoch%5==0:
            val_loss = eval(model,test_loader,base_path,str(epoch))
            print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if best_train_loss - train_loss > min_delta:
            best_train_loss = train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs with no loss decrease.")
            break

    #compute final scores and predictions 
    val_loss = eval(model,test_loader,base_path=base_path)

    return model, val_loss


def eval(model, test_loader, base_path, epoch=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    total_loss, total_n = 0.0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred, mask = model(batch)
            target = batch.y.view(pred.shape)
            loss = masked_mse(pred, target, mask)
            n = mask.sum().item()
            total_loss += loss.item() * n
            total_n += n
            all_preds.append(pred.detach().cpu())
            all_targets.append(target.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    if not epoch:
        epoch = 'final'
    save_path = os.path.join(base_path, str(epoch))
    os.makedirs(save_path, exist_ok=True)
    torch.save({'predictions': all_preds, 'targets': all_targets},
               os.path.join(save_path, 'predictions_targets.pt'))

    return total_loss / total_n