from utils import masked_mse
import torch
import os
import logging
import time

logger = logging.getLogger('pipeline.debug.training')

def run_epoch(model, optimizer, device, loader, train=True, debug=False, epoch=None):
    model.train() if train else model.eval()
    total_loss, total_n = 0.0, 0

    started = time.monotonic()
    if debug:
        logger.info('Starting %s epoch=%s batches=%d device=%s', 'training' if train else 'evaluation', epoch, len(loader), device)
    with torch.set_grad_enabled(train):
        for batch_index, batch in enumerate(loader, start=1):
            batch_started = time.monotonic()
            if debug:
                logger.debug('Batch start phase=%s epoch=%s batch=%d/%d', 'train' if train else 'eval', epoch, batch_index, len(loader))
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
            if debug:
                logger.debug('Batch done phase=%s epoch=%s batch=%d/%d loss=%.6f valid_targets=%s elapsed=%.3fs', 'train' if train else 'eval', epoch, batch_index, len(loader), loss.item(), n, time.monotonic() - batch_started)

    mean_loss = total_loss / total_n
    if debug:
        logger.info('Finished %s epoch=%s loss=%.6f valid_targets=%d elapsed=%.2fs', 'training' if train else 'evaluation', epoch, mean_loss, total_n, time.monotonic() - started)
    return mean_loss

def train_loop(model, optimizer, device, train_loader, test_loader,
               n_epochs, patience=4, min_delta=1e-5, base_path='', debug=False):
    best_val_loss = float('inf')
    best_model_state = None
    validation_checks_without_improvement = 0

    logger.info('Train loop start: epochs=%d train_batches=%d validation_batches=%d base_path=%s debug=%s', n_epochs, len(train_loader), len(test_loader), os.path.abspath(base_path or '.'), debug)
    for epoch in range(1, n_epochs + 1):
        if debug:
            logger.info('Epoch %d/%d started', epoch, n_epochs)
        train_loss = run_epoch(
            model,
            optimizer,
            device,
            train_loader,
            True,
            debug=debug,
            epoch=epoch
        )

        if epoch % 5 == 0:
            val_loss = eval(
                model,
                test_loader,
                base_path,
                str(epoch),
                debug=debug
            )

            print(
                f'Epoch {epoch:03d} | '
                f'train {train_loss:.4f} | '
                f'val {val_loss:.4f}'
            )

            if best_val_loss - val_loss > min_delta:
                best_val_loss = val_loss

                best_model_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }

                validation_checks_without_improvement = 0
            else:
                validation_checks_without_improvement += 1

            if validation_checks_without_improvement >= patience:
                print(
                    f'Early stopping after {patience} validation checks '
                    f'with no loss decrease.'
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    val_loss = eval(
        model,
        test_loader,
        base_path=base_path,
        debug=debug
    )

    return model, val_loss

def eval(model, test_loader, base_path, epoch=None, debug=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()

    total_loss, total_n = 0.0, 0

    all_preds = []
    all_targets = []
    all_masks = []

    eval_started = time.monotonic()
    if debug:
        logger.info('Eval start epoch=%s batches=%d device=%s', epoch, len(test_loader), device)
    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader, start=1):
            batch_started = time.monotonic()
            if debug:
                logger.debug('Eval batch start %d/%d', batch_index, len(test_loader))
            batch = batch.to(device)

            pred, mask = model(batch)
            target = batch.y.view(pred.shape)

            loss = masked_mse(
                pred,
                target,
                mask
            )

            n = mask.sum().item()

            total_loss += loss.item() * n
            total_n += n

            all_preds.append(pred.detach().cpu())
            all_targets.append(target.detach().cpu())
            all_masks.append(mask.detach().cpu())
            if debug:
                logger.debug('Eval batch done %d/%d loss=%.6f valid_targets=%s elapsed=%.3fs', batch_index, len(test_loader), loss.item(), n, time.monotonic() - batch_started)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    if not epoch:
        epoch = 'final'

    save_path = os.path.join(
        base_path,
        str(epoch)
    )

    os.makedirs(
        save_path,
        exist_ok=True
    )

    if debug:
        logger.info('Saving predictions/targets/mask to %s', os.path.abspath(os.path.join(save_path, 'predictions_targets.pt')))
    torch.save(
        {
            'predictions': all_preds,
            'targets': all_targets,
            'mask': all_masks
        },
        os.path.join(
            save_path,
            'predictions_targets.pt'
        )
    )

    result = total_loss / total_n
    if debug:
        logger.info('Eval complete epoch=%s loss=%.6f elapsed=%.2fs', epoch, result, time.monotonic() - eval_started)
    return result