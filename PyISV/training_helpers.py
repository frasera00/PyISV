import torch

def clip_gradients(model):
    if getattr(model, 'grad_clip', None) is not None:
        if getattr(model, 'amp', False):
            model.scaler.unscale_(model.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip)

def train_classifier_batch(model, x, y):
    outputs = model.forward(x)
    loss = model.criterion(outputs, y)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y).sum().item()
    total = y.size(0)
    return loss, correct, total

def train_autoencoder_batch(model, x, y):
    reconstructed, _ = model.forward(x)
    loss = model.criterion(reconstructed, y if model.use_separate_target else x)
    return loss, None, None

def scheduler_step(model, val_loss):
    if model.scheduler is not None:
        if hasattr(model.scheduler, 'step') and 'ReduceLROnPlateau' in model.scheduler.__class__.__name__:
            if val_loss is not None:
                model.scheduler.step(val_loss)
        else:
            model.scheduler.step()

def log_epoch(model, epoch, total_loss, correct, total):
    if model.model_type == "classifier":
        accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1}/{model.max_epochs}, Training Loss: {total_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
    else:
        print(f"Epoch {epoch + 1}/{model.max_epochs}, Training Loss: {total_loss:.4f}")

def early_stop_and_checkpoint(model, val_loss, epoch):
    stop = False
    if model.early_stopping is not None and val_loss is not None:
        if model.early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            stop = True
    if model.checkpoint_path is not None and val_loss is not None:
        torch.save(model.state_dict(), model.checkpoint_path)
    return stop
