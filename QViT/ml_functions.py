from torch.utils.data import Dataset
from tqdm.notebook import tqdm
import numpy as np
import torch

class simple_dataset(Dataset):
    """
    A simple dataset class to handle data and target labels.

    Args:
        data (Tensor): The input data.
        target (Tensor): Corresponding target labels.
        transform (callable, optional): Optional transform to be applied on a sample.

    Attributes:
        data (Tensor): The input data.
        target (Tensor): Target labels for the data.
        len (int): Number of samples in the dataset.
    """
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.len = self.data.shape[0]

    def __len__(self):
        """Returns the length of the dataset."""
        return self.len

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset at a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary with 'input' and 'output' keys containing data and target.
        """
        sample = {'input': self.data[idx], 'output': self.target[idx]}
        return sample


def patcher(data, sh):
    """
    Splits data into patches of a specified shape and returns them as a single tensor.

    Args:
        data (Tensor): Input tensor to split.
        sh (tuple): Shape of each patch (rows, columns).

    Returns:
        Tensor: Tensor containing the flattened patches.
    """
    r, c = sh
    # Calculate the number of patches along each dimension
    rmax = (data.shape[-2] // r)
    cmax = (data.shape[-1] // c)

    # Create an empty tensor to hold the patches
    patched = torch.empty(*data.shape[:-2], rmax * cmax, r * c, device=data.device)
    # Populate the tensor with patches
    for i in range(rmax):
        for j in range(cmax):
            patched[..., (i * cmax) + j, :] = data[..., (i * r):(i * r + r), (j * c):(j * c + c)].flatten(start_dim=-2)
    return patched


def patcher_with_color(data, sh):
    """
    Splits multi-channel (e.g., color) data into patches and returns as a tensor.

    Args:
        data (Tensor): Input tensor to split, expected to have a channel dimension.
        sh (tuple): Shape of each patch (rows, columns).

    Returns:
        Tensor: Tensor containing the flattened patches.
    """
    r, c = sh
    # Calculate the number of patches along each dimension
    rmax = (data.shape[-3] // r)
    cmax = (data.shape[-2] // c)

    # Create an empty tensor to hold the patches
    patched = torch.empty(*data.shape[:-3], rmax * cmax, r * c * 2, device=data.device)
    # Populate the tensor with patches
    for i in range(rmax):
        for j in range(cmax):
            patched[..., (i * cmax) + j, :] = data[..., (i * r):(i * r + r), (j * c):(j * c + c), :].flatten(start_dim=-3)
    return patched


def train(model, tr_dl, val_dl, loss_fn, optim, n_epochs, device='cuda'):
    """
    Trains a model on a given dataset and validates it, storing the training history.

    Args:
        model (torch.nn.Module): The model to train.
        tr_dl (DataLoader): DataLoader for training data.
        val_dl (DataLoader): DataLoader for validation data.
        loss_fn (callable): Loss function for training.
        optim (torch.optim.Optimizer): Optimizer for updating model parameters.
        n_epochs (int): Number of training epochs.
        device (str): Device to use for training (default is 'cuda').

    Returns:
        dict: Training history with losses and accuracies.
    """
    try:
        min_loss = np.inf
        # Progress bar for epochs
        bar_epoch = tqdm(range(n_epochs))
        # Dictionary to store training history
        history = {'tr': [], 'val': [], 'tr_acc': [], 'val_acc': []}

        # Training loop over epochs
        for epoch in bar_epoch:
            loss = 0
            val_loss = 0
            total_samples = 0
            
            bar_batch = tqdm(tr_dl)  # Progress bar for batches
            model.train()
            pred_tr, real_tr = [], []
            pred_val, real_val = [], []

            # Training loop over batches
            for i in bar_batch:
                optim.zero_grad()  # Zero the gradients
                yhat = model(i['input'].to(device))  # Model predictions
                y = i['output']  # True labels
                loss_ = loss_fn(yhat, y.to(device))  # Calculate loss
                loss_.sum().backward()  # Backpropagation
                optim.step()  # Update model parameters

                loss += loss_.sum().item()  # Accumulate batch loss
                total_samples += y.shape[0]  # Count samples processed

                # Record predictions for accuracy calculation
                if len(yhat.shape) == 1 or yhat.shape[-1] == 1:  # Binary classification
                    pred_tr.append((torch.sigmoid(yhat.detach()) > 0.5).cpu())
                    real_tr.append(y.detach().cpu().unsqueeze(-1))
                else:  # Multiclass classification
                    pred_tr.append(yhat.detach().argmax(axis=-1).cpu())
                    real_tr.append(y.detach().cpu())

                # Update progress bar for batch with current loss
                bar_batch.set_postfix_str(f'loss: {loss / total_samples}')

            # Validation loop over validation batches
            model.eval()
            for i in val_dl:
                with torch.no_grad():  # Disable gradients for validation
                    yhat = model(i['input'].to(device))
                    y = i['output']
                    val_loss_ = loss_fn(yhat, y.to(device))  # Calculate validation loss
                    val_loss += val_loss_.sum().item()  # Accumulate validation loss

                    # Record predictions for accuracy calculation
                    if len(yhat.shape) == 1 or yhat.shape[-1] == 1:
                        pred_val.append((torch.sigmoid(yhat.detach()) > 0.5).cpu())
                        real_val.append(y.detach().cpu().unsqueeze(-1))
                    else:
                        pred_val.append(yhat.detach().argmax(axis=-1).cpu())
                        real_val.append(y.detach().cpu())

            # Compute and store accuracy for training and validation sets
            history['tr_acc'].append((torch.cat(pred_tr) == torch.cat(real_tr)).sum() / total_samples)
            history['val_acc'].append((torch.cat(pred_val) == torch.cat(real_val)).sum() / len(val_dl.dataset))
            # Compute and store average loss for training and validation sets
            history['val'].append(val_loss / len(val_dl.dataset))
            history['tr'].append(loss / total_samples)
            
            # Update progress bar for epoch with current losses and accuracies
            bar_epoch.set_postfix_str(f'loss: {loss / total_samples}, v.loss: {val_loss / len(val_dl.dataset)}, \
            tr_acc: {history["tr_acc"][-1]}, val_acc: {history["val_acc"][-1]}')

            # Save model state if validation loss or accuracy improves
            if history['val'][-1] < min_loss:
                min_loss = history['val'][-1]
                torch.save(model.state_dict(), 'best_state_on_training_loss')
            if history['val_acc'][-1] == max(history['val_acc']):
                min_loss = history['val'][-1]
                torch.save(model.state_dict(), 'best_state_on_training_acc')

            # Save the training history after each epoch
            torch.save(history, 'temp_history')

        return history  # Return complete training history
    except KeyboardInterrupt:
        # Handle early termination and return current training history
        return history
