import numpy as np
import torch
import random
import os

def save_prompt_weights(model, file_path):
    torch.save(model.prompt.data, file_path)


def load_prompt_weights(model, file_path):
    model.prompt.data = torch.load(file_path)


def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mae_loss(preds, labels, null_val)
    

def seed_everything(seed):
    '''
    https://github.com/qhd1996/seed-everything/blob/master/seed-everything.py
    '''
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_variance(weight_snapshots):
    # Calculating variance across epochs for each weight matrix element-wise
    weight_snapshots = np.stack(weight_snapshots, axis=0)
    variance = np.var(weight_snapshots, axis=0)
    return variance
