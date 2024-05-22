import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys
import copy

from utils.utils import *
from utils.metrics import RMSE_MAE_MAPE
from utils.dataloader import get_dataloaders
from utils.logging import get_logger
from utils.args import create_parser
from model.models import CMuST

@torch.no_grad()
def eval_model(model, device, valset_loader, scaler, criterion):
    
    model = model.to(device)
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        out_batch = model(x_batch)
        out_batch = scaler.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, device, loader, scaler):
    
    model = model.to(device)
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        out_batch = model(x_batch)
        out_batch = scaler.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out


def train_one_epoch(model, device, trainset_loader, scaler, optimizer, scheduler, criterion, clip_grad):
    
    model.train()
    batch_loss_list = []
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        out_batch = model(x_batch)
        out_batch = scaler.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss

def train(model, device, trainset_loader, valset_loader, scaler, optimizer, scheduler, criterion, clip_grad, 
          max_epochs, patience, verbose=1, plot=False, logger=None, save_path=None):
    
    model = model.to(device)
    
    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, device, trainset_loader, scaler, optimizer, scheduler, criterion, clip_grad
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, device, valset_loader, scaler, criterion)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            message = "Epoch: {}\tTrain Loss: {:.4f} Val Loss: {:.4f} LR: {:.4e}"
            logger.info(message.format(epoch + 1, train_loss, val_loss, scheduler.get_last_lr()[0]))

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, device, trainset_loader, scaler))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, device, valset_loader, scaler))

    logger.info(f"Early stopping at epoch: {epoch+1} Best at epoch {best_epoch+1}")
    train_log = "Train Loss: {:.5f} MAE: {:.5f} RMSE: {:.5f} MAPE: {:.5f}"
    logger.info(train_log.format(train_loss_list[best_epoch], train_mae, train_rmse, train_mape))
    
    val_log = "Val Loss: {:.5f} MAE: {:.5f} RMSE: {:.5f} MAPE: {:.5f}"
    logger.info(val_log.format(val_loss_list[best_epoch], val_mae, val_rmse, val_mape))

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    torch.save(best_state_dict, save_path)
    return model

def train_record_w(model, device, trainset_loader, valset_loader, scaler, optimizer, scheduler, criterion, clip_grad, 
          max_epochs, patience, verbose=1, plot=False, logger=None, save_path=None):
    
    model = model.to(device)
    
    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []
    w_list = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, device, trainset_loader, scaler, optimizer, scheduler, criterion, clip_grad
        )
        train_loss_list.append(train_loss)
        
        parameters_copy = {name: param.clone() for name, param in model.named_parameters()}
        w_list.append(parameters_copy)

        val_loss = eval_model(model, device, valset_loader, scaler, criterion)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            message = "Epoch: {}\tTrain Loss: {:.4f} Val Loss: {:.4f} LR: {:.4e}"
            logger.info(message.format(epoch + 1, train_loss, val_loss, scheduler.get_last_lr()[0]))

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, device, trainset_loader, scaler))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, device, valset_loader, scaler))

    logger.info(f"Early stopping at epoch: {epoch+1} Best at epoch {best_epoch+1}")
    train_log = "Train Loss: {:.5f} MAE: {:.5f} RMSE: {:.5f} MAPE: {:.5f}"
    logger.info(train_log.format(train_loss_list[best_epoch], train_mae, train_rmse, train_mape))
    
    val_log = "Val Loss: {:.5f} MAE: {:.5f} RMSE: {:.5f} MAPE: {:.5f}"
    logger.info(val_log.format(val_loss_list[best_epoch], val_mae, val_rmse, val_mape))

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    torch.save(best_state_dict, save_path)
    return model, w_list

@torch.no_grad()
def test_model(model, device, testset_loader, scaler, logger):
    
    model = model.to(device)
    model.eval()
    y_true, y_pred = predict(model, device, testset_loader, scaler)
    
    test_mae = []
    test_mape = []
    test_rmse = []
    
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
        logger.info(log.format(i + 1, mae, rmse, mape))
        test_mae.append(mae)
        test_rmse.append(rmse)
        test_mape.append(mape)

    log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
    logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))