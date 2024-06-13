import numpy as np
import torch
import copy

from utils.utils import *
from utils.metrics import RMSE_MAE_MAPE
from utils.dataloader import get_dataloaders
from utils.logging import get_logger
from utils.args import create_parser
from model.models import CMuST

def train_epoch(model, device, dataloader, scaler, optimizer, scheduler, criterion):

    model.train()
    losses = []
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs)
        predictions = scaler.inverse_transform(predictions)

        loss = criterion(predictions, targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(losses)
    scheduler.step()

    return avg_loss


def train(model, device, trainset_loader, valset_loader, scaler, optimizer, scheduler, criterion, 
          max_epochs, patience, log_interval=1, logger=None, save_path=None):
    
    model = model.to(device)
    
    patience_counter = 0
    min_val_loss = np.inf

    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, device, trainset_loader, scaler, optimizer, scheduler, criterion)
        train_losses.append(train_loss)

        val_loss = evaluate(model, device, valset_loader, scaler, criterion)
        val_losses.append(val_loss)

        if (epoch + 1) % log_interval == 0:
            message = "Epoch: {}\tTrain Loss: {:.4f} Val Loss: {:.4f} LR: {:.4e}"
            logger.info(message.format(epoch + 1, train_loss, val_loss, scheduler.get_last_lr()[0]))

        if val_loss < min_val_loss:
            patience_counter = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, device, trainset_loader, scaler))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, device, valset_loader, scaler))

    logger.info(f"Early stopping at epoch: {epoch+1} Best at epoch {best_epoch+1}")
    train_log = "Train Loss: {:.5f} MAE: {:.5f} RMSE: {:.5f} MAPE: {:.5f}"
    logger.info(train_log.format(train_losses[best_epoch], train_mae, train_rmse, train_mape))
    
    val_log = "Val Loss: {:.5f} MAE: {:.5f} RMSE: {:.5f} MAPE: {:.5f}"
    logger.info(val_log.format(val_losses[best_epoch], val_mae, val_rmse, val_mape))

    torch.save(best_state_dict, save_path)
    return model


def train_record_w(model, device, trainset_loader, valset_loader, scaler, optimizer, scheduler, criterion, 
          max_epochs, patience, log_interval=1, logger=None, save_path=None):
    
    model = model.to(device)
    
    patience_counter = 0
    min_val_loss = np.inf

    train_losses = []
    val_losses = []
    w_list = []

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, device, trainset_loader, scaler, optimizer, scheduler, criterion)
        train_losses.append(train_loss)
        
        parameters_copy = {name: param.clone() for name, param in model.named_parameters()}
        w_list.append(parameters_copy)

        val_loss = evaluate(model, device, valset_loader, scaler, criterion)
        val_losses.append(val_loss)

        if (epoch + 1) % log_interval == 0:
            message = "Epoch: {}\tTrain Loss: {:.4f} Val Loss: {:.4f} LR: {:.4e}"
            logger.info(message.format(epoch + 1, train_loss, val_loss, scheduler.get_last_lr()[0]))

        if val_loss < min_val_loss:
            patience_counter = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, device, trainset_loader, scaler))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, device, valset_loader, scaler))

    logger.info(f"Early stopping at epoch: {epoch+1} Best at epoch {best_epoch+1}")
    train_log = "Train Loss: {:.5f} MAE: {:.5f} RMSE: {:.5f} MAPE: {:.5f}"
    logger.info(train_log.format(train_losses[best_epoch], train_mae, train_rmse, train_mape))
    
    val_log = "Val Loss: {:.5f} MAE: {:.5f} RMSE: {:.5f} MAPE: {:.5f}"
    logger.info(val_log.format(val_losses[best_epoch], val_mae, val_rmse, val_mape))

    torch.save(best_state_dict, save_path)
    return model, w_list


@torch.no_grad()
def evaluate(model, device, dataloader, scaler, criterion):
    
    model = model.to(device)
    model.eval()
    losses = []
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        outputs = scaler.inverse_transform(outputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())

    avg_loss = np.mean(losses)
    return avg_loss


@torch.no_grad()
def predict(model, device, dataloader, scaler):
    
    model = model.to(device)
    model.eval()
    true_labels = []
    predictions = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        preds = model(inputs)
        preds = scaler.inverse_transform(preds)

        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        predictions.append(preds)
        true_labels.append(labels)

    predictions = np.vstack(predictions).squeeze()
    true_labels = np.vstack(true_labels).squeeze()

    return true_labels, predictions


@torch.no_grad()
def test(model, device, testset_loader, scaler, logger):
    
    model = model.to(device)
    model.eval()
    true_labels, predictions = predict(model, device, testset_loader, scaler)
    
    test_maes = []
    test_mapes = []
    test_rmses = []
    
    out_steps = predictions.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(true_labels[:, i, :], predictions[:, i, :])
        log = 'Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
        logger.info(log.format(i + 1, mae, rmse, mape))
        test_maes.append(mae)
        test_rmses.append(rmse)
        test_mapes.append(mape)

    log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
    logger.info(log.format(np.mean(test_maes), np.mean(test_rmses), np.mean(test_mapes)))
    