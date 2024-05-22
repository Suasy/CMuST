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
from engine import *
from model.models import CMuST

def get_config():
    parser = create_parser()
    args = parser.parse_args()
    
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    log_dir = 'logs/{}/'.format(args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(log_dir, __name__, '{}.log'.format(now))
    logger.info(args)
    
    return args, logger, now

if __name__ == "__main__":
    
    # configuration parameters, logger, and current time
    args, logger, now = get_config()
    
    # random seed and CPU thread number
    seed = torch.randint(1000, (1,))
    seed_everything(seed)
    set_cpu_num(1)

    # GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # directory and model name
    data_dir = f"data/{args.dataset}"
    model_name = CMuST.__name__
    
    # load model
    model = CMuST(num_nodes=args.num_nodes,
                    in_steps=args.in_steps,
                    out_steps=args.out_steps,
                    steps_per_day=args.steps_per_day,
                    obser_dim=args.obser_dim,
                    output_dim=args.output_dim,
                    obser_embedding_dim=args.obser_embedding_dim,
                    tod_embedding_dim=args.tod_embedding_dim,
                    dow_embedding_dim=args.dow_embedding_dim,
                    timestamp_embedding_dim=args.timestamp_embedding_dim,
                    spatial_embedding_dim=args.spatial_embedding_dim,
                    temporal_embedding_dim=args.temporal_embedding_dim,
                    prompt_dim=args.prompt_dim,
                    self_atten_dim=args.self_atten_dim,
                    cross_atten_dim=args.cross_atten_dim,
                    feed_forward_dim=args.feed_forward_dim,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    )
    
    # load pre-trained model weights
    # model.load_state_dict(torch.load('2024-04-18-10-18-28.pt'))
    
    # load dataset
    
    tasks = os.listdir(data_dir)
    task_dirs = [os.path.join(data_dir, item) for item in tasks]
    logger.info(f"Load data {task_dirs}")
    trainset_loaders=[]
    valset_loaders=[]
    testset_loaders=[]
    scalers=[]
    for task_dir in task_dirs:
        trainset_loader, valset_loader, testset_loader, scaler = get_dataloaders(
            task_dir,
            batch_size=args.batch_size,
            logger=logger,
        )
        trainset_loaders.append(trainset_loader)
        valset_loaders.append(valset_loader)
        testset_loaders.append(testset_loader)
        scalers.append(scaler)

    # loss function, optimizer, and scheduler
    # criterion = MaskedMAELoss()
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.lr_decay_rate,
        verbose=False,
    )

    # model structure information
    logger.info('The number of parameters: {}'.format(sum([param.nelement() for param in model.parameters()])))
    
    # RoAda
    threshold = args.threshold
    
    # train task 0
    logger.info('Train for task 0')
    weight_histories = {name: [] for name, param in model.named_parameters()}
    save_dir = 'checkpoints/{}/{}/'.format(args.dataset,tasks[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{now}_first_round.pt")
    model = train(
        model,
        device,
        trainset_loaders[0],
        valset_loaders[0],
        scalers[0],
        optimizer,
        scheduler,
        criterion,
        clip_grad=0,
        max_epochs=args.max_epochs,
        patience=args.patience,
        verbose=1,
        logger=logger,
        save_path=save_path,
    )
    # model.load_state_dict(torch.load('checkpoints/' + 'NYC/' + 'TAXIPICK/' + '2024-05-21-10-11-18_first_round.pt'))
    # model.load_state_dict(torch.load('checkpoints/CHI/TAXIPICK/2024-05-21-12-12-21_first_round.pt'))
    # model.load_state_dict(torch.load('checkpoints/SIP/FLOW/2024-05-21-12-12-24_first_round.pt'))
    # test model
    test_model(model, device, testset_loaders[0], scalers[0], logger=logger)
    for name, param in model.named_parameters():
        weight_histories[name].append(param.data.cpu().numpy())
    
    # train to task k-1
    for i, task in enumerate(tasks):
        if i == 0:
            continue
        # train task 1 to k-1
        logger.info(f'Train for task {i}')
        save_dir = 'checkpoints/{}/{}/'.format(args.dataset,tasks[i])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{now}_first_round.pt")
        model, w_list = train_record_w(
            model,
            device,
            trainset_loaders[i],
            valset_loaders[i],
            scalers[i],
            optimizer,
            scheduler,
            criterion,
            clip_grad=0,
            max_epochs=args.max_epochs,
            patience=args.patience,
            verbose=1,
            logger=logger,
            save_path=save_path,
        )
        for w in w_list:
            for name, param in w.items():
                weight_histories[name].append(param.data.cpu().numpy())

        for name, param in model.named_parameters():
            if param.requires_grad == True:
                variances = calculate_variance(weight_histories[name])
                if np.all(variances < threshold):
                    param.requires_grad = False
        
        # frozen params
        logger.info('Total/Frozen Parameters: {}/{}'.format(sum([param.nelement() for param in model.parameters()]), 
                                                            sum([param.nelement() for param in model.parameters()]) - 
                                                            sum([param.nelement() for param in filter(lambda p : p.requires_grad, model.parameters())])))
        
        # test model
        test_model(model, device, testset_loaders[i], scalers[i], logger=logger)
        weight_histories = {name: [] for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            weight_histories[name].append(param.data.cpu().numpy())



    # train task 0
    logger.info(f'Train for task 0')
    save_dir = 'checkpoints/{}/{}/'.format(args.dataset,tasks[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{now}_second_round.pt")
    model, w_list = train_record_w(
        model,
        device,
        trainset_loaders[0],
        valset_loaders[0],
        scalers[0],
        optimizer,
        scheduler,
        criterion,
        clip_grad=0,
        max_epochs=args.max_epochs,
        patience=args.patience,
        verbose=1,
        logger=logger,
        save_path=save_path,
    )
    for w in w_list:
        for name, param in w.items():
            weight_histories[name].append(copy.deepcopy(param.data).cpu().numpy())

    for name, param in model.named_parameters():
        if param.requires_grad == True:
            variances = calculate_variance(weight_histories[name])
            if np.all(variances < threshold):
                param.requires_grad = False
    
    # frozen params
    logger.info('Total/Frozen Parameters: {}/{}'.format(sum([param.nelement() for param in model.parameters()]), 
                                                        sum([param.nelement() for param in model.parameters()]) - 
                                                        sum([param.nelement() for param in filter(lambda p : p.requires_grad, model.parameters())])))
    
    # test model
    test_model(model, device, testset_loaders[0], scalers[0], logger=logger)
    
    # saved model path
    # logger.info(f"Saved Model: {save_path}")

    
    # fine-tuning
    logger.info(f'Fine Tuning')
    for i, task in enumerate(tasks):
        logger.info(f'Train for task {i}')
        save_dir = 'checkpoints/{}/{}/'.format(args.dataset,tasks[i])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{now}_fine_tuning.pt")
        model = train(
            model,
            device,
            trainset_loaders[i],
            valset_loaders[i],
            scalers[i],
            optimizer,
            scheduler,
            criterion,
            clip_grad=0,
            max_epochs=args.max_epochs,
            patience=args.patience,
            verbose=1,
            logger=logger,
            save_path=save_path,
        )
        
        # test model
        test_model(model, device, testset_loaders[i], scalers[i], logger=logger)
       
    # test model
    # test_model(model, device, testset_loader, scaler, logger=logger)
