import numpy as np
import os
import torch
import torch.nn as nn
import datetime
import copy

from utils.utils import *
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
    
    # is random seed
    if args.seed == 0:
        seed = torch.randint(1000, (1,))
    else:
        seed = torch.tensor([args.seed])
        
    seed_everything(seed)

    # GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # directory and model name
    data_dir = f"data/{args.dataset}"
    model_name = CMuST.__name__
    
    # load model
    model = CMuST(num_nodes=args.num_nodes,
                    input_len=args.input_len,
                    output_len=args.output_len,
                    tod_size=args.tod_size,
                    obser_dim=args.obser_dim,
                    output_dim=args.output_dim,
                    obser_embed_dim=args.obser_embed_dim,
                    tod_embed_dim=args.tod_embed_dim,
                    dow_embed_dim=args.dow_embed_dim,
                    timestamp_embed_dim=args.timestamp_embed_dim,
                    spatial_embed_dim=args.spatial_embed_dim,
                    temporal_embed_dim=args.temporal_embed_dim,
                    prompt_dim=args.prompt_dim,
                    self_atten_dim=args.self_atten_dim,
                    cross_atten_dim=args.cross_atten_dim,
                    feed_forward_dim=args.feed_forward_dim,
                    num_heads=args.num_heads,
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
    criterion = MaskedMAELoss()
    # criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.steps,
        gamma=args.gamma
    )

    # model structure information
    logger.info('The number of parameters: {}'.format(sum([param.nelement() for param in model.parameters()])))
    
    # RoAda
    threshold = args.threshold
    
    # train task 1
    logger.info('Train for task 1')
    save_dir = 'checkpoints/{}/{}/'.format(args.dataset,tasks[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{now}_start.pt")
    # load prompt
    nn.init.xavier_uniform_(model.prompt)
    # start train
    model = train(
        model,
        device,
        trainset_loaders[0],
        valset_loaders[0],
        scalers[0],
        optimizer,
        scheduler,
        criterion,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_interval=1,
        logger=logger,
        save_path=save_path,
    )
    # model.load_state_dict(torch.load('checkpoints/' + 'NYC/' + 'TAXIPICK/' + '2024-05-21-10-11-18_first_round.pt'))
    # model.load_state_dict(torch.load('checkpoints/CHI/TAXIPICK/2024-05-21-12-12-21_first_round.pt'))
    # model.load_state_dict(torch.load('checkpoints/SIP/FLOW/2024-05-21-12-12-24_first_round.pt'))
    # test model
    test(model, device, testset_loaders[0], scalers[0], logger=logger)
    # save prompt
    save_prompt_weights(model, os.path.join(save_dir, f"{now}_start_prompt.pth"))
    # update weight list
    weight_histories = {name: [] for name, param in model.named_parameters()}
    for name, param in model.named_parameters():
        weight_histories[name].append(copy.deepcopy(param.data).cpu().numpy())
    
    # train to task k
    for i, task in enumerate(tasks):
        if i == 0:
            continue
        # reset train
        optimizer = torch.optim.Adam(
        filter(lambda p : p.requires_grad, model.parameters()),
        lr=args.lr*0.01,
        weight_decay=args.weight_decay,
        eps=1e-8,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            # milestones=args.steps,
            milestones=[500],
            gamma=args.gamma
        )
        # train task 2 to k
        logger.info(f'Train for task {i+1}')
        save_dir = 'checkpoints/{}/{}/'.format(args.dataset,tasks[i])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{now}_first_round.pt")
        # load prompt
        nn.init.xavier_uniform_(model.prompt)
        # start train
        model, w_list = train_record_w(
            model,
            device,
            trainset_loaders[i],
            valset_loaders[i],
            scalers[i],
            optimizer,
            scheduler,
            criterion,
            max_epochs=args.max_epochs,
            patience=args.patience,
            log_interval=1,
            logger=logger,
            save_path=save_path,
        )
        # append weight
        for w in w_list:
            for name, param in w.items():
                weight_histories[name].append(copy.deepcopy(param.data).cpu().numpy())
        # cal var & frozen
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                variances = calculate_variance(weight_histories[name])
                if 'prompt' not in name and np.all(variances < threshold):
                    param.requires_grad = False
        # print frozen params
        logger.info('Total/Frozen Parameters: {}/{}'.format(sum([param.nelement() for param in model.parameters()]), 
                                                            sum([param.nelement() for param in model.parameters()]) - 
                                                            sum([param.nelement() for param in filter(lambda p : p.requires_grad, model.parameters())])))
        # test model
        test(model, device, testset_loaders[i], scalers[i], logger=logger)
        # save prompt
        save_prompt_weights(model, os.path.join(save_dir, f"{now}_first_round_prompt.pth"))
        # update weight list
        weight_histories = {name: [] for name, param in model.named_parameters()}
        for name, param in model.named_parameters():
            weight_histories[name].append(copy.deepcopy(param.data).cpu().numpy())

    # train task 1
    # reset train
    optimizer = torch.optim.Adam(
    filter(lambda p : p.requires_grad, model.parameters()),
    lr=args.lr*0.01,
    weight_decay=args.weight_decay,
    eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        # milestones=args.steps,
        milestones=[500],
        gamma=args.gamma
    )
    logger.info(f'Train for task 1')
    save_dir = 'checkpoints/{}/{}/'.format(args.dataset,tasks[0])
    save_path = os.path.join(save_dir, f"{now}_first_round.pt")
    # load prompt
    load_prompt_weights(model, os.path.join(save_dir, f"{now}_start_prompt.pth"))
    # start train
    model, w_list = train_record_w(
        model,
        device,
        trainset_loaders[0],
        valset_loaders[0],
        scalers[0],
        optimizer,
        scheduler,
        criterion,
        max_epochs=args.max_epochs,
        patience=args.patience,
        log_interval=1,
        logger=logger,
        save_path=save_path,
    )
    # append weight
    for w in w_list:
        for name, param in w.items():
            weight_histories[name].append(copy.deepcopy(param.data).cpu().numpy())
    # cal var & frozen
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            variances = calculate_variance(weight_histories[name])
            if 'prompt' not in name and np.all(variances < threshold):
                param.requires_grad = False
    # print frozen params
    logger.info('Total/Frozen Parameters: {}/{}'.format(sum([param.nelement() for param in model.parameters()]), 
                                                        sum([param.nelement() for param in model.parameters()]) - 
                                                        sum([param.nelement() for param in filter(lambda p : p.requires_grad, model.parameters())])))
    # test model
    test(model, device, testset_loaders[0], scalers[0], logger=logger)
    # save prompt
    save_prompt_weights(model, os.path.join(save_dir, f"{now}_first_round_prompt.pth"))
    
    # fine-tuning
    logger.info(f'Fine Tuning')
    weights_star_path = save_path
    
    for i, task in enumerate(tasks):
        # load init weights
        model.load_state_dict(torch.load(weights_star_path))
        # reset train
        optimizer = torch.optim.Adam(
        filter(lambda p : p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-8,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.steps,
            gamma=args.gamma
        )
        logger.info(f'Train for task {i+1}')
        save_dir = 'checkpoints/{}/{}/'.format(args.dataset,tasks[i])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{now}_fine_tuning.pt")
        # load prompt
        load_prompt_weights(model, os.path.join(save_dir, f"{now}_first_round_prompt.pth"))
        # start train
        model = train(
            model,
            device,
            trainset_loaders[i],
            valset_loaders[i],
            scalers[i],
            optimizer,
            scheduler,
            criterion,
            max_epochs=args.max_epochs,
            patience=args.patience,
            log_interval=1,
            logger=logger,
            save_path=save_path,
        )
        
        # test model
        test(model, device, testset_loaders[i], scalers[i], logger=logger)
        # save prompt
        save_prompt_weights(model, os.path.join(save_dir, f"{now}_fine_tuning_prompt.pth"))
        
    # test model
    # test(model, device, testset_loader, scaler, logger=logger)
