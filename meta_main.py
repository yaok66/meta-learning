# -*- encoding: utf-8 -*-
'''
file       :meta_main.py
Date       :2025/05/28 16:17:28
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "20"
import numpy as np
import torch
from torch import nn
import random

from config import get_parser
from models import Base, MaxUpLLL
from trainers import  MetaTrainer, LLLTrainer, MaxUpLLLTrainer

from dataloaders import get_mutlisource_laoder, get_dataset, get_maxuppm_laoder

def setup_seed(seed):
    """
    Setup the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def weight_init(m):
    """
    Initialize model parameters.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.03)
        m.bias.data.zero_()

def create_dir_if_not_exists(path):
    """
    Create directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def get_model_utils(args):
    """
    Get model, optimizer, scheduler, and trainer.
    """
    # Model parameters
    base_params = {
        "input_dim": args.feature_dim,
        "num_of_class": args.num_of_class,
    }

    model = MaxUpLLL(**base_params).cuda()

    # Optimizer
    params = model.get_parameters()
    optimizer = torch.optim.RMSprop(params, lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    else:
        scheduler = None

    # Trainer parameters
    trainer_params = {
        "lr_scheduler": scheduler,
        "max_iter": args.n_epochs,
        "early_stop": args.early_stop,
        "log_interval": args.log_interval,
    }

    # trainer = MetaTrainer(
    #     model, 
    #     optimizer, 
    #     **trainer_params
    # )
    trainer = MaxUpLLLTrainer(
        model, 
        optimizer, 
        **trainer_params
    )
    return trainer

def train(dataset,  target, args):
    """
    Train the model for a specific target.
    """
    # Set random seed for each subject
    setup_seed(args.seed)
    source_loader, target_loader = get_maxuppm_laoder(args, dataset, target)
    # source_loader, target_loader = get_mutlisource_laoder(args, dataset, target)

    trainer = get_model_utils(args)
    
    # Train
    best_acc, np_log = trainer.train(source_loader, target_loader)    
    
    # Save model
    if args.saved_model:
        cur_target_saved_path = os.path.join(args.tmp_saved_path, str(target))
        create_dir_if_not_exists(cur_target_saved_path)
        # torch.save(trainer.get_model_state(), os.path.join(cur_target_saved_path, f"many_last.pth"))
        torch.save(trainer.get_best_model_state(), os.path.join(cur_target_saved_path, f"best.pth"))
    np.savetxt(os.path.join(args.tmp_saved_path, f"t{target}.csv"), np_log, delimiter=",", fmt='%.4f')
    return best_acc

def main(args):
    """
    Main function to run the training process.
    """
    setup_seed(args.seed)
    setattr(args, "max_iter", 1000) 
    create_dir_if_not_exists(args.tmp_saved_path)
    
    # load dataset 
    dataset = get_dataset(args)
    
    best_acc_mat = []
    for target in range(1, args.num_of_subjects + 1):
        best_acc = train(dataset, target, args)
        best_acc_mat.append(best_acc)
        print(f"target: {target}, best_acc: {best_acc}")
    
    mean = np.mean(best_acc_mat)
    std = np.std(best_acc_mat)

    # Write results to file
    with open(os.path.join(args.tmp_saved_path, f"mean_acc.txt"), 'w') as f:
        for target, best_acc in enumerate(best_acc_mat):
            output_line = f"target: {target+1}, best_acc: {best_acc:.6f}"
            print(output_line)  # Print to screen
            f.write(output_line + '\n')  # Write to file

        all_best_acc_line = f"all_best_acc: {mean:.4f} ± {std:.4f}"
        print(all_best_acc_line)  # Print to screen
        f.write(all_best_acc_line + '\n')  # Write to file

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    main(args)