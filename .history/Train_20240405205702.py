from JHL_Model import JHL
from pars_args import args
from Data_Loader import TrainValueDataLoader
from Trainer import Trainer
from logg import setup_logger

import numpy as np
import torch
import random


def train():
    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')
    print(args.gpu)
    #这里加上一个掩码的增强数据
    train_data_loader = TrainValueDataLoader(  #加载所有数据
        args.train_file, args.poi_file, args.batch_size)

    model = JHL(args, device, batch_first=True)

    trainer = Trainer(
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=None,
        n_epochs=args.n_epochs,
        device=device
    )

    trainer.train()


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('JHL_train.log')
    train()
