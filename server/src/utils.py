# utils.py
import torch
import torch.nn as nn
import numpy as np

def initialize_weights(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight.data, nonlinearity='relu')
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight.data, 1)
        nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight.data)
        nn.init.constant_(model.bias.data, 0)

def adjust_learning_rate(optimizer, epoch, init_lr=0.01, lr_decay_epoch=30):
    """Decays learning rate every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
