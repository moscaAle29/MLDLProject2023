import torch.nn as nn
from utils.logger import Logger, get_job_name, get_project_name
import os


class HardNegativeMining(nn.Module):

    def __init__(self, perc=0.25):
        super().__init__()
        self.perc = perc

    def forward(self, loss, _):
        b = loss.shape[0]
        loss = loss.reshape(b, -1)
        p = loss.shape[1]
        tk = loss.topk(dim=1, k=int(self.perc * p))
        loss = tk[0].mean()
        return loss


class MeanReduction:
    def __call__(self, x, target):
        x = x[target != 255]
        return x.mean()


def set_up_logger(args):

    logger = Logger(name = get_project_name(args), project = get_project_name(args))

    return logger

def get_checkpoint_path(args):
    if args.setting == 'federated':
        ckpt_path = os.path.join('checkpoints', args.setting, args.dataset ,args.dataset2, args.algorithm)
    elif args.setting == 'centralized':
        ckpt_path = os.path.join('checkpoints', args.setting, args.dataset ,args.dataset2)

    return ckpt_path
