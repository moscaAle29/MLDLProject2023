import os
import json

import torch

from torchvision.models import resnet18

from torch import nn
from client import Client
from server import Server
from utils.args import get_parser
from datasets.idda import IDDADataset
from datasets.gta5 import GTA5DataSet
from models.deeplabv3 import deeplabv3_mobilenetv2
#from ..PIDNet.models import pidnet
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics


def get_dataset_num_classes(dataset):
    if dataset == 'idda' or dataset == 'gta5':
        return 16
    if dataset == 'femnist':
        return 62
    raise NotImplementedError

def model_init(args):
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    if args.model == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(
            in_features=512, out_features=get_dataset_num_classes(args.dataset))
        return model
    if args.model == 'pidnet':
            # cfg="./models/PIDNet/configs/cityscapes/pidnet_large_cityscapes.yaml"
            # cfg="./models/PIDNet/configs/cityscapes/pidnet_medium_cityscapes.yaml"
            cfg="./models/PIDNet/configs/cityscapes/pidnet_small_cityscapes.yaml"
 #           return pidnet.get_seg_model(cfg,imgnet_pretrained=True)
    raise NotImplementedError

def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2':
        metrics = {
            'eval_train': StreamSegMetrics(num_classes, 'eval_train'),
            'test_same_dom': StreamSegMetrics(num_classes, 'test_same_dom'),
            'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom')
        }
    elif args.model == 'resnet18' or args.model == 'cnn':
        metrics = {
            'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
            'test': StreamClsMetrics(num_classes, 'test')
        }
    else:
        raise NotImplementedError
    return metrics

def gen_clients(args, train_datasets, evaluation_datasets, test_datasets, model):
    clients = [[], []]
    single_client = None

    # in federated mode, train_datasets and evaluations_datasets are the same
    if args.setting == 'federated':

        for i, datasets in enumerate([train_datasets, test_datasets]):
            for ds in datasets:
                clients[i].append(Client(args, ds, model, test_client=i == 1))

    elif args.setting == 'centralized':
        # single client on which the model is trained
        for ds in train_datasets:
            single_client = Client(args, ds, model, test_client=False)

        for i, datasets in enumerate([evaluation_datasets, test_datasets]):
            for ds in datasets:
                clients[i].append(Client(args, ds, model, test_client=i == 1))

    else:
        raise NotImplemented

    return single_client, clients[0], clients[1]

def get_datasets(args):

    train_datasets = []
    evaluation_datasets = []
    test_datasets = []

    # determine the training dataset
    if args.dataset == 'idda':
        # the repositary where data and metadata is stored
        root = 'data/idda'

        # create training dataset based on running mode
        if args.setting == 'federated':
            with open(os.path.join(root, 'train.json'), 'r') as f:
                all_data = json.load(f)
                for client_id in all_data.keys():
                    train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], client_name=client_id))
        else:
            raise NotImplementedError

    elif args.dataset == 'gta5':
        # the repositary where data and metadata is stored
        root = 'data/gta5'
        with open(os.path.join(root, 'train.txt'), 'r') as f:
            train_data = f.read().splitlines()
            train_datasets.append(GTA5DataSet(root=root, list_samples=train_data, client_name='single client'))
    else:
        raise NotImplementedError
    #shuffle dataset
    # if train_datasets!= None:
    #     random.shuffle(train_datasets)
        
    # determine evaluation and testing datasets
    if args.dataset2 == 'idda':
        # the repositary where data and metadata is stored
        root = 'data/idda'

        # if the train dataset is idda and the running mode is federated
        # then the evaluation datasets and train datasets are the same
        if not (args.dataset == 'idda' and args.setting == 'federated'):
            with open(os.path.join(root, 'train.json'), 'r') as f:
                all_data = json.load(f)
                for client_id in all_data.keys():
                    evaluation_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], client_name=client_id))
        else:
            evaluation_datasets = train_datasets

        # test datasets
        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, client_name='test_same_dom')

        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data,client_name='test_diff_dom')
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]
    else:
        raise NotImplementedError

    return train_datasets, evaluation_datasets, test_datasets


def main():
    parser=get_parser()
    args=parser.parse_args()
    
    print("initializing model...")
    model=model_init(args)
    
    if args.task_2 is True: 
        load_path = os.path.join('checkpoints', 'task2',f'{args.clients_per_round}clients_{args.num_epochs}epochs.ckpt')
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state"])
    
    model.cuda()
    print("Done!")
    
    print('Generate datasets...')
    train_datasets, evaluation_datasets, test_datasets = get_datasets(args)
    print('Done!')
    
    metrics = set_metrics(args)
    single_client, train_clients, test_clients = gen_clients(
        args, train_datasets, evaluation_datasets, test_datasets, model)
    server = Server(args, single_client, train_clients,
                    test_clients, model, metrics)
    server.test(final_test=True)

if __name__=="__main__":
    print("---------------------START-------------------------")
    
    main()
    
    print("---------------------FINISH-------------------------")