import os
import json

import torch

from torchvision.models import resnet18

import datasets.ss_transforms as sstr

import numpy as np
from PIL import Image
from torch import nn
from client import Client
from server import Server
from utils.args import get_parser
from utils.utils import extract_amp_spectrum
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

def create_style(args):
    if args.dataset2 == 'idda':
        root = 'data/idda'
        dir = os.path.join(root, 'bank_A')

        if not os.path.isdir(dir):
            os.makedirs(dir)

        with open(os.path.join(root, 'train.json'), 'r') as f:
            all_data = json.load(f)

            for client_id in all_data.keys():
                img_names = all_data[client_id]
                fft_magnitudes = []

                for filename in img_names:
                    path_to_image = os.path.join(
                        root, 'images', f'{filename}.jpg')
                    img = Image.open(path_to_image)
                    img_np = np.asanyarray(img, dtype=np.float32)

                    fft_magnitudes.append(extract_amp_spectrum(img_np))

                style = np.mean(np.array(fft_magnitudes), axis=0)

                np.save(file=os.path.join(dir, client_id), arr=style)
        
        return dir

def get_transforms(args):
    if args.model == 'deeplabv3_mobilenetv2':

        augmentations = []

        if args.rrc_transform is True:
            size = (args.h_resize, args.w_resize)
            scale = (args.min_scale, args.max_scale)

            augmentations.append(sstr.RandomResizedCrop(size = size, scale = scale))
 
        if args.canny is True:
            #canny edges transform
            augmentations.append(sstr.Canny())

        if args.flip is True:
            #randon horizonal flip transform
            augmentations.append(sstr.RandomHorizontalFlip())

        if args.random_rotation is True:
            #random rotation transform
            augmentations.append(sstr.RandomRotation(30))
        
        if args.domain_adapt == 'fda':
            dir = create_style(args)
            augmentations.append(sstr.TargetStyle(dir))
        
        if args.jitter is True:
            #color jitter transform
            augmentations.append(sstr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))

        augmentations.append(sstr.ToTensor())
       
        if args.canny is False:
            augmentations.append(sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        train_transforms = sstr.Compose(augmentations)

        if args.canny is False:
            test_transforms = sstr.Compose([
                sstr.ToTensor(),
                sstr.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            test_transforms = sstr.Compose([
                sstr.ToTensor()
            ])
    else:
        raise NotImplementedError
    return train_transforms, test_transforms



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
    
    train_transforms, test_transforms = get_transforms(args)

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
        elif args.setting == 'centralized':
            with open(os.path.join(root, 'train.txt'), 'r') as f:
                train_data = f.read().splitlines()
                train_datasets.append(IDDADataset(root=root, list_samples=train_data, client_name='single client'))
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
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')

        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]
    else:
        raise NotImplementedError

    return train_datasets, evaluation_datasets, test_datasets

def get_string(value):
    value=str(value)
    return value

def main():
    parser=get_parser()
    args=parser.parse_args()
    
    print("initializing model...")
    model=model_init(args)
    
    teacher=None
    teacher_kd = None
    if args.kd is True:
        teacher_kd = model_init(args)
    
    if args.task==1:
        load_path = os.path.join('checkpoints', 'task1',f'task1.ckpt')
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state"])
    
    elif args.task==2:
        if args.clients_per_round not in [2,4,8] or args.num_epochs not in [1,3,6,9,12]:
            print("you specified the wrong number of clients or epochs")
            return
        load_path = os.path.join('checkpoints', 'task2',f'{args.clients_per_round}clients_{args.num_epochs}epochs.ckpt')
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state"])
    
    elif args.task==3.2:
        if args.bs not in [2,4,6,8,10] or args.lr not in [0.01, 0.03, 0.05]:
            print("you specified the wrong batch size or learning rate")
            return
        #check with the guys
        load_path = os.path.join('checkpoints', 'task3.2',f'bs{args.bs}_lr{args.lr}')
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state"])
    
    elif args.task==3.4:
        if args.fda_alpha not in [0.1,0.01,0.05,0.005] :
            print("you specified the wrong value for alpha")
            return
        load_path = os.path.join('checkpoints', 'task3.4',f'bs4_fda{get_string(args.fda_alpha)}.ckpt')
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state"])
    
    elif args.task==4.2:
        if args.clients_per_round not in [2,8] or args.update_interval not in [0,1]:
            print("you specified the wrong number of clients or update interval")
            return
        teacher = model_init(args)
        load_path = os.path.join('checkpoints', 'task4.2',f'cl{args.clients_per_round}_ui{args.update_interval}.ckpt')
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state"])
        teacher.load_state_dict(checkpoint["model_state"])
    
    elif args.task==4.3:
        if args.clients_per_round not in [2,8] or args.update_interval not in [0,1]:
            print("you specified the wrong number of clients or update interval")
            return
        teacher = model_init(args)
        load_path = os.path.join('checkpoints', 'task4.3',f'cl{args.clients_per_round}_ui{args.update_interval}.ckpt')
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state"])
        teacher.load_state_dict(checkpoint["model_state"])
    
    else:
        print("not implemented")
        return
     
    if teacher is not None:
        teacher.cuda()  
    
    if teacher_kd is not None:
        teacher_kd.cuda()
        server.set_teacher_kd(teacher_kd)
    
    if teacher is not None:
        server.set_teacher(teacher)
    
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