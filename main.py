import copy
import os
import json
from collections import defaultdict

import torch
import random

import numpy as np
from torchvision.models import resnet18

import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr

from torch import nn
from client import Client
from server import Server
from utils.args import get_parser
from datasets.idda import IDDADataset
from datasets.gta5 import GTA5DataSet
from models.deeplabv3 import deeplabv3_mobilenetv2
#from ..PIDNet.models import pidnet
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics
from utils.utils import extract_amp_spectrum
from utils.logger import Logger

from PIL import Image

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#from models.vae import VAE
import torch.nn.functional as F

from torch.utils.data import DataLoader

from pl_bolts.models.autoencoders import VAE
import pytorch_lightning as pl



def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
        cfg = "./models/PIDNet/configs/cityscapes/pidnet_small_cityscapes.yaml"
 #           return pidnet.get_seg_model(cfg,imgnet_pretrained=True)
    raise NotImplementedError


def create_style(args):
    if args.dataset2 == 'idda':
        root = 'data/idda'
        dir = os.path.join(root, 'train')

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


def create_style_test(args):
    if args.dataset2 == 'idda':
        root = 'data/idda'
        dir = os.path.join(root, 'test')

        if not os.path.isdir(dir):
            os.makedirs(dir)

        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()

            all_data = {'test_same_dome': test_same_dom_data}

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

        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()

            all_data = {'test_diff_dome': test_diff_dom_data}

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
    # TODO: test your data augmentation by changing the transforms here!
    if args.model == 'deeplabv3_mobilenetv2':

        augmentations = []

        if args.domain_adapt == 'fda':
            dir = create_style(args)
            augmentations.append(sstr.TargetStyle(dir, args.fda_alpha))


        if args.rrc_transform is True:
            size = (args.h_resize, args.w_resize)
            scale = (args.min_scale, args.max_scale)

            augmentations.append(
                sstr.RandomResizedCrop(size=size, scale=scale))

        if args.flip is True:
            augmentations.append(sstr.RandomHorizontalFlip())

        if args.random_rotation is True:
            augmentations.append(sstr.RandomRotation(30))
        if args.jitter is True:
            augmentations.append(sstr.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4))

        augmentations.append(sstr.ToTensor())
        augmentations.append(sstr.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        train_transforms = sstr.Compose(augmentations)

        test_transforms = sstr.Compose([
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        ])
    elif args.model == 'cnn' or args.model == 'resnet18':
        train_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])
        test_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])
    else:
        raise NotImplementedError
    return train_transforms, test_transforms


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
                    train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms,
                                                      client_name=client_id))
        elif args.setting == 'centralized':
            with open(os.path.join(root, 'train.txt'), 'r') as f:
                train_data = f.read().splitlines()
                train_datasets.append(IDDADataset(root=root, list_samples=train_data, transform=train_transforms,
                                                  client_name='single client'))
        else:
            raise NotImplementedError

    elif args.dataset == 'gta5':
        # the repositary where data and metadata is stored
        root = 'data/gta5'
        with open(os.path.join(root, 'train.txt'), 'r') as f:
            train_data = f.read().splitlines()
            train_datasets.append(GTA5DataSet(root=root, list_samples=train_data, transform=train_transforms,
                                              client_name='single client'))
    # #TODO: implement cityscapes dataset
    # elif args.dataset == 'cityscapes':
    #     # the repositary where data and metadata is stored
    #     root = 'data/cityscapes'
    #     with open(os.path.join(root, 'train.txt'), 'r') as f:
    #         train_data = f.read().splitlines()
    #         train_datasets.append(GTA5DataSet(root=root, list_samples=train_data, transform=train_transforms,
    #                                           client_name='single client'))
    else:
        raise NotImplementedError
    # shuffle dataset
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
                    evaluation_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms,
                                                           client_name=client_id))
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


def create_style_based_clusters(args):

    cluster_mapping = {}

    dir = create_style(args)
    test_dir = create_style_test(args)

    # list of client id
    client_ids = os.listdir(dir)
    test_client_ids = os.listdir(test_dir)

    X = []
    X_test = []

    for client_id in client_ids:
        row = np.load(os.path.join(dir, client_id)).flatten()
        X.append(row)

    for client_id in test_client_ids:
        row = np.load(os.path.join(test_dir, client_id)).flatten()
        X_test.append(row)

    X = np.array(X)
    X_test = np.array(X_test)

    model_list = []
    res_list = []
    score_list = []

    # if self.args.force_k == 0:
    #    k_list = list(range(4, 20))
    # else:
    #    k_list = [self.args.force_k]

    k_list = list(range(4, 20))

    for k_size in k_list:
        model = KMeans(n_clusters=k_size, n_init=10).fit(X)
        model_list.append(model)
        res_list.append(model.labels_)
        score_list.append(silhouette_score(X, model.labels_))

    best_id = np.argmax(score_list)
    k_means_model = model_list[best_id]

    clusters_of_test_clients = k_means_model.predict(X_test)
    # if self.args.force_k == 0:
    #    k_means_relative_path = "_model.pkl"
    # else:
    #    k_means_relative_path = f"_model_{self.args.force_k}"
    #pickle.dump(self.k_means_model, open(self.cluster_path.split(".json")[0] + k_means_relative_path, "wb"))
    k_size = k_list[best_id]
    #self.writer.write(f"best k {self.k_size}")
    #self.writer.write(f"best silhouette_score {score_list[best_id]}")
    for cluster_id in range(k_size):
        cluster_mapping[cluster_id] = [client_ids[i]
                                       for i, val in enumerate(res_list[best_id])
                                       if val == cluster_id]
    for i, cluster_id in enumerate(clusters_of_test_clients):
        cluster_mapping[cluster_id].append(test_client_ids[i])

    return cluster_mapping

def create_vae_based_clusters1(args):
    root = './data/generated_imgs'
    if not os.path.isdir(root):
        os.makedirs(root)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_datasets, test_datasets = get_dataset_vae()
    # Initialize the network and the Adam optimizer
    for train_dataset in train_datasets:
        print(f"train vae on client_{train_dataset.client_name}")
        net = VAE(imgChannels=3).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
        data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

        num_epochs = 20
        for epoch in range(num_epochs):
            for idx, data in enumerate(data_loader, 0):
                imgs, _ = data
                imgs = imgs.to(device, dtype = torch.float32)

                # Feeding a batch of images into the network to obtain the output image, mu, and logVar
                out, mu, logVar = net(imgs)

                # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
                kl_divergence = 0.5 * \
                    torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
                loss = F.mse_loss(
                    out, imgs, reduction = 'mean') + kl_divergence

                # Backpropagation based on the loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #generate imgs to train the master VAE
        for i in range(20):
            name = f'{train_dataset.client_name}_{i}'
            path = os.path.join(root, name)

            img = net.generate_img()

            torch.save(img, path)
    
    #train the master VAE
    print("train the master VAE")
    net = VAE(imgChannels=3).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

    for epoch in range(num_epochs):
        for img_name in os.listdir(root):
            imgs= torch.load(os.path.join(root, img_name))
            imgs = imgs.to(device, dtype = torch.float32)

            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out, mu, logVar = net(imgs)

            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            kl_divergence = 0.5 * \
                torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.mse_loss(
                out, imgs, reduction='mean') + kl_divergence

            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    #Start clustering
    print("Start clustering")
    X = []
    client_ids = []
    X_test = []
    test_client_ids = []
    
    for train_dataset in train_datasets:
        client_ids.append(train_dataset.client_name)
        mu_list = []
        data_loader = DataLoader(train_dataset, batch_size=1)
        for idx, data in enumerate(data_loader, 0):
            imgs, _ = data
            imgs = imgs.to(device, dtype = torch.float32)

            with torch.no_grad():
                _, mu, _ = net(imgs)
            mu = mu.squeeze()
            mu_list.append(mu.cpu().numpy())

        mu_list = np.array(mu_list)
        avg = np.mean(mu_list, axis = 0)

        X.append(avg)
    
    X = np.array(X)

    print(f'mu_list shape = {np.shape(mu_list)}')
    print(f'avg     shape = {np.shape(avg)}')
    print(f'X       shape = {np.shape(X)}')

    for test_dataset in test_datasets:
        client_ids.append(test_dataset.client_name)
        mu_list = []
        data_loader = DataLoader(test_dataset, batch_size=1)

        for idx, data in enumerate(data_loader, 0):
            imgs, _ = data
            imgs = imgs.to(device, dtype = torch.float32)

            with torch.no_grad():
                _, mu, _ = net(imgs)
            mu = mu.squeeze()
            mu_list.append(mu.cpu().numpy())

        mu_list = np.array(mu_list)
        avg = np.mean(mu_list, axis = 0)

        X_test.append(avg)
    
    X = np.array(X)


    model_list = []
    res_list = []
    score_list = []

    # if self.args.force_k == 0:
    #    k_list = list(range(4, 20))
    # else:
    #    k_list = [self.args.force_k]

    k_list = list(range(4, 20))

    for k_size in k_list:
        model = KMeans(n_clusters=k_size, n_init=10).fit(X)
        model_list.append(model)
        res_list.append(model.labels_)
        score_list.append(silhouette_score(X, model.labels_))

    best_id = np.argmax(score_list)
    k_means_model = model_list[best_id]

    clusters_of_test_clients = k_means_model.predict(X_test)
    # if self.args.force_k == 0:
    #    k_means_relative_path = "_model.pkl"
    # else:
    #    k_means_relative_path = f"_model_{self.args.force_k}"
    #pickle.dump(self.k_means_model, open(self.cluster_path.split(".json")[0] + k_means_relative_path, "wb"))
    k_size = k_list[best_id]
    #self.writer.write(f"best k {self.k_size}")
    #self.writer.write(f"best silhouette_score {score_list[best_id]}")
    cluster_mapping = {}
    for cluster_id in range(k_size):
        cluster_mapping[cluster_id] = [client_ids[i]
                                       for i, val in enumerate(res_list[best_id])
                                       if val == cluster_id]
    for i, cluster_id in enumerate(clusters_of_test_clients):
        cluster_mapping[cluster_id].append(test_client_ids[i])

    print(cluster_mapping)
    return cluster_mapping

def create_vae_based_clusters(args):
    pretrained_dataset, train_datasets, test_datasets = get_dataset_vae()

    net = VAE().cuda()

    #pretrain
    print('pretrain on gta5')
    data_loader = DataLoader(pretrained_dataset, batch_size=4, shuffle=True, drop_last=True)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(net, train_dataloader=data_loader)

    #fine_tuning
    for train_dataset in train_datasets:
        print(f'fine tune on client_{train_dataset.client_name}')
        data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
        trainer = pl.Trainer(gpus=1)
        trainer.fit(net, train_dataloaders=data_loader)
    
    #find representation for each client in laten space
    print('start clustering')
    X = []
    X_test = []
    client_ids = []
    test_client_ids = []

    
    for train_dataset in train_datasets:
        data_loader = DataLoader(train_dataset, batch_size=1)
        client_ids.append(train_dataset.client_name)
        mu_list = []

        for idx, data in enumerate(data_loader, 0):
            img, _ = data
            img = img.cuda()

            x = net.encoder(img)
            mu = net.fc_mu(x)

            mu = mu.squeeze()
            mu_list.append(mu.cpu().numpy())
        
        mu_list = np.array(mu_list)
        avg = np.mean(mu_list, axis = 0)

        X.append(avg)

    X = np.array(X)

    for test_dataset in test_datasets:
        data_loader = DataLoader(test_dataset, batch_size=1)
        test_client_ids.append(test_dataset.client_name)
        mu_list = []

        for idx, data in enumerate(data_loader, 0):
            img, _ = data
            img = img.cuda()

            x = net.encoder(img)
            mu = net.fc_mu(x)

            mu = mu.squeeze()
            mu_list.append(mu.cpu().numpy())
        
        mu_list = np.array(mu_list)
        avg = np.mean(mu_list, axis = 0)

        X_test.append(avg)

    X_test = np.array(X)

    model_list = []
    res_list = []
    score_list = []

    # if self.args.force_k == 0:
    #    k_list = list(range(4, 20))
    # else:
    #    k_list = [self.args.force_k]

    k_list = list(range(4, 20))

    for k_size in k_list:
        model = KMeans(n_clusters=k_size, n_init=10).fit(X)
        model_list.append(model)
        res_list.append(model.labels_)
        score_list.append(silhouette_score(X, model.labels_))

    best_id = np.argmax(score_list)
    k_means_model = model_list[best_id]

    clusters_of_test_clients = k_means_model.predict(X_test)
    # if self.args.force_k == 0:
    #    k_means_relative_path = "_model.pkl"
    # else:
    #    k_means_relative_path = f"_model_{self.args.force_k}"
    #pickle.dump(self.k_means_model, open(self.cluster_path.split(".json")[0] + k_means_relative_path, "wb"))
    k_size = k_list[best_id]
    #self.writer.write(f"best k {self.k_size}")
    #self.writer.write(f"best silhouette_score {score_list[best_id]}")
    cluster_mapping = {}
    for cluster_id in range(k_size):
        cluster_mapping[cluster_id] = [client_ids[i]
                                       for i, val in enumerate(res_list[best_id])
                                       if val == cluster_id]
    for i, cluster_id in enumerate(clusters_of_test_clients):
        cluster_mapping[cluster_id].append(test_client_ids[i])

    print(cluster_mapping)
    return cluster_mapping

def get_dataset_vae():
    root = 'data/idda'
    train_datasets = []

    #transform
    resize = sstr.RandomResizedCrop(size=(50,50))
    normalization = sstr.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    
    transform = sstr.Compose([resize,sstr.ToTensor(), normalization])


    with open(os.path.join(root, 'train.json'), 'r') as f:
        all_data = json.load(f)
        for client_id in all_data.keys():
            train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=transform,
                                                client_name=client_id))
            
    with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=transform,
                                                client_name='test_same_dom')

    with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
        test_diff_dom_data = f.read().splitlines()
        test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=transform,
                                            client_name='test_diff_dom')
    test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]

    pretrained_root = 'data/gta5'

    with open(os.path.join(pretrained_root, 'train.txt'), 'r') as f:
        train_data = f.read().splitlines()
        pretrained_dataset = GTA5DataSet(root=pretrained_root, list_samples=train_data, transform=transform,
                                            client_name='single client')


    return pretrained_dataset, train_datasets, test_datasets


def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    print(f'Initializing model...')
    model = model_init(args)

    # initialize teacher if setting is semi_supervised
    teacher = None
    if args.self_supervised is True:
        teacher = model_init(args)

    # initialize teacher_kd if knowledge distillation is applied
    teacher_kd = None
    if args.kd is True:
        teacher_kd = model_init(args)

    # load pre_trained model if specified
    if args.load_pretrained is True:
        print('Loading pretrained model...')
        project = args.run_path.split('/')[1]
        repo = project.split('_')
        load_path = os.path.join('checkpoints',repo[0], repo[1], repo[2] ,f'round{args.round}.ckpt')
        run_path = args.run_path
        root = '.'

        Logger.restore(name=load_path, run_path=run_path, root=root)
        if args.model == 'deeplabv3_mobilenetv2':
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint["model_state"])
            if teacher is not None:
                teacher.load_state_dict(checkpoint["model_state"])
            if teacher_kd is not None:
                teacher_kd.load_state_dict(checkpoint["model_state"])


    if teacher is not None:
        teacher.cuda()  
    if teacher_kd is not None:
        teacher_kd.cuda()

    model.cuda()
    print('Done.')

    print('Generate datasets...')
    train_datasets, evaluation_datasets, test_datasets = get_datasets(args)
    print('Done.')

    metrics = set_metrics(args)
    single_client, train_clients, test_clients = gen_clients(
        args, train_datasets, evaluation_datasets, test_datasets, model)
    server = Server(args, single_client, train_clients,
                    test_clients, model, metrics)

    if args.self_supervised is True:
        server.set_teacher(teacher)

    if args.kd is True:
        server.set_teacher_kd(teacher_kd)

    if args.clustering is not None:
        # associate client to its cluster
        if args.clustering == 'ladd':
            cluster_mapping = create_style_based_clusters(args)
        elif args.clustering == 'vae':
            cluster_mapping = create_vae_based_clusters(args)

        client_dic = {}
        for client in train_clients:
            client_dic[client.name] = client
        for client in test_clients:
            client_dic[client.name] = client

        for cluster_id in cluster_mapping.keys():
            for client_id in cluster_mapping[cluster_id]:
                client = client_dic[client_id]
                client.cluster_id = cluster_id

        # inform server about the number of clusters
        server.number_of_clusters = len(cluster_mapping.items())
        server.model_params_dict = [copy.deepcopy(
            server.model_params_dict) for i in range(server.number_of_clusters)]

    server.train()
    server.test()

# this method is not utilized anymore
def centralised():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    print(f'Initializing model...')
    model = model_init(args)
    model.cuda()
    print('Done.')

    print('Generate datasets...')
    train_datasets, test_datasets = get_datasets_centralised(args)
    print('Done.')

    metrics = set_metrics(args)
    train_clients, test_clients = gen_clients(
        args, train_datasets, test_datasets, model)

    for train_client in train_clients:
        train_client.train()

    print("finish training")

    test_clients[0].test(metrics['test_same_dom'])
    test_clients[1].test(metrics['test_diff_dom'])

    print("finish testing")

# this method is not utilized anymore
def get_datasets_centralised(args):

    train_datasets = []
    train_transforms, test_transforms = get_transforms(args)

    if args.dataset == 'idda':
        root = 'data/idda'

        with open(os.path.join(root, 'train.txt'), 'r') as f:
            train_data = f.read().splitlines()
            train_datasets.append(IDDADataset(root=root, list_samples=train_data, transform=train_transforms,
                                              client_name='test_same_dom'))

        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')
        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]

    return train_datasets, test_datasets


if __name__ == '__main__':
    main()

    print("finish running script")
