import copy
import torch

from datasets.ss_transforms import Canny
from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.utils import HardNegativeMining, MeanReduction, SelfTrainingLoss

from PIL import Image

import wandb

import matplotlib.pyplot as plt

class_labels = {
    0: "road",
    1: "side walk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffict light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "vehicle",
    14: "motorcycle",
    15: "bicycle",
    255: "others"
}

class Client:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        if self.args.self_supervised is True:
            self.pseudo_label_generator = SelfTrainingLoss()

        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

        #this is used for test clients, dont delete it
        self.logger = None


    def __str__(self):
        return self.name

    @staticmethod
    def update_metric(metric, outputs, labels):
        _, prediction = outputs.max(dim=1) #return values, indices and we only want indices
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)
    
    def set_teacher(self, teacher):
        self.pseudo_label_generator.set_teacher(teacher)

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        raise NotImplementedError
    
    def calc_losses(self, images, labels):

        #imgs2 = images.clone().detach()

        outputs = self.model(images)['out']

        if self.args.self_supervised is True:
            #store current params of model
            labels = self.pseudo_label_generator(outputs, images)



        loss_tot = self.reduction(self.criterion(outputs, labels), labels)
        #dict_calc_losses = {'loss_tot' : loss_tot}

        return loss_tot, outputs
    
    def generate_update(self):
        return copy.deepcopy(self.model.state_dict())

    def run_epoch(self, cur_epoch, optimizer):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #dict_all_epoch_losses = defaultdict(lambda: 0)

        for cur_step, (images, labels) in enumerate(self.train_loader):
            #if we use canny we have to feed transformed images to the model
            if self.args.canny is True:
                canny_transform= Canny()
                new_images=[]
                for img in images:
                    new_images.append(canny_transform(img))
                images=new_images.to(device, dtype=torch.float32)
            else:
                images = images.to(device, dtype = torch.float32)
            labels = labels.to(device, dtype = torch.long)

            optimizer.zero_grad()

            loss_tot, outputs = self.calc_losses(images, labels)
            loss_tot.backward()

            optimizer.step()

    def train(self):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
        self.model.train()

        #run the specified number of epochs
        for epoch in range(self.args.num_epochs):
            self.run_epoch(epoch, optimizer)
        
        return len(self.dataset), self.generate_update()

    def eval_train(self,metric):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model.eval()

        with torch.no_grad():
            for cur_step, (images, labels) in enumerate(self.train_loader):
                images = images.to(device, dtype = torch.float32)
                labels = labels.to(device, dtype = torch.long)
            
                outputs = self.model(images)['out']

                #metric.update(label_trues=labels, label_preds=output)
                self.update_metric(metric, outputs, labels)
            
    def test(self, metric, test_phase = False):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        self.model.eval()

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                images_ = images.to(device, dtype = torch.float32)
                labels = labels.to(device, dtype = torch.long)

                outputs = self.model(images_)['out']

                self.update_metric(metric, outputs, labels)

                if i % 50 == 0 and test_phase:
                    #this is used to creat a table for wandb
                    #data = []
                    #columns = ['image', "prediction", "truth"]

                    print(f'{self.name}-{i}')
                    _, prediction = outputs.max(dim=1)

                    images = torch.squeeze(images, 0)
                    prediction = torch.squeeze(prediction.cpu(), 0)
                    labels = torch.squeeze(labels.cpu(), 0)

                    img1 = wandb.Image(images)
                    masks = {
                                "prediction": {"mask_data" : prediction.numpy(), "class_labels": class_labels},
                                "ground_truth": {"mask_data": labels.numpy(), "class_labels": class_labels}
                            }
                    self.logger.log_image(key=f'{self.name}-{i}', images = [img1], masks = [masks])
