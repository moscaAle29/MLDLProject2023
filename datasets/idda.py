import os
from typing import Any, List
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr

class_eval = [255, 2, 4, 255, 11, 5, 0, 0, 1, 8, 13, 3, 7, 6, 255, 255, 15, 14, 12, 9, 10]


class IDDADataset(VisionDataset):

    def __init__(self,
                 root: str,
                 list_samples: List[str],
                 transform: tr.Compose = None,
                 client_name: str = None):
        super().__init__(root=root, transform=transform, target_transform=None)
        self.list_samples = list_samples
        self.client_name = client_name
        self.target_transform = self.get_mapping()

        #load images + labels
        images, labels = [], []

        for i, filename in enumerate(list_samples):
            path_to_image = os.path.join(root, 'images', f'{filename}.jpg')
            path_to_label = os.path.join(root, 'labels', f'{filename}.png')

            #load image 
            img = Image.open(path_to_image)
            
            #load label into a numpy array
            label = Image.open(path_to_label)
  

            images.append(img)
            labels.append(label)
        
        self.images = images
        self.labels = labels


    @staticmethod
    def get_mapping():
        classes = class_eval
        mapping = np.zeros((256,), dtype=np.int64) + 255
        for i, cl in enumerate(classes):
            mapping[i] = cl
        return lambda x: torch.from_numpy(mapping[x])

    def __getitem__(self, index: int) -> Any:
        #transform image
        if self.transform is not None:
            image, label = self.transform(self.images[index], self.labels[index])

        #transform label
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.list_samples)

