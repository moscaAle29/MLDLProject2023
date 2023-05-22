import os
from typing import Any, List
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import VisionDataset
import datasets.ss_transforms as tr

class_map = {  
    1: 13,  # ego_vehicle : vehicle
    7: 0,   # road
    8: 1,   # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    18: 5,  # poleGroup: pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car : vehicle
    27: 13,  # truck : vehicle
    28: 13,  # bus : vehicle
    32: 14,  # motorcycle
    33: 15,  # bycicle
}


class GTA5DataSet(VisionDataset):
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
            path_to_image = os.path.join(root, 'images', f'{filename}')
            path_to_label = os.path.join(root, 'labels', f'{filename}')

            #load image 
            img = Image.open(path_to_image)
            
            #load label 
            label = Image.open(path_to_label)
  

            images.append(img)
            labels.append(label)
        
        self.images = images
        self.labels = labels

    @staticmethod
    def get_mapping():
        mapping = np.zeros((256,), dtype=np.int64) + 255        
        for i, cl in class_map.items():
            mapping[i] = cl

        return lambda x: torch.from_numpy(mapping[x])

    def __getitem__(self, index: int) -> Any:
        #transform image
        if self.transform is not None:
            image, label = self.transform(self.images[index].convert('RGB'), self.labels[index])

        #transform label
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.list_samples)



