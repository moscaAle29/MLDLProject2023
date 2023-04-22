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

            #load image into a numpy array
            im_frame = Image.open(path_to_image)
            numpy_image = np.array(im_frame)

            #load label into a numpy array
            im_frame = Image.open(path_to_label)
            numpy_label = np.array(im_frame)

            images.append(numpy_image)
            labels.append(numpy_label)
        
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
        # TODO: missing code here!
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.list_samples)


if __name__ == '__main__':
    list_samples = ["246999_T02_CN_A", "117898_T02_CN_A", "201551_T02_CN_A", "154799_T02_CN_A", "221819_T02_CN_A", "147584_T02_CN_A", "116081_T02_CN_A", "199782_T02_CN_A", "38549_T02_CN_A", "134949_T02_CN_A", "219658_T02_CN_A", "221088_T02_CN_A", "180859_T02_CN_A", "50442_T02_CN_A", "128671_T02_CN_A", "223244_T02_CN_A", "146645_T02_CN_A", "188954_T02_CN_A", "238348_T02_CN_A", "156597_T02_CN_A", "144880_T02_CN_A", "181787_T02_CN_A", "37489_T02_CN_A", "12048_T02_CN_A", "139486_T02_CN_A"]
    root = 'data/idda'

    dataset = IDDADataset(root=root,list_samples=list_samples)

    print("this is image")
    print(dataset.images[0])
    print("this is label")
    print(dataset.labels[0])