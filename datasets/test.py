import os
from PIL import Image
import numpy as np

if __name__=='__main__':
    root = 'D:\\Users\\Admin\\Dataset\\data\\GTA5\\labels'
    names = []

    with open(os.path.join('data/gta5', 'train.txt'), 'r') as f:
        names = f.read().splitlines()
    
    max = 0

    for name in names:
        #print(name)
        path = os.path.join(root, name)
        img = Image.open(path)

        number = np.max(np.array(img))

        if number > max:
            max = number
    
    print(np.array(img).shape)
    print(max)
