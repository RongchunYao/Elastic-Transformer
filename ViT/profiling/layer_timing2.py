import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
import pathlib

import sys
sys.path = sys.path[1:]
import timm
sys.path = [''] + sys.path

file_dir = str(pathlib.Path(__file__).parent.resolve())+'/'


class Small_Dataset(Dataset):
    
    def __init__(self, parent_dataset,  ratio, size, shuffle_seed):
        self.parent = parent_dataset
        index_array = np.arange(len(self.parent))
        self.shuffle_seed = shuffle_seed
        np.random.seed(self.shuffle_seed)
        np.random.shuffle(index_array)
        self.ratio = ratio
        if self.ratio<0:
            self.length = len(self.parent)
        else:
            self.length = int(len(self.parent)*ratio)
        if size>0 :
            self.length = size

        if self.length > len(self.parent):
            raise ValueError('subset could not be bigger than parent')
        self.lookup_array = index_array[:self.length]
        self.data_list = []
        for i in range(len(self.lookup_array)):
            self.data_list.append(self.parent[self.lookup_array[i]])


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data_list[index]


device = torch.device('cuda')

model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)

img_size = 224
normalize = torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
test_dataset = torchvision.datasets.ImageFolder(file_dir + '../ILSVRC2012',
                torchvision.transforms.Compose([
                    torchvision.transforms.Resize(int(img_size*8/7)),
                    torchvision.transforms.CenterCrop(img_size),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ]))        

small_dataset = Small_Dataset(test_dataset,-1,100,2345)

data_loader = torch.utils.data.DataLoader(small_dataset,persistent_workers = True, batch_size=10, shuffle=False, num_workers=4)

correct_num = 0
for data,target in data_loader:
    with torch.no_grad():
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        ans = output.data.max(1,keepdim=True)[1]
        correct_num += int(ans.eq(target.data.view_as(ans)).sum())

print(correct_num)