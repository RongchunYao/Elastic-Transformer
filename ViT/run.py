import time
import os
import torch
import timm
import query
import torchvision
import numpy as np
from torch.utils.data import Dataset
import vision_transformer
import pathlib

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


cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
device = torch.device('cpu')

# model = timm.models.vision_transformer.VisionTransformer(depth=12).to(device)
model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)

model.eval()
model.oracle = query.fake_oracle(file_dir+'/profiling/profiling_result/profiling_result')

img_size = 224
normalize = torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
test_dataset = torchvision.datasets.ImageFolder(file_dir+'/ILSVRC2012',
                torchvision.transforms.Compose([
                    torchvision.transforms.Resize(int(img_size*8/7)),
                    torchvision.transforms.CenterCrop(img_size),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ]))        
        



import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=float, default=0.6, help='please specify the budget')
    parser.add_argument('--policy', type=str, default='v0', help='please specify the budget')
    parser.add_argument('--seed', type=int, default=2345, help='random seed of dataset')
    args = parser.parse_args()

    model.oracle.policy = args.policy
    small_dataset = Small_Dataset(test_dataset,-1,100,args.seed)
    data_loader = torch.utils.data.DataLoader(small_dataset, batch_size=1, shuffle=False, num_workers=1)

    model.budget = args.budget
    model.with_budget = True

    correct_num = 0
    count = 0
    max_duration = 0
    duration_sum = 0


    for data,target in data_loader:
        with torch.no_grad():
            start_time = time.time()
            output = model(data)
            duration = time.time()- start_time
            # model.oracle.print_log()
            ans = output.data.max(1,keepdim=True)[1]
            correct_num += int(ans.eq(target.data.view_as(ans)).sum())

            if duration > max_duration:
                max_duration = duration
            count += 1
            duration_sum += duration
            print(correct_num, count)
            print(duration)
            
            
    print('average duration: {}'.format(duration_sum/count))
    print('maximum duration: {}'.format(max_duration))
            
