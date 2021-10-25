from torch.utils.data import Dataset
import numpy as np
import torchvision
import pathlib
import os
import torch


file_dir = str(pathlib.Path(__file__).parent.resolve())
project_root_dir = str(pathlib.PurePath(file_dir, '..'))



def set_cpu_resource(cpu_num=1):
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

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



def ILSVRC2012_val_dataset(image_size=224, dataset_dir_name = 'ILSVRC2012'):

    dataset_abs_path = dataset_dir_name
    for path, directories, files in os.walk(project_root_dir):
        if dataset_dir_name in directories:
            dataset_abs_path = str(pathlib.PurePath(path, dataset_dir_name))

    if pathlib.Path(dataset_dir_name).is_dir():
        dataset_abs_path = dataset_dir_name
    
    normalize = torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    transfer = torchvision.transforms.Compose([
                        torchvision.transforms.Resize(int(image_size*8/7)),
                        torchvision.transforms.CenterCrop(image_size),
                        torchvision.transforms.ToTensor(),
                        normalize,
                    ])
    test_dataset = torchvision.datasets.ImageFolder(dataset_abs_path, transfer)

    return test_dataset


if __name__ == '__main__':
    '''
        just to test if we could find the default dataset directory
    '''
    ILSVRC2012_val_dataset()