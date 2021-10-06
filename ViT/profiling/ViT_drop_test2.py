import time
import torch
import torchvision
import random
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import uuid
import os
import timm
import pathlib
import argparse

file_dir = str(pathlib.Path(__file__).parent.resolve()) + '/'
PARALLEL = True


class Small_Dataset(Dataset):
    def __init__(self, parent_dataset, ratio, size, shuffle_seed):
        self.parent = parent_dataset
        index_array = np.arange(len(self.parent))
        self.shuffle_seed = shuffle_seed
        np.random.seed(self.shuffle_seed)
        np.random.shuffle(index_array)
        self.ratio = ratio
        if self.ratio < 0:
            self.length = len(self.parent)
        else:
            self.length = int(len(self.parent) * ratio)
        if size > 0:
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


def select(candidate_list, left_count):
    """Return all the combination of candidate_list, each combinatino's length is left_count.

    Args:
        candidate_list (arr): An array of candidate token index which we want to drop.
        left_count (int): The count of drop tokens

    Raises:
        ValueError: [description]

    Returns:
        arr: A list of combinations
    """
    if left_count == 0:
        return [[]]
    if len(candidate_list) < left_count or left_count < 0:
        raise ValueError('not enough candidate or left_count is not positive')
    if len(candidate_list) == left_count:
        return [candidate_list]
    list2ret = []
    if left_count == 1:
        for item in candidate_list:
            list2ret.append([item])
        return list2ret

    for select_index in range(0, len(candidate_list) - left_count + 1):
        tmp_list = select(candidate_list[select_index + 1:], left_count - 1)
        for item in tmp_list:
            item.append(candidate_list[select_index])
            list2ret.append(item)
    return list2ret


patch_order = "nothing"


def drop_hook(module, input):
    global patch_order
    return torch.cat([input[0][:, patch_order], input[0][:, -1, :].unsqueeze(dim=1)], dim=1)


def forward_once(device, model, data_loader, order, layer_index, is_cpu=False):
    global patch_order
    patch_order = order
    handle = model.blocks[layer_index].register_forward_pre_hook(drop_hook)
    total_consume_time = 0
    correct_num = 0
    with torch.no_grad():
        for data, target in data_loader:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            ender.record()
            torch.cuda.synchronize()
            ans = output.data.max(1, keepdim=True)[1]
            correct_num += int(ans.eq(target.data.view_as(ans)).sum())
            last_time = starter.elapsed_time(ender) / 1000
            total_consume_time += last_time

    handle.remove()
    return round(total_consume_time, 3), round(100 * correct_num / len(data_loader.dataset), 2)


def forward_once_without_timer(device,
                               model,
                               data_loader,
                               order,
                               layer_index,
                               num_heads=12,
                               layer_nums=12,
                               token_nums=197,
                               get_attn=False):
    global patch_order
    patch_order = order

    if PARALLEL:
        handle = model.module.blocks[layer_index].register_forward_pre_hook(drop_hook)
    else:
        handle = model.blocks[layer_index].register_forward_pre_hook(drop_hook)
    correct_num = 0
    attn_list = [torch.Tensor().to(device) for i in range(layer_nums)]
    loss_tensor = torch.Tensor().to(device)
    ans_tensor = torch.Tensor().to(device)
    ground_truth_tensor = torch.Tensor().to(device)
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='none')
            loss_tensor = torch.cat((loss_tensor, loss))
            ans = output.data.max(1, keepdim=True)[1]
            ans_tensor = torch.cat((ans_tensor, ans))
            ground_truth_tensor = torch.cat((ground_truth_tensor, target.data.view_as(ans)))
            # print(ans_tensor.shape, ground_truth_tensor.shape)
            correct_num += int(ans.eq(target.data.view_as(ans)).sum())
            # attn_matrix_list.append([blk.attn_score.sum(dim=-1).squeeze(dim=0)] for blk in model.blocks)
            if get_attn:
                if PARALLEL:
                    print('do not support with dp')
                    exit()
                # attn_score = torch.Tensor()
                for idx, blk in enumerate(model.blocks):
                    attn_list[idx] = torch.cat((attn_list[idx], blk.attn_score.sum(dim=-2).squeeze(dim=0)))

    handle.remove()
    return attn_list, loss_tensor, ans_tensor, ground_truth_tensor, correct_num / len(data_loader.dataset)


def test(device,
         model,
         exp_name,
         data_loader,
         patch_num,
         drop_list,
         layer_index,
         log_option=False,
         version="v0",
         get_attn=False):
    record_dict = {}
    drop_count = 0

    for drop in drop_list:
        start_time = time.time()
        patches_order = [i for i in range(patch_num)]
        for patch_index in drop:
            patches_order.remove(patch_index)

        attn_list, loss_tensor, ans_tensor, ground_truth_tensor, acc = forward_once_without_timer(device,
                                                                                                  model,
                                                                                                  data_loader,
                                                                                                  patches_order,
                                                                                                  layer_index,
                                                                                                  get_attn=get_attn)

        result = {}
        result['attn'] = attn_list
        result['loss'] = loss_tensor
        result['acc'] = acc
        result['output'] = ans_tensor
        result['ground_truth'] = ground_truth_tensor
        record_dict[tuple(drop)] = result
        drop_count += 1
        print(drop_count, time.time() - start_time)

    if log_option:
        if not os.path.exists(file_dir + "acc-savings/"):
            os.makedirs(file_dir + "acc-savings/")
        save_file_name = file_dir + '/acc-savings/' + exp_name + '_' + str(
            len(drop)) + 'drop_time' + str(drop_count) + 'layer' + str(layer_index) + '_' + version + str(uuid.uuid4())
        print(save_file_name)
        torch.save(record_dict, save_file_name)


def generate_drops(max_patch_num, drop_num, generate_time=100, random_seed=1234):
    if generate_time == -1:
        return select([i for i in range(max_patch_num)], drop_num)
    drop_list2ret = []
    candidate_list = [i for i in range(max_patch_num)]
    random.seed(random_seed)
    for i in range(generate_time):
        one_sample = random.sample(candidate_list, drop_num)
        while one_sample in drop_list2ret:
            one_sample = random.sample(candidate_list, drop_num)
        drop_list2ret.append(one_sample)
    return drop_list2ret


configuration = {
    # 'vit_base_patch16_224_base' :
    # {
    #     'model_name' : 'vit_base_patch16_224' ,
    #     'version' : 'vbase',
    #     'cpu' : False,
    #     'img_size' : 224,
    #     'total_patch_num' : 196,
    #     'drop_num_list' : [0],
    #     'batch_size' : 64,
    #     'drop_layer_list' : [0],
    #     'num_workers' : 8,
    #     'dataset_ratio' : 1/50,
    #     'dataset_size' : 1024,
    #     'get_attn' : True,
    #     'drop_time' : -1,
    # },
    'vit_base_patch16_224_drop_exp0': {
        'model_name': 'vit_base_patch16_224',
        'version': 'vtest',
        'cpu': False,
        'img_size': 224,
        'total_patch_num': 196,
        'drop_num_list': [10],
        'batch_size': 256,
        'drop_layer_list': [i for i in range(12)],
        'num_workers': 24,
        'dataset_ratio': 1 / 50,
        'dataset_size': 1024,
        'get_attn': False,
        'drop_time': -1,
    },

    #  'vit_base_patch16_224_drop_exp1' :
    #  {
    #      'model_name' : 'vit_base_patch16_224' ,
    #      'version' : 'vtest',
    #      'cpu' : False,
    #      'img_size' : 224,
    #      'total_patch_num' : 196,
    #      'drop_num_list' : [i for i in range(2,6)],
    #      'batch_size' : 256,
    #      'drop_layer_list' : [i for i in range(12)],
    #      'num_workers' : 24,
    #      'dataset_ratio' : 1/50,
    #      'dataset_size' : 1024,
    #      'get_attn' : False,
    #      'drop_time' : 1000,
    #  },

    #  'vit_base_patch16_224_drop_exp2' :
    #  {
    #      'model_name' : 'vit_base_patch16_224' ,
    #      'version' : 'vtest',
    #      'cpu' : False,
    #      'img_size' : 224,
    #      'total_patch_num' : 196,
    #      'drop_num_list' : [i for i in range(2,6)],
    #      'batch_size' : 256,
    #      'drop_layer_list' : [i for i in range(12)],
    #      'num_workers' : 24,
    #      'dataset_ratio' : 1/50,
    #      'dataset_size' : 1024,
    #      'get_attn' : False,
    #      'drop_time' : 1000,
    #  },
}

if __name__ == '__main__':

    # cpu_num = 1
    # os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    # os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    # os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    # os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    # os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    # torch.set_num_threads(cpu_num)

    # print(len(generate_drops(196,0,-1)))
    # print(len(generate_drops(196,1,-1)))
    # print(len(generate_drops(196,2,-1)))
    # exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for exp_name, config in configuration.items():
        exp_name = 'vit_base_patch16_224'
        model_name = config['model_name']
        total_patch_num = config['total_patch_num']
        img_size = config['img_size']
        version = config['version']
        is_cpu = config['cpu']

        if is_cpu:
            device = torch.device('cpu')
        elif PARALLEL:
            device = torch.device("cuda")
        else:
            device = torch.device("cuda:" + str(args.gpu))

        model = timm.create_model(model_name, pretrained=True)
        model = model.to(device)
        if PARALLEL:
            model = torch.nn.DataParallel(model)
        model.eval()
        num_workers = config['num_workers']
        dataset_ratio = config['dataset_ratio']
        dataset_size = config['dataset_size']

        # this is for the warmup
        normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        test_dataset = torchvision.datasets.ImageFolder(
            file_dir + '../ILSVRC2012',
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(int(img_size * 8 / 7)),
                torchvision.transforms.CenterCrop(img_size),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))

        small_dataset = Small_Dataset(test_dataset, dataset_ratio, dataset_size, 1234)
        data_loader = torch.utils.data.DataLoader(small_dataset,
                                                  persistent_workers=True,
                                                  batch_size=config['batch_size'],
                                                  shuffle=False,
                                                  num_workers=num_workers)

        drop_num_list = config['drop_num_list']
        drop_layer_list = config['drop_layer_list']
        drop_dict = {}

        for drop_num in drop_num_list:
            drop_dict[drop_num] = generate_drops(total_patch_num, drop_num, generate_time=config['drop_time'])

        log_option = True

        for drop_layer_index in drop_layer_list:
            for drop_num in drop_num_list:
                test(device,
                     model,
                     exp_name,
                     data_loader,
                     total_patch_num,
                     drop_dict[drop_num],
                     drop_layer_index,
                     log_option,
                     version,
                     get_attn=config['get_attn'])
