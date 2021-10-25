import time
import os
import torch
import timm
from ViT import query, vision_transformer, cola_utils
import torchvision
import pathlib
import argparse

file_dir = str(pathlib.Path(__file__).parent.resolve())
project_root_dir = str(pathlib.PurePath(file_dir, '..'))



def run_demo(budget = 0.6,  policy='v0', profiling_file = 'profiling_result', dataset_dir = 'ILSVRC2012', sample_num = 100, dataset_seed = 2345):

    model_name = 'vit_base_patch16_224'
    device_type = 'cpu'
    batch = 1
    profiling_dir = str(pathlib.PurePath(file_dir, 'profiling', 'profiling_result'))
    abs_profiling_path = profiling_file
    
    for path, directories, files in os.walk(profiling_dir):
        if profiling_file in files:
            abs_profiling_path = str(pathlib.PurePath(path, profiling_file))
    
    if pathlib.Path(profiling_file).is_file():
        abs_profiling_path = profiling_file

    cola_utils.set_cpu_resource()
    device = torch.device(device_type)
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()
    dummy_oracle = query.fake_oracle(abs_profiling_path, policy)
    model.set_budget(budget)
    model.set_oracle(dummy_oracle)
    validation_dataset = cola_utils.ILSVRC2012_val_dataset(dataset_dir_name=dataset_dir)
    small_dataset = cola_utils.Small_Dataset(validation_dataset,-1,sample_num,dataset_seed)
    data_loader = torch.utils.data.DataLoader(small_dataset, batch_size=batch, shuffle=False, num_workers=1)
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', '-budget', type=float, default=0.6, help='please specify the budget')
    parser.add_argument('--policy', '-policy', type=str, default='v0', help='please specify the policy, default: v0.\n \
        v0 : dummy policy.\n \
        v1 : random drop.\n')
    parser.add_argument('--dataset', '-dataset', type=str, default='ILSVRC2012', help='specify the path of dataset, default: ILSVRC2012')
    parser.add_argument('--sample-num', '-sample-num', type=int, default=100, help='how many images to run as a subset of dataset (IMAGENET2012 validation), default value: 100')
    parser.add_argument('--seed', '-seed', type=int, default=2345, help='random seed of choosing a image from dataset, default: 2345')
    parser.add_argument('--profiling-file', '-profiling-file', type=str, default='profiling_result', help='specify which profiling file to use, default value: profiling_result')
    args = parser.parse_args()
    run_demo(budget=args.budget, policy=args.policy, profiling_file=args.profiling_file, dataset_seed=args.seed, sample_num=args.sample_num, dataset_dir=args.dataset)
