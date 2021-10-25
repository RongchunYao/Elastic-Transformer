'''
import torch

drop0 = torch.load('./acc-savings/vit_base_patch16_224_0drop_time1layer0_vbase20003a07-7b1b-4cca-ae74-bb86dd1bf73f')
drop1 = torch.load('./acc-savings/vit_base_patch16_224_1drop_time196layer0_vtest4da5977b-8ef6-4511-8e95-be62b7a9a942')

# print(drop0[tuple()]['attn'][0].shape)
# print(drop1.keys())

layer_0_attn_score = drop0[tuple()]['attn'][0].sum(dim=1)
sorted_indexes = layer_0_attn_score[0].argsort()
# print(layer_0_attn_score[0].argsort())
# print(layer_0_attn_score[0][sorted_indexes])
# print(drop1[tuple([0])]['loss'][0])
# print(drop1[tuple([183])]['loss'][0])
# print(drop1[tuple([79])]['loss'][0])

print(drop0[tuple()]['attn'][0].sum(dim=1)[:,0])

'''

import torch
import pathlib
import os
import numpy as np

file_dir = str(pathlib.Path(__file__).parent.resolve())+'/'
acc_file_dir = file_dir + '../measurement_data/'
# acc_file_dir = file_dir + './acc-savings/'
acc_files = os.listdir(acc_file_dir)

base_profiling_file = ''
drop_on_layer0_files = []

device = torch.device('cuda')

for filename in acc_files:
    if '_0drop' in filename:
        base_profiling_file = filename
    elif 'layer0' in filename and '_10drop' in filename:
        drop_on_layer0_files.append(filename)

base_profiling = torch.load(acc_file_dir+base_profiling_file)
# base_profiling = torch.load('./acc-savings/vit_base_patch16_224_0drop_time1layer0_vbase3c875927-70cb-486e-ac5d-2e224be91308')
# base_profiling = torch.load('acc-savings/vit_base_patch16_224_0drop_time1layer0_vbase538ab385-46a6-4e2e-87e8-2f0bf29cd1c8')
base_profiling = torch.load('acc-savings/vit_base_patch16_224_0drop_time1layer0_vbase0b9d73a6-c744-481f-b87c-d95d6f6e82fa')
# print(base_profiling[tuple()]['acc'])

# size of 1024,197
layer_0_attn_score = base_profiling[tuple()]['attn'][0].sum(dim=1) 
# layer_0_attn_score = base_profiling[tuple()]['attn'][8][:,4,:]
# print(layer_0_attn_score[0].shape)


attn_sum_ = torch.Tensor().to(device)
loss_ = torch.Tensor().to(device)
ans_ = torch.Tensor().to(device)
# np.poly1d(np.polyfit())

# print(drop_on_layer0_files)

for acc_file in drop_on_layer0_files:
    profiling = torch.load(acc_file_dir+acc_file)
    for key, content in profiling.items():
        if 0 in key:
            pass
        else:
            attn = layer_0_attn_score[:,key].sum(dim=1)[0:].to(device)
            loss = content['loss'][0:].to(device)
            output = content['output'][0:].to(device)
            groundtruth = content['ground_truth'][0:].to(device)

            attn_sum_ = torch.cat((attn_sum_, attn))
            loss_ = torch.cat((loss_, loss))
            ans_ = torch.cat((ans_, output==groundtruth))
            # print(attn_sum_.shape)
            # print(loss.shape)

# print(np.poly1d(np.polyfit(attn_sum_.to(torch.device('cpu')),loss_.to(torch.device('cpu')),1)))

# import matplotlib.pyplot as plt
# plt.scatter(attn_sum_.to(torch.device('cpu')), ans_.to(torch.device('cpu')), alpha=0.6)
# plt.show()


