import torch
import numpy as np
import pathlib
import sys
import timm
from ViT import cola_utils
from ViT.profiling import vision_transformer2

'''
    This file is used to see the acc after we mask a big rectangle
'''

file_dir = str(pathlib.Path(__file__).parent.resolve())

device = torch.device('cuda')

model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device).eval()

test_dataset = cola_utils.ILSVRC2012_val_dataset()

def origin_acc():

    small_dataset = cola_utils.Small_Dataset(test_dataset,-1,100,2345)
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


patch_order = "nothing"
def drop_hook(module, input):
    global patch_order
    return torch.cat([input[0][:,0,:].unsqueeze(dim=1), input[0][:,patch_order]],dim=1)


import PIL
from torchvision import transforms


def mask_token(input_tensor, mask_list):
    for mask in mask_list:
        y,x = mask
        input_tensor[:,y*16:y*16+16, x*16:x*16+16] = 0
    return input_tensor


def visualization(x_start=3, x_end=7, y_start=0, y_end=13, picture_index=6, sample_num=100):
    small_dataset = cola_utils.Small_Dataset(test_dataset,-1, sample_num ,1234)
    pic_tensor, target = small_dataset[picture_index]
    pic_tensor = mask_token(pic_tensor, [(j,i) for i in range(y_start,y_end) for j in range(x_start,x_end)])

    img = transforms.ToPILImage()(pic_tensor).convert('RGB')
    img.show()

    global patch_order
    patch_order = [i for i in range(1,197)]
    for i in range(y_start, y_end):
        for j in range(x_start, x_end):
            patch_order.remove(1+j*14+i)

    handle = model.blocks[0].register_forward_pre_hook(drop_hook)
    output = model(pic_tensor.unsqueeze(dim=0).to(device))
    handle.remove()
    ans = output.data.max(1,keepdim=True)[1]
    if_correct = int(ans)==target
 
    print(if_correct)
    


if __name__ == '__main__':
    visualization()