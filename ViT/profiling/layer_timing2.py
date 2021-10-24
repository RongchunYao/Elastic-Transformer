import torch
import torchvision
import numpy as np
import pathlib
import utils
import sys
sys.path = sys.path[1:]
import timm
sys.path = [''] + sys.path
# to get the attention
import vision_transformer2 

file_dir = str(pathlib.Path(__file__).parent.resolve())+'/'

device = torch.device('cuda')

model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device).eval()

img_size = 224
normalize = torchvision.transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
test_dataset = torchvision.datasets.ImageFolder(file_dir + '../ILSVRC2012',
                torchvision.transforms.Compose([
                    torchvision.transforms.Resize(int(img_size*8/7)),
                    torchvision.transforms.CenterCrop(img_size),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ]))        


def origin_acc():

    small_dataset = utils.Small_Dataset(test_dataset,-1,100,2345)
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
    return torch.cat([input[0][:,patch_order],input[0][:,-1,:].unsqueeze(dim=1)],dim=1)


import PIL
from torchvision import transforms


def mask_token(input_tensor, mask_list):
    for mask in mask_list:
        y,x = mask
        input_tensor[:,y*16:y*16+16, x*16:x*16+16] = 0
    return input_tensor


def visualization():
    small_dataset = utils.Small_Dataset(test_dataset,-1,100,1234)
    pic_tensor, target = small_dataset[6]
    pic_tensor = mask_token(pic_tensor, [(j,i) for i in range(3,7) for j in range(0,13)])


    img = transforms.ToPILImage()(pic_tensor).convert('RGB')
    img.show()

    global patch_order
    patch_order = [i for i in range(1,197)]
    for i in range(0,13):
        for j in range(3,7):
            patch_order.remove(1+j*14+i)

    handle = model.blocks[0].register_forward_pre_hook(drop_hook)
    output = model(pic_tensor.unsqueeze(dim=0).to(device))
    handle.remove()
    ans = output.data.max(1,keepdim=True)[1]
    if_correct = int(ans)==target
 
    print(if_correct)
    


if __name__ == '__main__':
    visualization()