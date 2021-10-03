import os
import torch
import timm
import torchvision
import pathlib

file_dir = str(pathlib.Path(__file__).parent.resolve()) + '/'

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
device = torch.device('cpu')

model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)

token_num_g = 196

img_size = 224
normalize = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
test_dataset = torchvision.datasets.ImageFolder(
    file_dir + '../ILSVRC2012',
    torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(img_size * 8 / 7)),
        torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.ToTensor(),
        normalize,
    ]))


def drop_hook(module, input):
    global token_num_g
    input, a, b = input
    return (torch.cat([input[:, :token_num_g], input[:, -1].unsqueeze(dim=1)], dim=1), a, b)


def print_global():
    global token_num
    print(token_num_g)


def forward_once(model, input_data, token_num_list):
    global token_num_g
    handle = model.blocks[0].register_forward_pre_hook(drop_hook)
    attention_time_list = []
    MLP_time_list = []
    prepare_time_list = []
    encoder_time_list = []
    total_time_list = []
    endphase_time_list = []
    model.eval()
    for token_num in token_num_list:
        token_num_g = token_num
        with torch.no_grad():
            model(input_data)
        attention_time_list.append(model.attention_time_list)
        MLP_time_list.append(model.MLP_time_list)
        prepare_time_list.append(model.prepare_time)
        encoder_time_list.append(model.encoder_time)
        total_time_list.append(model.total_time)
        endphase_time_list.append(model.endphase_time)
    handle.remove()
    return attention_time_list, MLP_time_list, prepare_time_list, encoder_time_list, endphase_time_list, total_time_list


if __name__ == '__main__':
    drop_max = 128
    token_num_list = [196 - i for i in range(drop_max)]
    global_attention_time = torch.zeros(drop_max, 12)
    global_MLP_time = torch.zeros(drop_max, 12)
    global_prepare_time = torch.zeros(drop_max)
    global_encoder_time = torch.zeros(drop_max)
    global_endphase_time = torch.zeros(drop_max)
    global_total_time = torch.zeros(drop_max)

    for i in range(10):
        input_data, _ = test_dataset[i]
        input_data = input_data.unsqueeze(dim=0)
        attention_time_list, MLP_time_list, prepare_time_list, encoder_time_list, endphase_time_list, total_time_list = forward_once(
            model, input_data, token_num_list)
        global_attention_time += torch.Tensor(attention_time_list)
        global_MLP_time += torch.Tensor(MLP_time_list)
        global_prepare_time += torch.Tensor(prepare_time_list)
        global_encoder_time += torch.Tensor(encoder_time_list)
        global_endphase_time += torch.Tensor(endphase_time_list)
        global_total_time += torch.Tensor(total_time_list)

    global_attention_time /= 10
    global_MLP_time /= 10
    global_prepare_time /= 10
    global_encoder_time /= 10
    global_endphase_time /= 10
    global_total_time /= 10

    dict2save = {}
    dict2save['attn'] = global_attention_time
    dict2save['MLP'] = global_MLP_time
    dict2save['prepare'] = global_prepare_time
    dict2save['encoder'] = global_encoder_time
    dict2save['endphase'] = global_endphase_time
    dict2save['total'] = global_total_time

    if not os.path.exists(file_dir + 'profiling_result/'):
        os.makedirs(file_dir + 'profiling_result/')

    torch.save(dict2save, file_dir + 'profiling_result/profiling_result')
    # print(time_list)
