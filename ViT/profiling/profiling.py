import torch
import timm
from ViT.profiling import vision_transformer
import pathlib
import numpy as np
from ViT import cola_utils
import argparse

file_dir = str(pathlib.Path(__file__).parent.resolve())
project_root_dir = str(pathlib.PurePath(file_dir, '..', '..'))

token_num_g = 'undefined'

def drop_hook(module, input):
    global token_num_g
    input, a, b = input
    return (torch.cat([input[:,0:1], input[:,1:token_num_g+1] ],dim=1), a, b)

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
    return attention_time_list, MLP_time_list, prepare_time_list, encoder_time_list,  endphase_time_list, total_time_list


def do_profiling(forward_times = 10, saving_name = 'profiling_result'):

    drop_max = 180
    none_cls_token_num = 196
    image_size = 224
    layer_num = 12
    in_channel = 3
    batch = 1
    device_type = 'cpu'
    model_name = 'vit_base_patch16_224'
    saving_dir = str(pathlib.PurePath(file_dir, 'profiling_result'))
    abs_saving_path = str(pathlib.PurePath(saving_dir, saving_name))


    pathlib.Path(saving_dir).mkdir(parents=True, exist_ok=True)
    cola_utils.set_cpu_resource()
    device = torch.device(device_type)
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.eval()
    token_num_list = [ none_cls_token_num-i for i in range(drop_max)]

    dict2save = {}
    dict2save['attn'] = torch.zeros(drop_max, layer_num)
    dict2save['MLP'] = torch.zeros(drop_max, layer_num)
    dict2save['prepare'] = torch.zeros(drop_max)
    dict2save['encoder'] = torch.zeros(drop_max)
    dict2save['endphase'] = torch.zeros(drop_max)
    dict2save['total'] = torch.zeros(drop_max)
    
    for i in range(forward_times):
        print('iteration {:d}/{:d}'.format(i+1,forward_times))
        input_data = torch.randn(batch,in_channel,image_size,image_size)
        attention_time_list, MLP_time_list, prepare_time_list, encoder_time_list,  endphase_time_list, total_time_list = forward_once(model, input_data, token_num_list)
        dict2save['attn'] += torch.Tensor(attention_time_list)
        dict2save['MLP'] += torch.Tensor(MLP_time_list)
        dict2save['prepare'] += torch.Tensor(prepare_time_list)
        dict2save['encoder'] += torch.Tensor(encoder_time_list)
        dict2save['endphase'] += torch.Tensor(endphase_time_list)
        dict2save['total'] += torch.Tensor(total_time_list)
    
    for key in ['attn', 'MLP', 'prepare', 'encoder', 'endphase', 'total']:
        dict2save[key] /= forward_times

    torch.save(dict2save, abs_saving_path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--forward-times', '-forward', type=int, default=10, help='specify how many times do we run the time \
        profiling, we average the results as the final output, default value: 10')

    parser.add_argument('--output', '-output', type=str, default='profiling_result', help='specify the output filename, default value: profiling_result')
    args = parser.parse_args()

    do_profiling(forward_times= args.forward_times, saving_name=args.output)