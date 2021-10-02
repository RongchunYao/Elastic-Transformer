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