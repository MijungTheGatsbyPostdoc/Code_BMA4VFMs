# to be able to load all features of candidate models from imagenet OOD datasets


import torch
import os

# data_name = 'imagenet'
data_name = 'imagenet_sketch'

device = 'cuda:0'
home_dir = os.getcwd()
savedir = home_dir +  f"/feature_representations/all_models_for_{data_name}"
os.makedirs(savedir, exist_ok=True)


# Load pre-processed data
## to load them
print('pre-processed data loading')
# model_names = ['ViT-H-14-378-quickgelu','ViT-bigG-14-CLIPA-336','ViT-SO400M-14-SigLIP-384','EVA02-E-14','ViT-H-14-quickgelu']
# datasets_pretrained_with = ['dfn5b','datacomp1b','webli','laion2b_s4b_b115k','metaclip_fullcc']

# model_names = ['ViT-H-14-378-quickgelu','EVA02-E-14-plus', 'ViT-bigG-14-CLIPA-336','ViT-SO400M-14-SigLIP-384','EVA02-E-14','ViT-H-14-quickgelu', 'convnext_xxlarge']
# datasets_pretrained_with = ['dfn5b','laion2b_s9b_b144k', 'datacomp1b','webli','laion2b_s4b_b115k','metaclip_fullcc', 'laion2b_s34b_b82k_augreg_soup']

model_names = ['ViT-H-14-quickgelu']
datasets_pretrained_with = ['dfn5b']

numb_candidates = len(model_names)
label_across_models = []

for i in range(numb_candidates):
    # print(f'{i}th model')
    # i=4
    print(f'{model_names[i]} model')
    model_name = model_names[i]
    print(model_name)
    pretrained_with = datasets_pretrained_with[i]

    data_dir = f"feature_representations/prepro_dataset={data_name}_with_model={model_name}_pretrained_with_{pretrained_with}"


    feat_mat = torch.load(
        data_dir + f"/feat_data={data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
    label_mat = torch.load(
        data_dir + f"/labels_data={data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")



    if data_name == 'imagenet_v2' \
            or data_name=='imagenet_r' or data_name=='imagenet_a' or data_name=='imagenet_sketch' or data_name =='objectnet':
        feat_train = torch.concatenate(feat_mat[:-1])  # [(len(feat_mat) - 1 ) x batch_size] by feature_dim
        feat_train = torch.concatenate((feat_train, feat_mat[-1]), dim=0)
        label_train = torch.concatenate((label_mat[:-1]))
        label_train = torch.concatenate((label_train, label_mat[-1]), dim=0)
    else:
        feat_train = feat_mat
        label_train = torch.tensor(label_mat)

    label_across_models.append(label_train)

    torch.save(feat_train,
               savedir + f"/feat_data={data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
    torch.save(label_train,
               savedir + f"/labels_data={data_name}.pt")

    del feat_train


    print(f"pre-processed test or validation data of {i}th model is loaded.")


###### now check if the labels are all mixed up or not #####
# labels0 = label_across_models[0].to(device)
# labels1 = label_across_models[1].to(device)
# labels2 = label_across_models[2].to(device)
# labels3 = label_across_models[3].to(device)
# labels4 = label_across_models[4].to(device)
# labels5 = label_across_models[5].to(device)
# labels6 = label_across_models[6].to(device)
#
# print(f'difference beetween label0 and label1: {torch.sum((labels0-labels1)**2)}')
# print(f'difference beetween label0 and label2: {torch.sum((labels0-labels2)**2)}')
# print(f'difference beetween label0 and label3: {torch.sum((labels0-labels3)**2)}')
# print(f'difference beetween label0 and label4: {torch.sum((labels0-labels4)**2)}')
# print(f'difference beetween label0 and label5: {torch.sum((labels0-labels5)**2)}')
# print(f'difference beetween label0 and label6: {torch.sum((labels0-labels6)**2)}')


