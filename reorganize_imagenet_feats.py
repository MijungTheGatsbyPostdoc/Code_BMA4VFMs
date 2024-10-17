# to be able to load all features of candidate models from imagenet dataset


import torch
import os


data_name = 'imagenet'
# data_name = 'sun397'
# data_name = 'flowers102'

device = 'cuda:0'

home_dir = os.getcwd()
savedir = home_dir +  f"/feature_representations/all_models_for_{data_name}"
os.makedirs(savedir, exist_ok=True)


# Load pre-processed data
## to load them
print('pre-processed data loading')
# model_names = ['ViT-H-14-378-quickgelu','ViT-bigG-14-CLIPA-336','ViT-SO400M-14-SigLIP-384','EVA02-E-14','ViT-H-14-quickgelu', 'ViT-L-14-CLIPA-336']
# datasets_pretrained_with = ['dfn5b','datacomp1b','webli','laion2b_s4b_b115k','metaclip_fullcc', 'datacomp1b']

# model_names = ['ViT-H-14-378-quickgelu','ViT-bigG-14-CLIPA-336','ViT-SO400M-14-SigLIP-384','EVA02-E-14','ViT-H-14-quickgelu']
# datasets_pretrained_with = ['dfn5b','datacomp1b','webli','laion2b_s4b_b115k','metaclip_fullcc']


# model_names = ['ViT-H-14-378-quickgelu','EVA02-E-14-plus', 'ViT-bigG-14-CLIPA-336','ViT-SO400M-14-SigLIP-384','EVA02-E-14','ViT-H-14-quickgelu', 'convnext_xxlarge']
# datasets_pretrained_with = ['dfn5b','laion2b_s9b_b144k', 'datacomp1b','webli','laion2b_s4b_b115k','metaclip_fullcc', 'laion2b_s34b_b82k_augreg_soup']

model_names=['EVA02-E-14-plus']
datasets_pretrained_with = ['laion2b_s9b_b144k']

### for imagenet dataset: when convnet extraction is over.
# model_names = ['ViT-bigG-14-CLIPA-336', 'convnext_xxlarge']
# datasets_pretrained_with = ['datacomp1b', 'laion2b_s34b_b82k_augreg_soup']

# model_names= ['ViT-H-14-378-quickgelu', 'ViT-H-14-quickgelu', 'ViT-bigG-14-CLIPA-336', 'EVA02-E-14', 'ViT-H-14-quickgelu']
# datasets_pretrained_with = ['dfn5b', 'dfn5b','datacomp1b', 'laion2b_s4b_b115k', 'metaclip_fullcc']

# model_names= ['ViT-H-14-378-quickgelu', 'ViT-H-14-quickgelu', 'EVA02-E-14', 'ViT-H-14-quickgelu']
# datasets_pretrained_with = ['dfn5b', 'dfn5b', 'laion2b_s4b_b115k', 'metaclip_fullcc']

# python preprocess.py --data-name 'flowers102' --model-name 'ViT-SO400M-14-SigLIP-384' --pretrained-with 'webli'
# python preprocess.py --data-name 'flowers102' --model-name 'EVA02-E-14' --pretrained-with 'laion2b_s4b_b115k'
# python preprocess.py --data-name 'flowers102' --model-name 'ViT-H-14-quickgelu' --pretrained-with 'metaclip_fullcc'

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
    try:
        label_mat = torch.load(
            data_dir + f"/labels_data={data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
        no_label_file = False
    except:
        print('there is no label mat file')
        no_label_file = True

    if data_name == 'imagenet':
        feat_train = torch.concatenate(feat_mat[:-1])  # [(len(feat_mat) - 1 ) x batch_size] by feature_dim
        feat_train = torch.concatenate((feat_train, feat_mat[-1]), dim=0) # because the last batch has a different number of datapoints
        if not no_label_file:
            label_train = torch.concatenate((label_mat[:-1]))
            label_train = torch.concatenate((label_train, label_mat[-1]), dim=0)
    else:
        feat_train = feat_mat
        label_train = torch.tensor(label_mat)

    # sorted_labels, indices = torch.sort(label_train)
    # del label_train, feat_mat, label_mat
    # feat_train = feat_train[indices, :]
    # label_train = sorted_labels
    # print(f'number of feat_train samples is', feat_train.shape[0])
    # print(f'feat dim is', feat_train.shape[1])

    # tot_train = label_train.shape[0]
    # print(f"pre-processed training data of {i}th model is loaded.")

    # feat_mat_val = torch.load(
    #     data_dir + f"/feat_val_data={data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
    # label_mat_val = torch.load(
    #     data_dir + f"/labels_val_data={data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
    # if data_name == 'imagenet':
    #     feat_test = torch.concatenate(feat_mat_val[:-1])  # [(len(feat_mat) - 1 ) x batch_size] by feature_dim
    #     feat_test = torch.concatenate((feat_test, feat_mat_val[-1]), dim=0)
    #     label_test = torch.concatenate((label_mat_val[:-1]))
    #     label_test = torch.concatenate((label_test, label_mat_val[-1]), dim=0)
    # else:
    #     feat_test = feat_mat_val
    #     label_test = torch.tensor(label_mat_val)

    # sorted_labels, indices = torch.sort(label_test)
    # feat_test = feat_test[indices, :]
    # label_test = sorted_labels
    # print(f'number of feat_test samples is', feat_test.shape[0])
    if not no_label_file:
        label_across_models.append(label_train)

    torch.save(feat_train,
               savedir + f"/feat_data={data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
    if not no_label_file:
        torch.save(label_train,
                   savedir + f"/labels_data={data_name}.pt")
        print('saving labels')

    # torch.save(feat_test,
    #            savedir + f"/feat_val_data={data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
    # torch.save(label_test,
    #            savedir + f"/labels_val_data={data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")

    del feat_train
    # del feat_mat_val, label_mat_val


    print(f"pre-processed test or validation data of {i}th model is loaded.")


###### now check if the labels are all mixed up or not #####
# labels0 = label_across_models[0].to(device)
# labels1 = label_across_models[1].to(device)
# labels2 = label_across_models[2].to(device)
# labels3 = label_across_models[3].to(device)
# labels4 = label_across_models[4].to(device)
# labels5 = label_across_models[5].to(device)
# labels6 = label_across_models[6].to(device)

# print(f'difference beetween label0 and label1: {torch.sum((labels0-labels1)**2)}')
# print(f'difference beetween label0 and label2: {torch.sum((labels0-labels2)**2)}')
# print(f'difference beetween label0 and label3: {torch.sum((labels0-labels3)**2)}')
# print(f'difference beetween label0 and label4: {torch.sum((labels0-labels4)**2)}')
# print(f'difference beetween label0 and label5: {torch.sum((labels0-labels5)**2)}')
# print(f'difference beetween label0 and label6: {torch.sum((labels0-labels6)**2)}')

