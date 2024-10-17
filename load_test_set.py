import torch
import os
def load_test_set(data_name, which_model_to_use, device):

    ########### load pre-processed data (feature extracted data) ###########
    model_names_tot = ['ViT-H-14-378-quickgelu', 'ViT-H-14-quickgelu', 'EVA02-E-14-plus',
                       'ViT-SO400M-14-SigLIP-384',
                       'ViT-bigG-14-CLIPA-336', 'EVA02-E-14', 'ViT-H-14-quickgelu', 'convnext_xxlarge']
    datasets_pretrained_with_tot = ['dfn5b', 'dfn5b', 'laion2b_s9b_b144k', 'webli',
                                    'datacomp1b', 'laion2b_s4b_b115k', 'metaclip_fullcc',
                                    'laion2b_s34b_b82k_augreg_soup']

    dim_tot_tot = [1024, 1024, 1024, 1152, 1280, 1024, 1024, 1024]

    indices_for_selected_models = [int(i) for i in which_model_to_use.split(',')]  # ['1','2','3','4']

    model_names = [model_names_tot[i] for i in indices_for_selected_models]
    datasets_pretrained_with = [datasets_pretrained_with_tot[i] for i in indices_for_selected_models]
    dim_tot = [dim_tot_tot[i] for i in indices_for_selected_models]

    numb_candidates = len(model_names)
    tot_feat_dim = sum(dim_tot[0:numb_candidates])
    print(f'total feature dimension of selected models is {tot_feat_dim}')

    #### load the validation / test set ####
    feat_dir = os.getcwd()
    feat_dir = feat_dir + f"/feature_representations/all_models_for_{data_name}"

    model_name = model_names[0]
    pretrained_with = datasets_pretrained_with[0]
    if data_name == 'imagenet':

        feat_test = torch.load(feat_dir + f"/feat_val_data={data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt", map_location=torch.device(device))
        label_test = torch.load(feat_dir + f"/labels_val_data={data_name}.pt", map_location=torch.device(device))

    elif data_name == 'imagenet_v2' or data_name == 'imagenet_r' or data_name == 'imagenet_a' or data_name == 'imagenet_sketch' \
 \
            or data_name == 'objectnet':

        feat_test = torch.load(feat_dir + f"/feat_data={data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt", map_location=torch.device(device))
        label_test = torch.load(feat_dir + f"/labels_data={data_name}.pt", map_location=torch.device(device))

    return feat_test, label_test
