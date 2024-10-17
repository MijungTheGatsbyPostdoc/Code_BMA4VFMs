# load all the zeroshot weights and individually trained mle (map) weights
# here to test  on validation set of imagenet-1k

import argparse
import matplotlib.pyplot as plt
from training_with_processed_data import accuracy
from torch.nn import Linear
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from BMA_processed_data import linear_model
import torch.nn as nn
import time
from torch.nn import Linear
from load_val_set import load_val_set
import matplotlib.pyplot as plt
import csv
import fnmatch

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='imagenet', help='target dataset: imagenet, imagenet_r, imagenet_a, imagenet_sketch, imagenet_v2, objectnet')
    parser.add_argument('--device', type=str, default='cpu', help='gpu or cpu')
    parser.add_argument('--which-model-to-use', type=str, default='0,2,3,4,5', help='Any combination among 0,1,2,3,4,5,6,7')
    # parser.add_argument('--which-rate-to-use', type=str, default='1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1',
    #                     help='Any combination among 1.0, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01')
    parser.add_argument('--which-epoch', type=int, default=30)
    parser.add_argument('--batch-size', '-bs', type=int, default=1000, help='batch size during validation')
    parser.add_argument('--normalize-features', action='store_true', default=True,
                        help='normalize all features of all models')
    parser.add_argument('--normalize-weights', action='store_true', default=False,
                        help='normalize weights of all models')

    ar = parser.parse_args()
    return ar

def main(**kwargs):

    ar = get_args()
    print(ar)
    device = ar.device

    # based on imagenet, which works probably well on imagenet and imagenet-v2
    # post_w_mle = torch.tensor([0.001611, 0.001605, 0.001478, 0.001633, 0.001822, 0.001536, 0.001566, 0.001535])
    # post_w_zeroshot = torch.tensor([0.001497, 0.001492, 0.001325, 0.0014, 0.001571, 0.001289, 0.001381, 0.001429])

    ### this is the weights computed on the imagenet-1k test data
    post_w_mle = torch.tensor([0.00143, 0.001424, 0.001434, 0.001387, 0.001422, 0.001507, 0.001366, 0.001439])
    post_w_zeroshot = torch.tensor([0.001253, 0.0012434, 0.00124, 0.001178, 0.001197, 0.001242, 0.001166, 0.00124])

    model_names_tot = ['ViT-H-14-378-quickgelu', 'ViT-H-14-quickgelu', 'EVA02-E-14-plus', 'ViT-SO400M-14-SigLIP-384',
                       'ViT-bigG-14-CLIPA-336',  'EVA02-E-14', 'ViT-H-14-quickgelu', 'convnext_xxlarge']
    datasets_pretrained_with_tot = ['dfn5b', 'dfn5b', 'laion2b_s9b_b144k', 'webli',
                                    'datacomp1b', 'laion2b_s4b_b115k','metaclip_fullcc', 'laion2b_s34b_b82k_augreg_soup']

    dim_tot_tot = [1024, 1024, 1024, 1152, 1280, 1024, 1024, 1024]

    indices_for_selected_models = [int(i) for i in ar.which_model_to_use.split(',')] # ['1','2','3','4']

    model_names = [model_names_tot[i] for i in indices_for_selected_models]
    datasets_pretrained_with = [datasets_pretrained_with_tot[i] for i in indices_for_selected_models]
    dim_tot = [dim_tot_tot[i] for i in indices_for_selected_models]

    numb_candidates = len(model_names)
    tot_feat_dim = sum(dim_tot[0:numb_candidates])
    print(f'total feature dimension of selected models is {tot_feat_dim}')


    if ar.data_name == 'imagenet' or ar.data_name=='imagenet_v2' or ar.data_name=='imagenet_sketch' or ar.data_name =='objectnet':
        numb_classes = 1000
    elif ar.data_name=='imagenet_a' or ar.data_name=='imagenet_r':
        numb_classes = 200
    else:
        print('oops, you need to specify numb_classes')

    # eval data directory
    feat_dir = os.getcwd()
    feat_dir = feat_dir + f"/feature_representations/all_models_for_{ar.data_name}"

    # store zeroshot weights or trained weights
    if ar.data_name == 'imagenet':
        tot_test = 50000
    elif ar.data_name == 'imagenet_v2':
        tot_test = 10000
    elif ar.data_name == 'imagenet_a':
        tot_test = 7500
    elif ar.data_name == 'imagenet_r':
        tot_test = 30000
    elif ar.data_name == 'imagenet_sketch':
        tot_test = 50889
    elif ar.data_name == 'objectnet':
        tot_test = 18574

    feat_test_tot = torch.zeros(tot_test, tot_feat_dim)

    zeroshot_weights_tot = torch.zeros(numb_classes, tot_feat_dim)
    mle_weights_tot = torch.zeros(numb_classes, tot_feat_dim)

    mle_pw_tot = torch.zeros(numb_candidates)
    zeroshot_pw_tot = torch.zeros(numb_candidates)

    for i in range(numb_candidates):
        model_name = model_names[i]
        pretrained_with = datasets_pretrained_with[i]

        mle_pw_tot[i] = post_w_mle[indices_for_selected_models[i]]
        zeroshot_pw_tot[i] = post_w_zeroshot[indices_for_selected_models[i]]

        #### load the pretrained and zershot models to test ####
        ### first, load zeroshot models ###
        if ar.data_name == 'imagenet' or ar.data_name == 'imagenet_v2' or ar.data_name == 'imagenet_sketch' or ar.data_name == 'objectnet':
            home_dir = os.getcwd() + f"/imagenet_results"
        else:  #
            home_dir = os.getcwd() + f"/{ar.data_name}_results"

        if ar.data_name == 'imagenet' or ar.data_name == 'imagenet_v2' or ar.data_name == 'imagenet_sketch' or ar.data_name == 'objectnet':
                zeroshot_weights = torch.load(
                    home_dir + f"/zeroshot_weights/zero_shot_weights_imagenet_{model_name}_{pretrained_with}.pt", map_location=torch.device(device)) # feat_dim by numb_classes
        else: #ar.data_name == 'imagenet_a' and ar.data_name == 'imagenet_r'
            zeroshot_weights = torch.load(
                home_dir + f"/zeroshot_weights/zero_shot_weights_{ar.data_name}_{model_name}_{pretrained_with}.pt", map_location=torch.device(device))  # feat_dim by numb_classes

        zeroshot_weights = zeroshot_weights.t()


        ### second, load mle ###
        if ar.data_name == 'imagenet' or ar.data_name == 'imagenet_v2' or ar.data_name == 'imagenet_r' \
                or ar.data_name == 'imagenet_a' or ar.data_name == 'imagenet_sketch' or ar.data_name == 'objectnet':
            model_dir = os.getcwd() + f"/imagenet_results"
            # filename = fnmatch.filter(os.listdir(model_dir + f"/[{indices_for_selected_models[i]}]_model_mle"), 'At_*.pt')
            # mle = torch.load(model_dir + f"/[{indices_for_selected_models[i]}]_model_mle/"+filename[0], map_location=torch.device(device))

            if ar.data_name == 'objectnet':
                sel_rate = 0.1
            elif ar.data_name=='imagenet_a':
                sel_rate = 0.2
            elif ar.data_name =='imagenet_v2':
                sel_rate = 0.8
            elif ar.data_name=='imagenet':
                sel_rate = 1.0
            else:
                sel_rate = 0.9
            filename = fnmatch.filter(os.listdir(model_dir + f"/[{indices_for_selected_models[i]}]_model_map"), f'At_*selrate={sel_rate}*.pt')
            mle = torch.load(model_dir + f"/[{indices_for_selected_models[i]}]_model_map/"+filename[0],map_location=torch.device(device))


        mle_weights = mle['weight']

        if ar.data_name == 'imagenet_a':
            from imagenet_a import CLASS_SUBLIST
            mle_weights = mle_weights[CLASS_SUBLIST, :]
        elif ar.data_name == 'imagenet_r':
            from imagenet_r import CLASS_SUBLIST
            mle_weights = mle_weights[CLASS_SUBLIST, :]


        #### load the validation / test set ####
        if ar.data_name == 'imagenet':
            feat_test = torch.load(
                feat_dir + f"/feat_val_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt", map_location=torch.device(device))
            label_test = torch.load(
                feat_dir + f"/labels_val_data={ar.data_name}.pt", map_location=torch.device(device))

        elif ar.data_name == 'imagenet_v2' or ar.data_name=='imagenet_r' or ar.data_name=='imagenet_a' or ar.data_name=='imagenet_sketch' \
                or ar.data_name =='objectnet':
            feat_test = torch.load(
                feat_dir + f"/feat_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt", map_location=torch.device(device))
            label_test = torch.load(
                feat_dir + f"/labels_data={ar.data_name}.pt", map_location=torch.device(device))

        if ar.normalize_weights:
            zeroshot_weights = zeroshot_weights / torch.norm(zeroshot_weights, dim=1,
                                                             keepdim=True)  # classes by feature dim
            mle_weights = mle_weights / torch.norm(mle_weights, dim=1,
                                                   keepdim=True)


        if i == 0:
            zeroshot_weights_tot[:, 0:dim_tot[0]] = zeroshot_weights # classes by features
            mle_weights_tot[:, 0:dim_tot[0]] = mle_weights
            feat_test_tot[:, 0:dim_tot[0]] = feat_test
            print(f'{i + 1}st models zeroshot and mle weights loaded')
            print(f'index from 0 to {dim_tot[0]}')
        else:
            zeroshot_weights_tot[:, sum(dim_tot[0:i]):sum(dim_tot[0:(i + 1)])] = zeroshot_weights
            mle_weights_tot[:, sum(dim_tot[0:i]):sum(dim_tot[0:(i + 1)])] = mle_weights
            feat_test_tot[:, sum(dim_tot[0:i]):sum(dim_tot[0:(i + 1)])] = feat_test
            print(f'{i + 1}rd models zeroshot and mle weights loaded')
            print(f'index from {sum(dim_tot[0:i])} to {sum(dim_tot[0:(i + 1)])}')

    ### normalize the posterior model weights
    pw_tot = torch.cat((mle_pw_tot, zeroshot_pw_tot),0)
    mle_pw_tot = mle_pw_tot/sum(pw_tot)
    zeroshot_pw_tot = zeroshot_pw_tot/sum(pw_tot)

    print(f'normalized posterior model weights for mle models are {mle_pw_tot}')
    print(f'normalized posterior model weights for zeroshot models are {zeroshot_pw_tot}')

    if ar.normalize_features:
        print('we normalize the features of each datapoint')

        ## memory problem if more than 5 models were used.
        # norm = feat_test_tot.norm(p=2, dim=1, keepdim=True)  # tot_test by 1
        # feat_test_tot = feat_test_tot.div(norm)

        for i in range(tot_test):
            l2norm_val_tst = feat_test_tot[i,:].norm(p=2)
            feat_test_tot[i, :] = feat_test_tot[i, :] / l2norm_val_tst
            # print(feat_test_tot[i,:].norm(p=2))

    ################## Now evaluation starts ###################
    # (1) simple concatenation of all chosen models
    # (2) output averaging
    # (3) fisher merging
    # (4) bma

    tot_test = feat_test.shape[0]
    how_many_steps_in_test = int(tot_test / ar.batch_size)

    m = nn.Softmax(dim=1)

    if ar.data_name == 'objectnet':
        data_dir = '../databank/'
        from model_soups.datasets.objectnet import get_metadata, ObjectNetDataset, ObjectNetBase, ObjectNet
        from model_soups.datasets.imagenet_classnames import objectnet_classnames
        dataset = ObjectNet(location=data_dir, preprocess=[], batch_size=10, num_workers=int(1),
                            classnames=objectnet_classnames)
        projection_fn = getattr(dataset, 'project_logits', None)



    test_accs_top1_avg_output = []
    test_accs_top1_bma = []

    tot_num_test_evaluated = 0

    for i in range(how_many_steps_in_test):

        if i == (how_many_steps_in_test - 1):
            feat1 = feat_test_tot[ar.batch_size * i:, :]
            labels = label_test[ar.batch_size * i:]
        else:
            feat1 = feat_test_tot[ar.batch_size * i:ar.batch_size * (i + 1), :]
            labels = label_test[ar.batch_size * i:ar.batch_size * (i + 1)]

        # print(labels)
        tot_num_test_evaluated = tot_num_test_evaluated + feat1.shape[0]

        feat1 = feat1.to(ar.device)
        labels = labels.to(ar.device)

        ### average output probability (ensemble) ###
        if ar.data_name == 'objectnet':
            output_tot = torch.zeros(feat1.shape[0], 113, numb_candidates)
            output_tot_mle = torch.zeros(feat1.shape[0], 113, numb_candidates)
            output_tot_bma = torch.zeros(feat1.shape[0], 113, numb_candidates)
            output_tot_mle_bma = torch.zeros(feat1.shape[0], 113, numb_candidates)
        else:
            output_tot = torch.zeros(feat1.shape[0], numb_classes, numb_candidates)
            output_tot_mle = torch.zeros(feat1.shape[0], numb_classes, numb_candidates)
            output_tot_bma = torch.zeros(feat1.shape[0], numb_classes, numb_candidates)
            output_tot_mle_bma = torch.zeros(feat1.shape[0], numb_classes, numb_candidates)

        for j in range(numb_candidates):

            if j == 0:
                model = zeroshot_weights_tot[:, :dim_tot[0]]
                model_mle = mle_weights_tot[:, :dim_tot[0]]
                feat = feat1[:, :dim_tot[0]]
            elif j == 1:
                model = zeroshot_weights_tot[:, dim_tot[0]:sum(dim_tot[0:2])]
                model_mle = mle_weights_tot[:, dim_tot[0]:sum(dim_tot[0:2])]
                feat = feat1[:, dim_tot[0]:sum(dim_tot[0:2])]
            else:
                model = zeroshot_weights_tot[:, sum(dim_tot[0:j]):sum(dim_tot[0:(j + 1)])]
                model_mle = mle_weights_tot[:, sum(dim_tot[0:j]):sum(dim_tot[0:(j + 1)])]
                feat = feat1[:,sum(dim_tot[0:j]):sum(dim_tot[0:(j + 1)])]

            model = model.to(ar.device)
            model_mle = model_mle.to(ar.device)
            output_model = torch.mm(feat, model.t())
            output_model_mle = torch.mm(feat, model_mle.t())

            if ar.data_name == 'objectnet':
                output_model = projection_fn(output_model.detach(), ar.device)
                output_model_mle = projection_fn(output_model_mle.detach(), ar.device)

            softmax_output = m(output_model)
            output_tot[:, :, j] = softmax_output
            output_tot_bma[:,:,j] = softmax_output * zeroshot_pw_tot[j]

            softmax_output_mle = m(output_model_mle)
            output_tot_mle[:, :, j] = softmax_output_mle
            output_tot_mle_bma[:,:,j] = softmax_output_mle * mle_pw_tot[j]

        ### Just Average Output ###
        output_tot_tot = torch.cat((output_tot, output_tot_mle), 2)
        outputs = torch.mean(output_tot_tot, dim=2).to(ar.device)
        top1_acc = accuracy(outputs, labels, topk=(1,))
        test_accs_top1_avg_output.append(top1_acc[0])

        ### BMA ###
        output_tot_tot_bma = torch.cat((output_tot_bma, output_tot_mle_bma), 2)
        outputs_bma = torch.sum(output_tot_tot_bma, dim=2)
        top1_acc_bma = accuracy(outputs_bma, labels, topk=(1,))
        test_accs_top1_bma.append(top1_acc_bma[0])

    # print(f'combined zeroshot top1 acc on {ar.data_name}: {np.mean(test_accs_top1_zeroshot_tot):.3f}')
    print(f'output avg: top1 acc on {ar.data_name}: {np.mean(test_accs_top1_avg_output):.3f}')
    print(f'bma: top1 acc on {ar.data_name}: {np.mean(test_accs_top1_bma):.3f}')
    # print(f'output max top1 acc on {ar.data_name}: {np.mean(test_accs_top1_max_output):.3f}')
    print(f'total number of test points:{tot_num_test_evaluated}')




#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
