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

def bring_models_and_val_features(model_names, datasets_pretrained_with, dim_tot, numb_candidates, numb_classes, data_name, method_name, models_for, normalize_models, normalize_features, device):
    # eval data directory
    feat_dir = os.getcwd()
    feat_dir = feat_dir + f"/feature_representations/all_models_for_{data_name}"

    # store zeroshot weights or trained weights
    tot_feat_dim = sum(dim_tot)
    weights_tot = torch.zeros(numb_classes, tot_feat_dim)

    if data_name == 'imagenet':
        tot_test = 50000
    elif data_name == 'imagenet_v2':
        tot_test = 10000
    elif data_name == 'imagenet_a':
        tot_test = 7500
    elif data_name == 'imagenet_r':
        tot_test = 30000
    elif data_name == 'imagenet_sketch':
        tot_test = 50889
    elif data_name == 'objectnet':
        tot_test = 18574

    feat_test_tot = torch.zeros(tot_test, tot_feat_dim)

    for i in range(numb_candidates):

        model_name = model_names[i]
        pretrained_with = datasets_pretrained_with[i]

        #### load the pretrained or zershot models to test ####
        if method_name == 'zeroshot':
            if data_name == 'imagenet' or data_name == 'imagenet_v2' or data_name == 'imagenet_sketch' or data_name == 'objectnet':
                home_dir = os.getcwd() + f"/imagenet_results"
                zeroshot_weights = torch.load(
                    home_dir + f"/zeroshot_weights/zero_shot_weights_imagenet_{model_name}_{pretrained_with}.pt", map_location=torch.device(device))  # feat_dim by numb_classes
            elif data_name == 'imagenet_a' or data_name == 'imagenet_r':
                home_dir = os.getcwd() + f"/{data_name}_results"
                zeroshot_weights = torch.load(
                    home_dir + f"/zeroshot_weights/zero_shot_weights_{data_name}_{model_name}_{pretrained_with}.pt", map_location=torch.device(device))  # feat_dim by numb_classes

        elif method_name == 'mle':

            model_dir = os.getcwd() + f"/imagenet_results"
            filename = fnmatch.filter(os.listdir(model_dir + f"/[{models_for[i]}]_model_mle"), 'At_*.pt')
            mle = torch.load(model_dir + f"/[{models_for[i]}]_model_mle/" + filename[0], map_location=torch.device(device))
            zeroshot_weights = mle['weight']

            if data_name == 'imagenet_a':
                from imagenet_a import CLASS_SUBLIST
                zeroshot_weights = zeroshot_weights[CLASS_SUBLIST, :]
            elif data_name == 'imagenet_r':
                from imagenet_r import CLASS_SUBLIST
                zeroshot_weights = zeroshot_weights[CLASS_SUBLIST, :]

        #### load the validation / test set ####
        if data_name == 'imagenet':
            feat_test = torch.load(
                feat_dir + f"/feat_val_data={data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt", map_location=torch.device(device))
            label_test = torch.load(
                feat_dir + f"/labels_val_data={data_name}.pt", map_location=torch.device(device))

        elif data_name == 'imagenet_v2' or data_name == 'imagenet_r' or data_name == 'imagenet_a' or data_name == 'imagenet_sketch' \
                or data_name == 'objectnet':
            feat_test = torch.load(
                feat_dir + f"/feat_data={data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt", map_location=torch.device(device))
            label_test = torch.load(
                feat_dir + f"/labels_data={data_name}.pt", map_location=torch.device(device))

        if i == 0:
            if method_name == 'zeroshot':
                if normalize_models==1:
                    weights_tot[:, 0:dim_tot[0]] = zeroshot_weights.t()/torch.norm(zeroshot_weights)
                else:
                    weights_tot[:, 0:dim_tot[0]] = zeroshot_weights.t()
            else:
                if normalize_models==1:
                    weights_tot[:, 0:dim_tot[0]] = zeroshot_weights/torch.norm(zeroshot_weights)
                else:
                    weights_tot[:, 0:dim_tot[0]] = zeroshot_weights

            if normalize_features:
                feat_test_tot[:, 0:dim_tot[0]] = feat_test/torch.norm(feat_test)
            else:
                feat_test_tot[:, 0:dim_tot[0]] = feat_test
            print(f'{i + 1}st models zeroshot weights loaded')
            print(f'index from 0 to {dim_tot[0]}')
        else:
            if method_name == 'zeroshot':
                if normalize_models==1:
                    weights_tot[:, sum(dim_tot[0:i]):sum(dim_tot[0:(i + 1)])] = zeroshot_weights.t()/torch.norm(zeroshot_weights)
                else:
                    weights_tot[:, sum(dim_tot[0:i]):sum(dim_tot[0:(i + 1)])] = zeroshot_weights.t()
            else:
                if normalize_models==1:
                    weights_tot[:, sum(dim_tot[0:i]):sum(dim_tot[0:(i + 1)])] = zeroshot_weights/torch.norm(zeroshot_weights)
                else:
                    weights_tot[:, sum(dim_tot[0:i]):sum(dim_tot[0:(i + 1)])] = zeroshot_weights

            if normalize_features:
                feat_test_tot[:, sum(dim_tot[0:i]):sum(dim_tot[0:(i + 1)])] = feat_test/torch.norm(feat_test)
            else:
                feat_test_tot[:, sum(dim_tot[0:i]):sum(dim_tot[0:(i + 1)])] = feat_test
            print(f'{i + 1}rd models zeroshot weights loaded')
            print(f'index from {sum(dim_tot[0:i])} to {sum(dim_tot[0:(i + 1)])}')


    return weights_tot, feat_test_tot, label_test, tot_test

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='imagenet', help='target dataset: imagenet, imagenet_r, imagenet_a, imagenet_sketch, imagenet_v2, objectnet')
    parser.add_argument('--device', type=str, default='cpu', help='gpu or cpu')
    # parser.add_argument('--method-model-comb', type=str, default=['zeroshot','2,3,4', 'mle','0,2,3,4'], help='zeroshot or mle or bma (in this order), and model combinations')
    parser.add_argument('--zeroshot-models', type=str, default='4',
                        help='any combinations among 0,1,2,3,4,5,6,7')
    parser.add_argument('--mle-models', type=str, default='0,2,3,4',
                        help='any combinations among 0,1,2,3,4,5,6,7')

    parser.add_argument('--which-epoch', type=int, default=30)
    parser.add_argument('--batch-size', '-bs', type=int, default=1000, help='batch size during validation')
    parser.add_argument('--normalize-features', action='store_true', default=True,
                        help='normalize all features of all models')
    parser.add_argument('--normalize-models', type=int, default=0,
                        help='normalize all models so they are comparable in magnitude')

    ar = parser.parse_args()
    return ar

def main(**kwargs):

    ar = get_args()
    print(ar)

    model_names_tot = ['ViT-H-14-378-quickgelu', 'ViT-H-14-quickgelu', 'EVA02-E-14-plus', 'ViT-SO400M-14-SigLIP-384',
                       'ViT-bigG-14-CLIPA-336',  'EVA02-E-14', 'ViT-H-14-quickgelu', 'convnext_xxlarge']
    datasets_pretrained_with_tot = ['dfn5b', 'dfn5b', 'laion2b_s9b_b144k', 'webli',
                                    'datacomp1b', 'laion2b_s4b_b115k','metaclip_fullcc', 'laion2b_s34b_b82k_augreg_soup']

    dim_tot_tot = [1024, 1024, 1024, 1152, 1280, 1024, 1024, 1024]


    models_for_zeroshot = [int(i) for i in ar.zeroshot_models.split(',')]
    model_names_zeroshot = [model_names_tot[i] for i in models_for_zeroshot]
    datasets_pretrained_with_zeroshot = [datasets_pretrained_with_tot[i] for i in models_for_zeroshot]
    dim_tot_zeroshot = [dim_tot_tot[i] for i in models_for_zeroshot]
    numb_candidates_zeroshot = len(model_names_zeroshot)

    models_for_mle = [int(i) for i in ar.mle_models.split(',')]
    model_names_mle = [model_names_tot[i] for i in models_for_mle]
    datasets_pretrained_with_mle = [datasets_pretrained_with_tot[i] for i in models_for_mle]
    dim_tot_mle = [dim_tot_tot[i] for i in models_for_mle]
    numb_candidates_mle = len(model_names_mle)


    if ar.data_name == 'imagenet' or ar.data_name=='imagenet_v2' or ar.data_name=='imagenet_sketch' or ar.data_name =='objectnet':
        numb_classes = 1000
    elif ar.data_name=='imagenet_a' or ar.data_name=='imagenet_r':
        numb_classes = 200
    else:
        print('oops, you need to specify numb_classes')

    zeroshot_weights_tot, feat_test_tot_zeroshot, label_test, tot_test = bring_models_and_val_features(model_names_zeroshot,
                                                                                                       datasets_pretrained_with_zeroshot,
                                                                                                       dim_tot_zeroshot,
                                                                                                       numb_candidates_zeroshot,
                                                                                                       numb_classes,
                                                                                                       ar.data_name,
                                                                                                       'zeroshot',
                                                                                                       models_for_zeroshot, normalize_models=0, normalize_features = False, device=ar.device)
    mle_weights_tot, feat_test_tot_mle, label_test, tot_test = bring_models_and_val_features(model_names_mle,
                                                                                         datasets_pretrained_with_mle,
                                                                                         dim_tot_mle,
                                                                                         numb_candidates_mle,
                                                                                         numb_classes, ar.data_name,
                                                                                         'mle', models_for_mle, normalize_models=0, normalize_features=False, device=ar.device)


    if ar.normalize_features:
        print('we normalize the features of each datapoint')

        # # memory problem if more than 5 models were used.
        # norm = feat_test_tot.norm(p=2, dim=1, keepdim=True)  # tot_test by 1
        # feat_test_tot = feat_test_tot.div(norm)

        for i in range(tot_test):
            # l2norm_val_tst = torch.concat((feat_test_tot_zeroshot[i,:],feat_test_tot_mle[i,:])).norm(p=2)
            l2norm_val_tst = feat_test_tot_zeroshot[i, :].norm(p=2)
            feat_test_tot_zeroshot[i, :] = feat_test_tot_zeroshot[i, :] / l2norm_val_tst

            l2norm_val_tst = feat_test_tot_mle[i, :].norm(p=2)
            feat_test_tot_mle[i, :] = feat_test_tot_mle[i, :] / l2norm_val_tst

    if ar.normalize_models==1:
        # l2norm = torch.concat((zeroshot_weights_tot, mle_weights_tot), dim=1).norm(p=2)
        # zeroshot_weights_tot = zeroshot_weights_tot / l2norm
        # mle_weights_tot = mle_weights_tot / l2norm
        zeroshot_weights_tot = zeroshot_weights_tot / torch.norm(zeroshot_weights_tot)
        mle_weights_tot = mle_weights_tot / torch.norm(mle_weights_tot)


    ################## Now evaluation starts ###################
    # (1) simple concatenation of all chosen models
    # (2) output averaging
    # (3) fisher merging
    # (4) bma

    how_many_steps_in_test = int(tot_test / ar.batch_size)

    m = nn.Softmax(dim=1)

    if ar.data_name == 'objectnet':
        data_dir = '../databank/'
        from model_soups.datasets.objectnet import ObjectNet
        from model_soups.datasets.imagenet_classnames import objectnet_classnames
        dataset = ObjectNet(location=data_dir, preprocess=[], batch_size=10, num_workers=int(1),
                            classnames=objectnet_classnames)
        projection_fn = getattr(dataset, 'project_logits', None)



    test_accs_top1_tot = []
    test_accs_top1_avg_output = []
    test_accs_top1_max_output = []

    model_to_evaluate = torch.concat((zeroshot_weights_tot, mle_weights_tot), dim=1).to(ar.device)
    # model_to_evaluate = zeroshot_weights_tot.to(ar.device)

    tot_num_test_evaluated = 0

    for i in range(how_many_steps_in_test):

        ### method 1: concatenate weights ###

        if i == (how_many_steps_in_test - 1):
            # feat1 = feat_test_tot_zeroshot[ar.batch_size * i:, :]
            feat1 =  torch.concat((feat_test_tot_zeroshot[ar.batch_size * i:, :], feat_test_tot_mle[ar.batch_size * i:, :]), dim=1)
            labels = label_test[ar.batch_size * i:]
        else:
            #feat1 = feat_test_tot_zeroshot[ar.batch_size * i:ar.batch_size * (i + 1), :]
            feat1 = torch.concat((feat_test_tot_zeroshot[ar.batch_size * i:ar.batch_size * (i + 1), :], feat_test_tot_mle[ar.batch_size * i:ar.batch_size * (i + 1), :]), dim=1)
            labels = label_test[ar.batch_size * i:ar.batch_size * (i + 1)]

        # print(labels)
        tot_num_test_evaluated = tot_num_test_evaluated + feat1.shape[0]

        feat1 = feat1.to(ar.device)
        labels = labels.to(ar.device)

        if ar.data_name == 'objectnet':
            outputs = torch.mm(feat1,  model_to_evaluate.t()) # bs x 1000
            outputs = projection_fn(outputs.detach(), ar.device) # bs x 113
        else:
            outputs = torch.mm(feat1,  model_to_evaluate.t())

        top1_acc = accuracy(outputs, labels, topk=(1,))
        test_accs_top1_tot.append(top1_acc[0])


        ### method 2: average output probability (ensemble) ###

        if ar.data_name == 'objectnet':
            # output_tot = torch.zeros(feat1.shape[0], 113, numb_candidates_zeroshot)
            output_tot = torch.zeros(feat1.shape[0], 113, numb_candidates_zeroshot+numb_candidates_mle)
            #output_tot_max = torch.zeros(feat1.shape[0], numb_candidates_zeroshot + numb_candidates_mle)
        else:
            # output_tot = torch.zeros(feat1.shape[0], numb_classes, numb_candidates_zeroshot)
            output_tot = torch.zeros(feat1.shape[0], numb_classes, numb_candidates_zeroshot+numb_candidates_mle)
            #output_tot_max = torch.zeros(feat1.shape[0], numb_candidates_zeroshot + numb_candidates_mle)

        for j in range(numb_candidates_zeroshot):

            if j == 0:
                model = zeroshot_weights_tot[:, :dim_tot_zeroshot[0]]
                feat = feat1[:, :dim_tot_zeroshot[0]]
            elif j == 1:
                model = zeroshot_weights_tot[:, dim_tot_zeroshot[0]:sum(dim_tot_zeroshot[0:2])]
                feat = feat1[:, dim_tot_zeroshot[0]:sum(dim_tot_zeroshot[0:2])]
            else:
                model = zeroshot_weights_tot[:, sum(dim_tot_zeroshot[0:j]):sum(dim_tot_zeroshot[0:(j + 1)])]
                feat = feat1[:,sum(dim_tot_zeroshot[0:j]):sum(dim_tot_zeroshot[0:(j + 1)])]

            model = model.to(ar.device)
            output_model = torch.mm(feat, model.t())

            if ar.data_name == 'objectnet':
                output_model = projection_fn(output_model.detach(), ar.device)

            softmax_output = m(output_model)
            output_tot[:, :, j] = softmax_output

        last_ind_zeroshot = sum(dim_tot_zeroshot[0:(j + 1)])

        for j in range(numb_candidates_mle):

            if j == 0:
                model = mle_weights_tot[:, :dim_tot_mle[0]]
                feat = feat1[:, last_ind_zeroshot:last_ind_zeroshot+dim_tot_mle[0]]
            elif j == 1:
                model = mle_weights_tot[:, dim_tot_mle[0]:sum(dim_tot_mle[0:2])]
                feat = feat1[:, last_ind_zeroshot+dim_tot_mle[0]:last_ind_zeroshot+sum(dim_tot_mle[0:2])]
            else:
                model = mle_weights_tot[:, sum(dim_tot_mle[0:j]):sum(dim_tot_mle[0:(j + 1)])]
                feat = feat1[:,last_ind_zeroshot+sum(dim_tot_mle[0:j]):last_ind_zeroshot+sum(dim_tot_mle[0:(j + 1)])]

            model = model.to(ar.device)
            output_model = torch.mm(feat, model.t())

            if ar.data_name == 'objectnet':
                output_model = projection_fn(output_model.detach(), ar.device)

            softmax_output = m(output_model)
            output_tot[:, :, j+numb_candidates_zeroshot] = softmax_output

            # max_val, max_ind = torch.max(softmax_output, dim=1)
            # output_tot_max[:, j + numb_candidates_zeroshot] = max_val



        # for k in range(numb_candidates_zeroshot+numb_candidates_mle):
        #     outputs = output_tot[:,:,k].to(ar.device)
        #     # outputs = torch.mean(output_tot, dim=2).to(ar.device)
        #     top1_acc = accuracy(outputs, labels, topk=(1,))
        #     test_accs_top1_avg_output.append(top1_acc[0]) #tot_num_test_evaluatedx(numb_candidates_zeroshot+numb_candidates_mle)
        outputs = torch.mean(output_tot, dim=2).to(ar.device)
        top1_acc = accuracy(outputs, labels, topk=(1,))
        test_accs_top1_avg_output.append(top1_acc[0]) #tot_num_test_evaluatedx(numb_candidates_zeroshot+numb_candidates_mle)

        #all_zero = torch.zeros_like(softmax_output, device=ar.device) # batch_size by numb_classes
        max_val, max_ind = torch.max(output_tot, dim=2) # max_ind contains which candidate
        # mm_val, mm_ind = torch.max(max_val, dim=1)
        #
        # all_zero[:, mm_ind] = mm_val.to(device=ar.device, dtype = torch.float64)

        top1_acc = accuracy(m(max_val.to(device=ar.device, dtype = torch.float64)), labels, topk=(1,))
        test_accs_top1_max_output.append(top1_acc[0])



    print(f'combined top1 acc on {ar.data_name}: {np.mean(test_accs_top1_tot):.3f}')
    print(f'output avg top1 acc on {ar.data_name}: {np.mean(test_accs_top1_avg_output):.3f}')
    print(f'output max top1 acc on {ar.data_name}: {np.mean(test_accs_top1_max_output):.3f}')
    print(f'total number of test points:{tot_num_test_evaluated}')

    #
    # ######## write the results on a txt file ##########
    filename = 'acc_' + ar.data_name +'_several_models.csv'



    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            g = csv.writer(file,delimiter='\t')
            g.writerow(['zeroshot_models', 'mle_models', 'normalize_models', 'concat_acc', 'avg_acc'])


    with open(filename, 'a') as file:
        g = csv.writer(file, delimiter='\t')
        g.writerow(['{:}'.format(ar.zeroshot_models), '{:}'.format(ar.mle_models), ar.normalize_models, '{:}'.format(np.mean(test_accs_top1_tot)),
                    '{:.2f}'.format(np.mean(test_accs_top1_avg_output))])


#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
