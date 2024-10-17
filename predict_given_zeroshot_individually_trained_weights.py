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
    parser.add_argument('--device', type=str, default='cuda:1', help='gpu or cpu')
    parser.add_argument('--method', type=str, default='mle', help='zeroshot or mle or bma')
    parser.add_argument('--which-model-to-use', type=str, default='4,3,0,2', help='Any combination among 0,1,2,3,4,5,6,7')
    parser.add_argument('--which-rate-to-use', type=str, default='1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1',
                        help='Any combination among 1.0, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01')
    parser.add_argument('--which-epoch', type=int, default=30)
    parser.add_argument('--batch-size', '-bs', type=int, default=1000, help='batch size during validation')
    parser.add_argument('--normalize-features', action='store_true', default=True,
                        help='normalize all features of all models')

    # '1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1',
    # '1.0, 0.8, 0.6, 0.4, 0.2, 0.01',
    # '0.9, 0.7, 0.5, 0.3, 0.1'
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

    # model directory
    if ar.method=='zeroshot':
        if ar.data_name == 'imagenet' or ar.data_name == 'imagenet_v2' or ar.data_name=='imagenet_sketch' or ar.data_name=='objectnet':
            home_dir = os.getcwd() + f"/imagenet_results"
        else: #
            home_dir = os.getcwd() + f"/{ar.data_name}_results"
    else:
        home_dir = os.getcwd() + f"/imagenet_results"

    # eval data directory
    feat_dir = os.getcwd()
    feat_dir = feat_dir + f"/feature_representations/all_models_for_{ar.data_name}"

    # store zeroshot weights or trained weights
    if ar.method=='bma':
        selection_rates = [float(i) for i in ar.which_rate_to_use.split(',')] # ['1','2','3','4']
        zeroshot_weights_tot = torch.zeros(numb_classes, tot_feat_dim, len(selection_rates))
        model_posterior_weight = torch.zeros(numb_candidates, len(selection_rates))
    else:
        zeroshot_weights_tot = torch.zeros(numb_classes, tot_feat_dim)

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

    for i in range(numb_candidates):
        model_name = model_names[i]
        pretrained_with = datasets_pretrained_with[i]

        #### load the pretrained or zershot models to test ####
        if ar.method == 'zeroshot':
            if ar.data_name == 'imagenet' or ar.data_name == 'imagenet_v2' or ar.data_name == 'imagenet_sketch' or ar.data_name == 'objectnet':
                    zeroshot_weights = torch.load(
                        home_dir + f"/zeroshot_weights/zero_shot_weights_imagenet_{model_name}_{pretrained_with}.pt") # feat_dim by numb_classes
            else: #ar.data_name == 'imagenet_a' and ar.data_name == 'imagenet_r'
                zeroshot_weights = torch.load(
                    home_dir + f"/zeroshot_weights/zero_shot_weights_{ar.data_name}_{model_name}_{pretrained_with}.pt")  # feat_dim by numb_classes

        elif ar.method =='mle':
            if ar.data_name == 'imagenet' or ar.data_name == 'imagenet_v2' or ar.data_name == 'imagenet_r' \
                    or ar.data_name == 'imagenet_a' or ar.data_name == 'imagenet_sketch' or ar.data_name == 'objectnet':
                # mle = torch.load(
                #     home_dir + f"/{model_name}_imagenet_mle_50/imagenet_mle_{ar.which_epoch}.pt") # feat_dim by numb_classes
                model_dir = os.getcwd() + f"/imagenet_results"
                filename = fnmatch.filter(os.listdir(model_dir + f"/[{indices_for_selected_models[i]}]_model_mle"), 'At_*.pt')
                mle = torch.load(model_dir + f"/[{indices_for_selected_models[i]}]_model_mle/"+filename[0])
            zeroshot_weights = mle['weight']

            if ar.data_name == 'imagenet_a':
                from imagenet_a import CLASS_SUBLIST
                zeroshot_weights = zeroshot_weights[CLASS_SUBLIST, :]
            elif ar.data_name == 'imagenet_r':
                from imagenet_r import CLASS_SUBLIST
                zeroshot_weights = zeroshot_weights[CLASS_SUBLIST, :]

        elif ar.method =='bma':
            savedir = f"/[{indices_for_selected_models[i]}]_model_map"
            savedir = home_dir + savedir
            # load each model and store it in one giant tensor
            # numb_classes by feat_dim by len(selection_rates)
            feat_dim = dim_tot[i]
            zeroshot_weights = torch.zeros(numb_classes, feat_dim, len(selection_rates))
            # classifier = Linear(in_features=feat_dim, out_features=1000, bias=False).to(ar.device)
            for j in range(len(selection_rates)):
                sel_rate = selection_rates[j]

                filename = fnmatch.filter(os.listdir(savedir), f'At_*selrate={sel_rate}*.pt')
                classifier = torch.load(savedir + "/" + filename[0])
                classifier = classifier['weight']
                # classifier = classifier.to(ar.device)

                if ar.data_name == 'imagenet_a':
                    from imagenet_a import CLASS_SUBLIST
                    classifier = classifier[CLASS_SUBLIST, :]
                elif ar.data_name == 'imagenet_r':
                    from imagenet_r import CLASS_SUBLIST
                    classifier = classifier[CLASS_SUBLIST, :]

                zeroshot_weights[:,:,j] = classifier

                # load model posterior weight
                if ar.data_name == 'imagenet' or ar.data_name == 'imagenet_v2' or ar.data_name == 'imagenet_r' \
                        or ar.data_name == 'imagenet_a' or ar.data_name == 'imagenet_sketch' or ar.data_name == 'objectnet':

                    filename = fnmatch.filter(os.listdir(savedir), f'weight_diag_imagenet_*sel_rate={sel_rate}.pt')
                    a = torch.load(savedir + "/" + filename[0])
                    model_posterior_weight[i,j] = a


        #### load the validation / test set ####
        if ar.data_name == 'imagenet':

            feat_test = torch.load(
                feat_dir + f"/feat_val_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            label_test = torch.load(
                feat_dir + f"/labels_val_data={ar.data_name}.pt")

        elif ar.data_name == 'imagenet_v2' or ar.data_name=='imagenet_r' or ar.data_name=='imagenet_a' or ar.data_name=='imagenet_sketch' \
                or ar.data_name =='objectnet':

            feat_test = torch.load(
                feat_dir + f"/feat_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            label_test = torch.load(
                feat_dir + f"/labels_data={ar.data_name}.pt")

        if i == 0:
            if ar.method == 'mle':
                zeroshot_weights_tot[:, 0:dim_tot[0]] = zeroshot_weights
            elif ar.method =='bma':
                zeroshot_weights_tot[:, 0:dim_tot[0],:] = zeroshot_weights
            else:
                zeroshot_weights_tot[:, 0:dim_tot[0]] = zeroshot_weights.t()
            feat_test_tot[:, 0:dim_tot[0]] = feat_test
            print(f'{i + 1}st models zeroshot weights loaded')
            print(f'index from 0 to {dim_tot[0]}')
        else:
            if ar.method == 'mle':
                zeroshot_weights_tot[:, sum(dim_tot[0:i]):sum(dim_tot[0:(i + 1)])] = zeroshot_weights
            elif ar.method == 'bma':
                zeroshot_weights_tot[:, sum(dim_tot[0:i]):sum(dim_tot[0:(i + 1)]), :] = zeroshot_weights
            else:
                zeroshot_weights_tot[:, sum(dim_tot[0:i]):sum(dim_tot[0:(i + 1)])] = zeroshot_weights.t()
            feat_test_tot[:, sum(dim_tot[0:i]):sum(dim_tot[0:(i + 1)])] = feat_test
            print(f'{i + 1}rd models zeroshot weights loaded')
            print(f'index from {sum(dim_tot[0:i])} to {sum(dim_tot[0:(i + 1)])}')

    ### normalize the posterior model weights
    if ar.method=='bma':
        print(f'unnormalized posterior model weights are {model_posterior_weight}')
        model_posterior_weight = model_posterior_weight/torch.sum(model_posterior_weight)
        print(f'normalized posterior model weights for  are {model_posterior_weight}')



    if ar.normalize_features:
        print('we normalize the features of each datapoint')

        ## memory problem if more than 5 models were used.
        # norm = feat_test_tot.norm(p=2, dim=1, keepdim=True)  # tot_test by 1
        # feat_test_tot = feat_test_tot.div(norm)

        for i in range(tot_test):
            l2norm_val_tst = feat_test_tot[i,:].norm(p=2)
            feat_test_tot[i, :] = feat_test_tot[i, :] / l2norm_val_tst
            # print(feat_test_tot[i,:].norm(p=2))

        # zeroshot_weights_tot = zeroshot_weights_tot/zeroshot_weights_tot.norm(p=2)
        # if ar.method == 'bma':
        #     for i in range(zeroshot_weights_tot.shape[2]):
        #         l2norm_val = zeroshot_weights_tot[:,:,i].norm(p=2)
        #         zeroshot_weights_tot[:,:,i] = zeroshot_weights_tot[:,:,i]/l2norm_val


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

    if ar.method=='zeroshot' or ar.method=='mle':

        model_to_evaluate = zeroshot_weights_tot.to(ar.device)

        test_accs_top1_zeroshot_tot = []
        test_accs_top1_avg_output = []
        test_accs_top1_max_output = []

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

            if ar.data_name == 'objectnet':
                outputs = torch.mm(feat1,  model_to_evaluate.t()) # bs x 1000
                outputs = projection_fn(outputs.detach(), ar.device) # bs x 113
            else:
                outputs = torch.mm(feat1,  model_to_evaluate.t())

            top1_acc = accuracy(outputs, labels, topk=(1,))
            test_accs_top1_zeroshot_tot.append(top1_acc[0])

            ### method 2: average output probability (ensemble) ###
            if ar.data_name == 'objectnet':
                output_tot = torch.zeros(feat1.shape[0], 113, numb_candidates)
            else:
                output_tot = torch.zeros(feat1.shape[0], numb_classes, numb_candidates)

            for j in range(numb_candidates):

                if j == 0:
                    model = zeroshot_weights_tot[:, :dim_tot[0]]
                    feat = feat1[:, :dim_tot[0]]
                elif j == 1:
                    model = zeroshot_weights_tot[:, dim_tot[0]:sum(dim_tot[0:2])]
                    feat = feat1[:, dim_tot[0]:sum(dim_tot[0:2])]
                else:
                    model = zeroshot_weights_tot[:, sum(dim_tot[0:j]):sum(dim_tot[0:(j + 1)])]
                    feat = feat1[:,sum(dim_tot[0:j]):sum(dim_tot[0:(j + 1)])]

                model = model.to(ar.device)
                output_model = torch.mm(feat, model.t())

                if ar.data_name == 'objectnet':
                    output_model = projection_fn(output_model.detach(), ar.device)

                softmax_output = m(output_model)
                output_tot[:, :, j] = softmax_output

            outputs = torch.mean(output_tot, dim=2).to(ar.device)
            top1_acc = accuracy(outputs, labels, topk=(1,))
            test_accs_top1_avg_output.append(top1_acc[0])

            max_val, max_ind = torch.max(output_tot, dim=2)  # max_ind contains which candidate
            top1_acc = accuracy(m(max_val.to(device=ar.device, dtype=torch.float64)), labels, topk=(1,))
            test_accs_top1_max_output.append(top1_acc[0])

        print(f'combined zeroshot top1 acc on {ar.data_name}: {np.mean(test_accs_top1_zeroshot_tot):.3f}')
        print(f'output avg top1 acc on {ar.data_name}: {np.mean(test_accs_top1_avg_output):.3f}')
        print(f'output max top1 acc on {ar.data_name}: {np.mean(test_accs_top1_max_output):.3f}')
        print(f'total number of test points:{tot_num_test_evaluated}')

    elif ar.method=='bma':

        test_accs_top1_bma = []
        test_accs_top1_output_avg = []
        test_accs_top1_max_output = []

        tot_num_test_evaluated = 0

        ### method 3: bma ###
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

            if ar.data_name == 'objectnet':
                output_tot = torch.zeros(feat1.shape[0], 113, numb_candidates, len(selection_rates))
                output_tot_uniform = torch.zeros(feat1.shape[0], 113, numb_candidates, len(selection_rates))
            else:
                output_tot = torch.zeros(feat1.shape[0], numb_classes, numb_candidates, len(selection_rates))
                output_tot_uniform = torch.zeros(feat1.shape[0], numb_classes, numb_candidates, len(selection_rates))

            for j in range(numb_candidates):
                for k in range(len(selection_rates)):

                    if j == 0:
                        model = zeroshot_weights_tot[:, :dim_tot[0], k]
                        feat = feat1[:, :dim_tot[0]]
                    elif j == 1:
                        model = zeroshot_weights_tot[:, dim_tot[0]:sum(dim_tot[0:2]), k]
                        feat = feat1[:, dim_tot[0]:sum(dim_tot[0:2])]
                    else:
                        model = zeroshot_weights_tot[:, sum(dim_tot[0:j]):sum(dim_tot[0:(j + 1)]), k]
                        feat = feat1[:,sum(dim_tot[0:j]):sum(dim_tot[0:(j + 1)])]

                    model = model.to(ar.device)
                    output_model = torch.mm(feat, model.t())

                    if ar.data_name == 'objectnet':
                        output_model = projection_fn(output_model.detach(), ar.device)

                    softmax_output = m(output_model)
                    output_tot[:, :, j, k] = softmax_output * model_posterior_weight[j,k]
                    output_tot_uniform[:, :, j, k] = softmax_output

            outputs = torch.sum(output_tot, dim=3).to(ar.device)
            outputs = torch.sum(outputs, dim=2)
            top1_acc = accuracy(outputs, labels, topk=(1,))
            test_accs_top1_bma.append(top1_acc[0])

            outputs = torch.sum(output_tot_uniform, dim=3).to(ar.device)
            outputs = torch.sum(outputs, dim=2)
            top1_acc = accuracy(outputs, labels, topk=(1,))
            test_accs_top1_output_avg.append(top1_acc[0])

            m_v, m_i = torch.max(output_tot_uniform, dim=3)
            max_val, max_ind = torch.max(m_v, dim=2)
            # max_val, max_ind = torch.max(torch.sum(output_tot_uniform, dim=3), dim=2)  # max_ind contains which candidate
            top1_acc = accuracy(m(max_val.to(device=ar.device, dtype=torch.float64)), labels, topk=(1,))
            test_accs_top1_max_output.append(top1_acc[0])

        print(f'bma top1 acc on {ar.data_name}: {np.mean(test_accs_top1_bma):.3f}')
        print(f'output avg top1 acc on {ar.data_name}: {np.mean(test_accs_top1_output_avg):.3f}')
        print(f'output max top1 acc on {ar.data_name}: {np.mean(test_accs_top1_max_output):.3f}')
        print(f'total number of test points:{tot_num_test_evaluated}')


    # ######## write the results on a txt file ##########
    if ar.method=='zeroshot':
        filename = 'acc_'+ar.data_name+'.csv'
    else:
        filename = 'acc_' + ar.data_name +'_'+ar.method+'.csv'



    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            g = csv.writer(file,delimiter='\t')
            if ar.method=='zeroshot' or ar.method=='mle':
                g.writerow(['model_ind', 'method', 'concat_acc', 'avg_acc'])
            elif ar.method=='bma':
                g.writerow(['model_ind', 'method', 'sel_rates', 'top1_acc'])


    with open(filename, 'a') as file:
        g = csv.writer(file, delimiter='\t')
        if ar.method=='zeroshot' or ar.method=='mle':
            g.writerow(['{:}'.format(ar.which_model_to_use), ar.method, '{:}'.format(np.mean(test_accs_top1_zeroshot_tot)),
                        '{:.2f}'.format(np.mean(test_accs_top1_avg_output))])

        elif ar.method == 'bma':
            g.writerow(['{:}'.format(ar.which_model_to_use), ar.method, '{:}'.format(selection_rates), '{:.2f}'.format(np.mean(test_accs_top1_bma))])



#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
