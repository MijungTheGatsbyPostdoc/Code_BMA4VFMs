# test computing Hessian
import torch
from torch.autograd.functional import hessian
import argparse
import backpack
from backpack import backpack, extend
from backpack.extensions import (
    DiagHessian,
)

import torch
import os
import torch.nn as nn
from torch.nn import Linear
from load_train_set import load_train_set
import fnmatch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', type=str, default='imagenet', help='target dataset')
    parser.add_argument('--selection-rate', type=float, default=0.35, help='1 means select all ; 0.1 means select 10% of features')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu or cpu')
    # Load a single model's features or all models' features
    parser.add_argument('--which-model-to-use', type=str, default='3',
                        help='Choose one of 0,1,2,3,4,5,6,7')
    parser.add_argument('--prior-var', type=float, default=80, help='prior var')

    ar = parser.parse_args()
    return ar

def main(**kwargs):

    ar = get_args()
    # print(ar)
    device = ar.device
    data_name = ar.data_name
    method = 'map'

    home_dir = os.getcwd() + f"/{data_name}_results"

    ########### load pre-processed data (feature extracted data) ###########
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

    savedir = f"/{indices_for_selected_models}_model_{method}"
    savedir = home_dir + savedir

    prior_var = ar.prior_var

    ### select feature dimensions
    # MLE magnitude based selection
    if data_name == 'imagenet':  # later change where MLE is located

        batch_size = 1000
        tot_train = 1281167
        numb_classes = 1000

        classifier = Linear(in_features=tot_feat_dim, out_features=numb_classes, bias=False).to(ar.device)

        filename = fnmatch.filter(os.listdir(savedir), f'At_*selrate={ar.selection_rate}*.pt')
        classifier.load_state_dict(torch.load(savedir+ "/" + filename[0]))
        classifier = classifier.to(device)

    else:
        print('you need to specify classifier size, etc. depending on the choice of model')

    ## to load the training data
    feat_train, label_train = load_train_set(data_name, ar.which_model_to_use, device)

    train_idx = torch.randperm(tot_train)
    how_many_steps_in_each_epoch = int(tot_train / batch_size)

    diag_h = 0
    grad_sum = 0
    loss_val = 0

    loss_fn = nn.CrossEntropyLoss(reduction='sum')  # default is mean

    # to use backpack to compute Hessian
    classifier = extend(classifier)
    loss_fn = extend(loss_fn)

    for i in range(how_many_steps_in_each_epoch):

        if i == (how_many_steps_in_each_epoch - 1):  # in case of imagenet: 1281000 + 167
            train_idx_per_batch = train_idx[batch_size * i:]
            # print(f"batch index : from {ar.batch_size * i} till {ar.batch_size * i +  train_idx_per_batch.shape[0]}")
        else:
            train_idx_per_batch = train_idx[batch_size * i:batch_size * (i + 1)]
            # print(f"batch index : from {ar.batch_size*i} till {ar.batch_size*(i+1)-1}")

        feat1 = feat_train[train_idx_per_batch, :]
        labels = label_train[train_idx_per_batch]
        feat1 = feat1.to(ar.device)
        labels = labels.to(ar.device)

        outputs = classifier(feat1)
        loss = loss_fn(outputs, labels) # negative log likelihood

        with backpack(DiagHessian()):
            loss.backward()

        for name, param in classifier.named_parameters():
            grad_sum = grad_sum + (param.grad)**2
            diag_h = diag_h + param.diag_h
            ############## add param.grad or something ###########

        loss_val = loss_val + loss.detach()
        del loss, feat1, labels, outputs


    # compute logdet(H) here
    # HS + I
    HS_I = diag_h*prior_var + 1
    logdet_HS_I_diag = torch.sum(torch.log(HS_I))

    # for testing fisher merging
    grad_avg = grad_sum / tot_train
    torch.save(grad_avg, savedir + "/" + f"grad_{data_name}_{ar.which_model_to_use}_sel_rate=={ar.selection_rate}.pt")

    torch.save(loss_val, savedir + "/" + f"neg_log_likeli_{data_name}_{ar.which_model_to_use}_sel_rate=={ar.selection_rate}.pt") # negative log likelihood

    neg_log_prior = torch.sum((classifier.weight.t()) ** 2) /(2*prior_var)
    torch.save(neg_log_prior, savedir + "/" + f"neg_log_prior_{data_name}_{ar.which_model_to_use}_sel_rate={ar.selection_rate}.pt")

    weight_diag = torch.exp((- loss_val - neg_log_prior - 0.5*logdet_HS_I_diag)/tot_train)

    torch.save(weight_diag, savedir + "/" + f"weight_diag_{data_name}_{ar.which_model_to_use}_sel_rate={ar.selection_rate}.pt")

    print(f'with selection rate={ar.selection_rate}, weight with diag H is {weight_diag}')

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
