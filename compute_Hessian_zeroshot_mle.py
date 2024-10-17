# test computing Hessian and compute posterior model weights for zeroshot models

import torch
from torch.autograd.functional import hessian
import argparse
import backpack
from backpack import backpack, extend
from backpack.extensions import (
    DiagHessian, BatchGrad
)

import torch
import os
import torch.nn as nn
from torch.nn import Linear
from load_test_set import load_test_set
import fnmatch
from training_with_processed_data import accuracy
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--which-model-to-use', type=str, default='1',
                        help='Choose one of 0,1,2,3,4,5,6,7')

    parser.add_argument('--method', type=str, default='mle', help='zeroshot or mle')
    parser.add_argument('--data-name-model', type=str, default='objectnet', help='if method==zeroshot, imagenet_a and r have different models from all other models. if method==mle, use imagenet')
    parser.add_argument('--data-name-marginal', type=str, default='objectnet', help='dataset to compute marginal likelihood with')
    parser.add_argument('--device', type=str, default='cpu', help='gpu or cpu')
    parser.add_argument('--prior-var', type=float, default=0.000001, help='prior var')

    ar = parser.parse_args()
    return ar

def main(**kwargs):

    ar = get_args()
    # print(ar)
    device = ar.device
    data_name = ar.data_name_model
    data_name_marg = ar.data_name_marginal
    method = ar.method

    model_names_tot = ['ViT-H-14-378-quickgelu', 'ViT-H-14-quickgelu', 'EVA02-E-14-plus', 'ViT-SO400M-14-SigLIP-384',
                       'ViT-bigG-14-CLIPA-336',  'EVA02-E-14', 'ViT-H-14-quickgelu', 'convnext_xxlarge']
    datasets_pretrained_with_tot = ['dfn5b', 'dfn5b', 'laion2b_s9b_b144k', 'webli',
                                    'datacomp1b', 'laion2b_s4b_b115k','metaclip_fullcc', 'laion2b_s34b_b82k_augreg_soup']

    dim_tot_tot = [1024, 1024, 1024, 1152, 1280, 1024, 1024, 1024]
    indices_for_selected_models = [int(i) for i in ar.which_model_to_use.split(',')]  # ['1','2','3','4']
    model_name = [model_names_tot[i] for i in indices_for_selected_models]
    pretrained_with = [datasets_pretrained_with_tot[i] for i in indices_for_selected_models]
    dim_tot = [dim_tot_tot[i] for i in indices_for_selected_models]
    numb_candidates = len(model_name)
    tot_feat_dim = sum(dim_tot[0:numb_candidates])
    print(f'total feature dimension of selected models is {tot_feat_dim}')

    ######################## zeroshot models #############################

    if data_name == 'imagenet' or data_name == 'imagenet_v2' or data_name == 'imagenet_sketch' or data_name == 'objectnet':
        home_dir = os.getcwd() + f"/imagenet_results"
    else:  #
        home_dir = os.getcwd() + f"/{data_name}_results"
    ### directory to store results
    savedir = f"/{indices_for_selected_models[0]}_model_{method}"
    savedir = home_dir + savedir
    os.makedirs(savedir, exist_ok=True)

    if data_name == 'imagenet' or data_name == 'imagenet_v2' or data_name == 'imagenet_sketch' or data_name == 'objectnet':
        if method=='zeroshot':
            zeroshot_weights = torch.load(home_dir + f"/zeroshot_weights/zero_shot_weights_imagenet_{model_name[0]}_{pretrained_with[0]}.pt", map_location=torch.device(device))  # feat_dim by numb_classes

        else: #mle:
            model_dir = os.getcwd() + f"/imagenet_results"
            filename = fnmatch.filter(os.listdir(model_dir + f"/[{indices_for_selected_models[0]}]_model_mle"),
                                      'At_*.pt')
            zeroshot_weights = torch.load(model_dir + f"/[{indices_for_selected_models[0]}]_model_mle/" + filename[0], map_location=torch.device(device))
            zeroshot_weights = zeroshot_weights['weight'].t()

    else:  # ar.data_name == 'imagenet_a' and ar.data_name == 'imagenet_r'
        if method=='zeroshot':
            zeroshot_weights = torch.load(
                home_dir + f"/zeroshot_weights/zero_shot_weights_{data_name}_{model_name[0]}_{pretrained_with[0]}.pt", map_location=torch.device(device))  # feat_dim by numb_classes

        else: #mle:
            model_dir = os.getcwd() + f"/imagenet_results"
            filename = fnmatch.filter(os.listdir(model_dir + f"/[{indices_for_selected_models[0]}]_model_mle"),
                                      'At_*.pt')
            zeroshot_weights = torch.load(model_dir + f"/[{indices_for_selected_models[0]}]_model_mle/" + filename[0], map_location=torch.device(device))
            zeroshot_weights = zeroshot_weights['weight']

            if data_name == 'imagenet_a':
                from imagenet_a import CLASS_SUBLIST
                zeroshot_weights = zeroshot_weights[CLASS_SUBLIST, :]
            elif data_name == 'imagenet_r':
                from imagenet_r import CLASS_SUBLIST
                zeroshot_weights = zeroshot_weights[CLASS_SUBLIST, :]

            zeroshot_weights = zeroshot_weights.t()

    ### we normalize weights such that their magnitude doesn't alter the loss value
    zeroshot_weights = zeroshot_weights/torch.norm(zeroshot_weights,dim=0, keepdim=True) # feature dim by classes

    numb_classes = zeroshot_weights.shape[1]

    classifier = Linear(in_features=tot_feat_dim, out_features=numb_classes, bias=False).to(device)
    classifier.weight.data = zeroshot_weights.t()
    classifier = classifier.to(device)

    ## to load the data you want to evaluate the model marginal likelihood
    feat_train, label_train = load_test_set(data_name_marg, ar.which_model_to_use, device)

    if data_name=='imagenet_r' or data_name=='imagenet_a':
        # when using the zeroshot model for imagenet_r and imagenet_a, numb_classes == 200, while objectnet has 112 classes
        # need to map imagenet classes to those selected classes
        if data_name == 'imagenet_a':
            from imagenet_a import CLASS_SUBLIST
        elif data_name == 'imagenet_r':
            from imagenet_r import CLASS_SUBLIST
        labels=[]
        feats=[]
        label_train = label_train.to(device)
        for i in range(len(CLASS_SUBLIST)):
            j = CLASS_SUBLIST[i]
            labels.append(label_train[label_train==j])
            feats.append(feat_train[label_train==j])
        label_train = torch.concatenate(labels).to(device)
        feat_train = torch.concatenate(feats)


    batch_size = 1000
    prior_var = ar.prior_var
    tot_train = feat_train.shape[0]
    print("total datapoints to evaluate marginal-likelihood: ", tot_train)


    train_idx = torch.randperm(tot_train)
    how_many_steps_in_each_epoch = int(tot_train / batch_size)

    diag_h = 0
    grad_sum = 0
    loss_val = 0

    loss_fn = nn.CrossEntropyLoss(reduction='sum')  # default is mean
    # loss_fn = nn.CrossEntropyLoss()  # default is mean

    # to use backpack to compute Hessian
    classifier = extend(classifier)
    loss_fn = extend(loss_fn)
    test_acc = []

    # m = nn.Softmax(dim=1)

    if data_name == 'objectnet':
        data_dir = '../databank/'
        from model_soups.datasets.objectnet import get_metadata, ObjectNetDataset, ObjectNetBase, ObjectNet
        from model_soups.datasets.imagenet_classnames import objectnet_classnames
        dataset = ObjectNet(location=data_dir, preprocess=[], batch_size=10, num_workers=int(1),
                            classnames=objectnet_classnames)
        projection_fn = getattr(dataset, 'project_logits', None)

    for i in range(how_many_steps_in_each_epoch):

        print(f'{i}th batch out of {how_many_steps_in_each_epoch} batches')

        if i == (how_many_steps_in_each_epoch - 1):  # in case of imagenet: 1281000 + 167
            train_idx_per_batch = train_idx[batch_size * i:]
            # print(f"batch index : from {ar.batch_size * i} till {ar.batch_size * i +  train_idx_per_batch.shape[0]}")
        else:
            train_idx_per_batch = train_idx[batch_size * i:batch_size * (i + 1)]
            # print(f"batch index : from {ar.batch_size*i} till {ar.batch_size*(i+1)-1}")

        feat1 = feat_train[train_idx_per_batch, :] # bs x feat_dim
        feat1 = feat1 / torch.norm(feat1, dim=1, keepdim=True)
        labels = label_train[train_idx_per_batch]
        feat1 = feat1.to(device)
        labels = labels.to(device)

        outputs = classifier(feat1)
        if data_name == 'objectnet':
            # outputs # bs x 1000
            outputs = projection_fn(outputs.detach(), device)  # bs x 113

        loss = loss_fn(outputs, labels) # negative log likelihood

        top1_acc = accuracy(outputs, labels, topk=(1,))
        test_acc.append(top1_acc[0])

        # with backpack(DiagHessian()):
        # # with backpack(BatchGrad()):
        #     loss.backward()
        #
        # for name, param in classifier.named_parameters():
        #     # grad_sum = grad_sum + (param.grad)**2
        #     # grad_sum = grad_sum + torch.sum(param.grad_batch**2,dim=0)
        #     diag_h = diag_h + param.diag_h
        #     ############## add param.grad or something ###########

        loss_val = loss_val + loss.detach()
        del loss, feat1, labels, outputs

    print('test acc: ', np.mean(test_acc))


    # compute logdet(H) here
    # HS + I
    # diag_F = grad_sum
    # diag_h = 1/diag_F
    # HS_I = diag_h*prior_var + 1
    # logdet_HS_I_diag = torch.sum(torch.log(HS_I))

    # for testing fisher merging
    # grad_avg = grad_sum / tot_train
    # torch.save(grad_avg, savedir + "/" + f"grad_model:{data_name}_{ar.which_model_to_use}_data_for_marg:{data_name_marg}.pt")

    # torch.save(loss_val, savedir + "/" + f"neg_log_likeli_{data_name}_{ar.which_model_to_use}_data_for_marg:{data_name_marg}.pt") # negative log likelihood
    #
    # neg_log_prior = torch.sum((classifier.weight.t()) ** 2) /(2*prior_var)
    # torch.save(neg_log_prior, savedir + "/" + f"neg_log_prior_{data_name}_{ar.which_model_to_use}_data_for_marg:{data_name_marg}.pt")

    # weight_diag = torch.exp((- loss_val - neg_log_prior - 0.5*logdet_HS_I_diag)/tot_train)
    # weight_diag = torch.exp((- loss_val - neg_log_prior - 0.5 * logdet_HS_I_diag))
    weight_diag = torch.exp(- loss_val/tot_train)

    torch.save(weight_diag, savedir + "/" + f"weight_diag_{data_name}_{ar.which_model_to_use}_data_for_marg:{data_name_marg}.pt")

    print(f'{method}, {ar.which_model_to_use}, weight is {weight_diag}')

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
