# load all the posterior model weights and weigh the predictions accordingly.
# here to test as well
# (1) Fisher merging: based on the grad**2 values, mix weights
# (2) average weights (uniform soup/isotropic mergig): just average the weights
# (3) average logits (ensemble) : just average predictions
# (4) BMA with diag H:

import argparse
import matplotlib.pyplot as plt
from training_with_processed_data import accuracy
from torch.nn import Linear
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from training_with_processed_data import linear_model
import torch.nn as nn
import time
from torch.nn import Linear
from load_val_set import load_val_set
import matplotlib.pyplot as plt

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='imagenet_v2', help='target dataset: imagenet, imagenet_r, imagenet_a, imagenet_sketch, imagenet_v2')
    parser.add_argument('--device', type=str, default='cuda:1', help='gpu or cpu')
    parser.add_argument('--method', type=str, default='map', help='map or mle')
    parser.add_argument('--epochs', '-ep', type=int, default=100)
    parser.add_argument('--batch-size', '-bs', type=int, default=1000, help='batch size during validation')

    ar = parser.parse_args()
    return ar

def main(**kwargs):

    ar = get_args()
    print(ar)

    # load validation data
    feat_test, label_test = load_val_set(ar.data_name)

    # load all models
    if ar.data_name == 'imagenet' or ar.data_name == 'imagenet_v2' or ar.data_name=='imagenet_r' \
            or ar.data_name=='imagenet_a' or ar.data_name=='imagenet_sketch' or ar.data_name=='objectnet':
        home_dir = os.getcwd() + f"/imagenet_results"
    else:
        home_dir = os.getcwd() + f"/{ar.data_name}_results"

    feat_dim = feat_test.shape[1]

    # if ar.data_name=='imagenet':
        # selection_rate = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
        # # prior_var = 40.0*torch.ones_like(selection_rate)
        # prior_var = [10.0, 40.0, 40.0, 80.0, 10.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]
        # train_epoch =[21, 13, 11, 82, 97, 28, 31, 36, 53, 182, 387]

    # selection_rate = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
    # prior_var = [40.0, 80.0, 10.0, 10.0, 40.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]
    # train_epoch = [60, 84, 10, 72, 10, 32, 66, 26, 53, 43, 98]

    # selection_rate = [1.0, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
    # prior_var = [40.0, 40.0, 10.0, 80.0, 10.0, 40.0, 80.0, 80.0, 80.0, 80.0, 80.0, 40.0]
    # train_epoch = [13, 21, 50, 14, 25, 25, 21, 44, 55, 62, 73, 99]

    # selection_rate = [1.0, 0.8,  0.6, 0.4, 0.2, 0.01]
    # prior_var = [40.0, 80.0, 40.0, 80.0, 80.0, 40.0]
    # train_epoch = [13, 14, 25, 21, 44, 62, 99]

    # selection_rate = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
    # prior_var = [40.0, 10.0, 80.0, 10.0, 40.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]
    # train_epoch = [13, 50, 14, 25, 25, 21, 44, 55, 62, 73, 399]

    # selection_rate = [1.0, 0.9, 0.7,  0.5, 0.3,  0.1, 0.01]
    # prior_var = [40.0, 10.0, 10.0, 80.0, 80.0, 80.0, 40.0]
    # train_epoch = [13, 50, 25, 21, 55, 73, 99]
    # prior_var = 80*torch.ones_like(torch.tensor(prior_var))
    # train_epoch = 100*torch.ones_like(torch.tensor(train_epoch))

    selection_rate = [1.0, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
    prior_var = [40.0, 40.0, 10.0, 80.0, 10.0, 40.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]
    train_epoch = [13, 21, 50, 14, 25, 25, 21, 44, 55, 62, 73, 399]
    #
    # selection_rate = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
    # prior_var = [40.0, 10.0, 80.0, 10.0, 40.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0]
    # train_epoch = [13, 50, 14, 25, 25, 21, 44, 55, 62, 73, 399]
    # selection_rate = [0.99, 0.9, 0.7, 0.5, 0.3, 0.2]
    # prior_var = [40.0, 10.0, 10.0, 80.0, 80.0, 80.0]
    # train_epoch = [21, 50, 25, 21, 55, 62]


    if ar.data_name == 'imagenet' or ar.data_name == 'imagenet_v2' or ar.data_name=='imagenet_r' \
            or ar.data_name=='imagenet_a' or ar.data_name=='imagenet_sketch' or ar.data_name=='objectnet':
        mle_dir = f"/all_models_imagenet_mle_100"
        filename = home_dir + mle_dir + f"/imagenet_mle_18.pt"
        numb_classes = 1000
        map_dir = f"/all_models_imagenet_map_100"
        filename_map = home_dir + map_dir + f"/imagenet_map_25_prior_var==10.0_sel_rate==0.7.pt"
    else:
        numb_classes = len(torch.unique(label_test))
        mle_dir = f"/all_models_{ar.data_name}_mle_100"
        filename = home_dir + mle_dir + f"/{ar.data_name}_mle_18.pt"

    mle = Linear(in_features=feat_dim, out_features=numb_classes, bias=False).to(ar.device)
    mle.load_state_dict(torch.load(filename))
    mle = mle.weight  # numb_classes by tot_feat_dim

    # select the feature dimension to train

    ######################  Selection Criterion ############################

    # original_magnitude_of_mle = torch.sum(torch.abs(mle), dim=0)
    # torch.save(original_magnitude_of_mle, home_dir+'/orig_mag_mle.pt')


    sorted, indices = torch.sort(torch.sum(torch.abs(mle), dim=0), descending=True)
    # sorted, indices = torch.sort(torch.sum(mle**2, dim=0), descending=True)
    #####################################################################
    # torch.save(sorted, home_dir+'/mle_values_sorted.pt')
    # torch.save(indices, home_dir+'/mle_indices.pt')


    linear_model_tot = torch.zeros((feat_dim, numb_classes, len(selection_rate)), device=ar.device)
    weight_diag = torch.zeros(len(selection_rate))
    diag_h = torch.zeros((numb_classes, feat_dim, len(selection_rate)), device=ar.device)
    grad = torch.zeros((numb_classes, feat_dim, len(selection_rate)), device=ar.device)
    map_model = torch.zeros((numb_classes, feat_dim), device=ar.device)

    # plt.figure(1)
    #
    for i in range(len(selection_rate)):
        # print(i)
        epoch = train_epoch[i]
        sel_rate = selection_rate[i]
        var = prior_var[i]

        sel_feat_dim = int(sel_rate * feat_dim)  # selected feature dimension
        sel_feat_ind = indices[0:sel_feat_dim]
        sel_feat_ind.to('cpu')
        linear_model = Linear(in_features=sel_feat_dim, out_features=numb_classes, bias=False).to(ar.device)

        if ar.data_name == 'imagenet' or ar.data_name == 'imagenet_v2' or ar.data_name == 'imagenet_r' \
                or ar.data_name == 'imagenet_a' or ar.data_name == 'imagenet_sketch' or ar.data_name=='objectnet':
            savedir = f"/all_models_imagenet_{ar.method}_{ar.epochs}"
            savedir = home_dir + savedir
            filename = savedir + f"/imagenet_{ar.method}_{epoch}_prior_var=={var}_sel_rate=={sel_rate}.pt"
        else:
            savedir = f"/all_models_{ar.data_name}_{ar.method}_{ar.epochs}"
            savedir = home_dir + savedir
            filename = savedir + f"/{ar.data_name}_{ar.method}_{epoch}_prior_var=={var}_sel_rate=={sel_rate}.pt"
        linear_model.load_state_dict(torch.load(filename))
        linear_model_tot[sel_feat_ind, :, i] = linear_model.weight.t().to(ar.device)


        if ar.data_name == 'imagenet' or ar.data_name == 'imagenet_v2' or ar.data_name == 'imagenet_r' \
                or ar.data_name == 'imagenet_a' or ar.data_name == 'imagenet_sketch' or ar.data_name=='objectnet':
            # filename = savedir + f"/all_models_imagenet_{ar.method}_{ar.epochs}" + f"weight_diag_imagenet_sel_rate={sel_rate}.pt"
            filename = savedir + f"/all_models_imagenet_{ar.method}_{ar.epochs}" + f"weight_diag_imagenet_prior_var=={var}_sel_rate={sel_rate}.pt"
        else:
            # filename = savedir + f"/all_models_{ar.data_name}_{ar.method}_{ar.epochs}" + f"weight_diag_{ar.data_name}_sel_rate={sel_rate}.pt"
            # torch.save(weight_diag,
            #            savedir + f"weight_diag_{data_name}_prior_var=={prior_var}_sel_rate={ar.selection_rate}.pt")
            filename = savedir + f"/all_models_{ar.data_name}_{ar.method}_{ar.epochs}" + f"weight_diag_{ar.data_name}_prior_var=={var}_sel_rate={sel_rate}.pt"
            # f"weight_diag_{data_name}_prior_var=={prior_var}_sel_rate={ar.selection_rate}.pt")
        weight_diag[i] = torch.load(filename)

        if ar.data_name == 'imagenet' or ar.data_name == 'imagenet_v2' or ar.data_name == 'imagenet_r' \
                or ar.data_name == 'imagenet_a' or ar.data_name == 'imagenet_sketch' or ar.data_name=='objectnet':
            # filename = savedir + f"/all_models_imagenet_{ar.method}_{ar.epochs}" + f"diag_Hess_imagenet_sel_rate=={sel_rate}.pt"
            filename = savedir + f"/all_models_imagenet_{ar.method}_{ar.epochs}" + f"diag_Hess_imagenet__prior_var=={var}_sel_rate=={sel_rate}.pt"
        else:
            # filename = savedir + f"/all_models_{ar.data_name}_{ar.method}_{ar.epochs}" + f"diag_Hess_{ar.data_name}_sel_rate=={sel_rate}.pt"
            filename = savedir + f"/all_models_{ar.data_name}_{ar.method}_{ar.epochs}" + f"diag_Hess_{ar.data_name}__prior_var=={var}_sel_rate=={sel_rate}.pt"

        h = torch.load(filename).to(ar.device)
        sel_feat_ind.to(ar.device)
        diag_h[:,sel_feat_ind,i] = h

        if ar.data_name == 'imagenet' or ar.data_name == 'imagenet_v2' or ar.data_name == 'imagenet_r' \
                or ar.data_name == 'imagenet_a' or ar.data_name == 'imagenet_sketch' or ar.data_name=='objectnet':
            # filename = savedir + f"/all_models_imagenet_{ar.method}_{ar.epochs}" + f"grad_imagenet_sel_rate=={sel_rate}.pt"
            filename = savedir + f"/all_models_imagenet_{ar.method}_{ar.epochs}" + f"grad_imagenet_prior_var=={var}_sel_rate=={sel_rate}.pt"
        else:
            # filename = savedir + f"/all_models_{ar.data_name}_{ar.method}_{ar.epochs}" + f"grad_{ar.data_name}_sel_rate=={sel_rate}.pt"
            filename = savedir + f"/all_models_{ar.data_name}_{ar.method}_{ar.epochs}" + f"grad_{ar.data_name}_prior_var=={var}_sel_rate=={sel_rate}.pt"

        grd = torch.load(filename).to(ar.device)

        grad[:, sel_feat_ind,i] = grd

        if i==len(selection_rate)-1:
            sel_rate = 0.7
            sel_feat_dim = int(sel_rate * feat_dim)  # selected feature dimension
            sel_feat_ind = indices[0:sel_feat_dim]
            sel_feat_ind.to('cpu')

            map = Linear(in_features=sel_feat_dim, out_features=numb_classes, bias=False).to(ar.device)
            map.load_state_dict(torch.load(filename_map))
            map = map.weight  # numb_classes by tot_feat_dim
            map_model[:, sel_feat_ind] = map

    ## normalizing the weights ##
    print(f'unnormalized weights are {weight_diag}')
    # weight_diag = weight_diag / torch.sum(weight_diag)
    # print(f'BMA weights are {weight_diag}')

    # torch.save(weight_diag, home_dir + '/weight.pt')
    # torch.save(selection_rate, home_dir + '/selection_rate.pt')





    ################## Now evaluation starts ###################
    test_accs_top1_mle = []
    test_accs_top1_map = []
    test_accs_top1_average_weight = []
    test_accs_top1_average_logits = []
    test_accs_top1_fisher_merging = []
    test_accs_top1_bma_diag = []

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
        # project_labels = getattr(dataset, 'project_labels', None)
        # if hasattr(dataset, 'project_labels'):
        #     target = dataset.project_labels(target, device)
        # if projection_fn is not None:
        #     logits = projection_fn(logits, device) # map from imagenet to objectnet classes

    ### method 1: average weights ###
    average_weight = torch.mean(linear_model_tot,dim=2).to(ar.device)
    tot_num_test_evaluated = 0

    for i in range(how_many_steps_in_test):

        if i == (how_many_steps_in_test - 1):
            feat1 = feat_test[ar.batch_size * i:, :]
            labels = label_test[ar.batch_size * i:]
        else:
            feat1 = feat_test[ar.batch_size * i:ar.batch_size * (i + 1), :]
            labels = label_test[ar.batch_size * i:ar.batch_size * (i + 1)]

        print(labels)
        tot_num_test_evaluated = tot_num_test_evaluated + feat1.shape[0]

        feat1 = feat1.to(ar.device)
        labels = labels.to(ar.device)

        ## MLE
        if ar.data_name == 'imagenet_a':
            from imagenet_a import CLASS_SUBLIST
            mle_sel = mle[CLASS_SUBLIST, :]
            outputs = torch.mm(feat1, mle_sel.t())
        elif ar.data_name == 'imagenet_r':
            from imagenet_r import CLASS_SUBLIST
            mle_sel = mle[CLASS_SUBLIST, :]
            outputs = torch.mm(feat1, mle_sel.t())
        elif ar.data_name == 'objectnet':
            outputs = torch.mm(feat1, mle.t()) # bs x 1000
            outputs = projection_fn(outputs.detach(), ar.device) # bs x 113
            # labels = project_labels(labels.detach, ar.deviace)
            # if hasattr(dataset, 'project_labels'):
            #     target = dataset.project_labels(target, device)
        else:
            outputs = torch.mm(feat1, mle.t())

        top1_acc = accuracy(outputs, labels, topk=(1,))
        test_accs_top1_mle.append(top1_acc[0])

        ## Best MAP
        if ar.data_name == 'imagenet_a':
            from imagenet_a import CLASS_SUBLIST
            mle_sel = map_model[CLASS_SUBLIST, :]
            outputs = torch.mm(feat1, mle_sel.t())
        elif ar.data_name == 'imagenet_r':
            from imagenet_r import CLASS_SUBLIST
            mle_sel = map_model[CLASS_SUBLIST, :]
            outputs = torch.mm(feat1, mle_sel.t())
        elif ar.data_name == 'objectnet':
            outputs = torch.mm(feat1, map_model.t()) # bs x 1000
            outputs = projection_fn(outputs.detach(), ar.device) # bs x 113
            # labels = project_labels(labels.detach, ar.deviace)
            # if hasattr(dataset, 'project_labels'):
            #     target = dataset.project_labels(target, device)
        else:
            outputs = torch.mm(feat1, map_model.t())

        top1_acc = accuracy(outputs, labels, topk=(1,))
        test_accs_top1_map.append(top1_acc[0])






        if ar.data_name=='imagenet_a' or ar.data_name=='imagenet_r':
            outputs = torch.mm(feat1, average_weight[:,CLASS_SUBLIST]) # classifier has to be
        else:
            outputs = torch.mm(feat1, average_weight)

        if ar.data_name == 'objectnet':
            outputs = projection_fn(outputs.detach(), ar.device) # bs x 113

        top1_acc = accuracy(outputs, labels, topk=(1,))
        test_accs_top1_average_weight.append(top1_acc[0])

        ### method 2: average output probability (ensemble) ###
        if ar.data_name == 'imagenet_a' or ar.data_name == 'imagenet_r':
            output_tot = torch.zeros(feat1.shape[0], len(CLASS_SUBLIST), len(selection_rate))
        elif ar.data_name == 'objectnet':
            output_tot = torch.zeros(feat1.shape[0], 113, len(selection_rate))
        else:
            output_tot = torch.zeros(feat1.shape[0], numb_classes, len(selection_rate))
        for i in range(len(selection_rate)):
            model = linear_model_tot[:,:,i]
            model = model.to(ar.device)
            if ar.data_name == 'imagenet_a' or ar.data_name == 'imagenet_r':
                output_model = torch.mm(feat1, model[:,CLASS_SUBLIST])
            else:
                output_model = torch.mm(feat1, model)

            if ar.data_name == 'objectnet':
                output_model = projection_fn(output_model.detach(), ar.device)

            softmax_output = m(output_model)
            output_tot[:, :, i] = softmax_output

        outputs = torch.mean(output_tot, dim=2).to(ar.device)
        top1_acc = accuracy(outputs, labels, topk=(1,))
        test_accs_top1_average_logits.append(top1_acc[0])

        ### method 3: Fisher merging ###
        ### change this part to gradient**2, rather than hessian
        if ar.data_name == 'imagenet_a' or ar.data_name == 'imagenet_r':
            reweighted_model = torch.zeros(feat_dim, len(CLASS_SUBLIST), len(selection_rate), device=ar.device)
        else:
            reweighted_model = torch.zeros(feat_dim, numb_classes, len(selection_rate), device=ar.device)
        for i in range(len(selection_rate)):
            model = linear_model_tot[:,:,i]
            model = model.to(ar.device) # feat_dim by output_dim
            if ar.data_name == 'imagenet_a' or ar.data_name == 'imagenet_r':
                # reweighted_model[:, :, i] = diag_h[CLASS_SUBLIST, :, i].t() * model[:,CLASS_SUBLIST]
                reweighted_model[:, :, i] = grad[CLASS_SUBLIST, :, i].t() * model[:, CLASS_SUBLIST]
            else:
                # reweighted_model[:,:,i] = diag_h[:,:,i].t() * model
                reweighted_model[:, :, i] = grad[:, :, i].t() * model

        if ar.data_name == 'imagenet_a' or ar.data_name == 'imagenet_r':
            # sum_h = torch.sum(diag_h[CLASS_SUBLIST,:], dim=2) + 1e-6 # to avoid numerical errors
            sum_h = torch.sum(grad[CLASS_SUBLIST, :], dim=2) + 1e-6  # to avoid numerical errors
        else:
            # sum_h = torch.sum(diag_h, dim=2) + 1e-6  # to avoid numerical errors
            sum_h = torch.sum(grad, dim=2) + 1e-6  # to avoid numerical errors
        fisher_merging = torch.sum(reweighted_model, dim=2) / sum_h.t()
        fisher_merging = fisher_merging.to(ar.device)
        outputs = torch.mm(feat1, fisher_merging)
        if ar.data_name == 'objectnet':
            outputs = projection_fn(outputs.detach(), ar.device)
        top1_acc = accuracy(outputs, labels, topk=(1,))
        test_accs_top1_fisher_merging.append(top1_acc[0])

        ### method 4: BMA with diag H ###
        if ar.data_name == 'imagenet_a' or ar.data_name == 'imagenet_r':
            output_tot_diag = torch.zeros(feat1.shape[0], len(CLASS_SUBLIST), len(selection_rate))
        elif ar.data_name =='objectnet':
            output_tot_diag = torch.zeros(feat1.shape[0], 113, len(selection_rate))
        else:
            output_tot_diag = torch.zeros(feat1.shape[0], numb_classes, len(selection_rate))
        # output_tot_kp = torch.zeros(feat1.shape[0], numb_classes, len(prior_vals))
        for i in range(len(selection_rate)):
            model = linear_model_tot[:,:,i]
            model = model.to(ar.device)
            if ar.data_name == 'imagenet_a' or ar.data_name == 'imagenet_r':
                output_model = torch.mm(feat1, model[:,CLASS_SUBLIST])
            else:
                output_model = torch.mm(feat1, model)

            if ar.data_name =='objectnet':
                output_model = projection_fn(output_model.detach(), ar.device)

            softmax_output = m(output_model)
            # softmax_output = output_model
            # softmax_output = torch.mm(feat1, model)
            output_tot_diag[:, :, i] = softmax_output * weight_diag[i]
            # output_tot_kp[:, :, i] = softmax_output * weight_kp[i]

        outputs_diag = torch.mean(output_tot_diag, dim=2).to(ar.device)
        top1_acc = accuracy(outputs_diag, labels, topk=(1,))
        test_accs_top1_bma_diag.append(top1_acc[0])

        # outputs_kp = torch.mean(output_tot_kp, dim=2).to(ar.device)
        # top1_acc = accuracy(outputs_kp, labels, topk=(1,))
        # test_accs_top1_bma_kp.append(top1_acc[0])

    # print(f'top1 validation acc of mle is {np.mean(test_accs_top1_mle):.3f}')
    # print(f'top1 validation acc of best map is {np.mean(test_accs_top1_map):.3f}')
    # print(f'top1 validation acc of average weight is {np.mean(test_accs_top1_average_weight):.3f}')
    # print(f'top1 validation acc of average logits is {np.mean(test_accs_top1_average_logits):.3f}')
    # print(f'top1 validation acc of fisher merging is {np.mean(test_accs_top1_fisher_merging):.3f}')
    # print(f'top1 validation acc of bma diag is {np.mean(test_accs_top1_bma_diag):.3f}')

    # print(f'tot_num_test_evaluated is {tot_num_test_evaluated} and the total test datapoints are {tot_test}')
    print(f'{np.mean(test_accs_top1_mle):.3f}')
    print(f'{np.mean(test_accs_top1_map):.3f}')
    print(f'{np.mean(test_accs_top1_average_weight):.3f}')
    print(f'{np.mean(test_accs_top1_average_logits):.3f}')
    print(f'{np.mean(test_accs_top1_fisher_merging):.3f}')
    print(f'{np.mean(test_accs_top1_bma_diag):.3f}')

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
