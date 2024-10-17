
import torch
import os
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
import argparse
import backpack
from backpack import backpack, extend
from backpack.extensions import (
    DiagHessian,
)
from torch.nn import Linear
import csv
import fnmatch

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

def log_args(log_dir, args):
  """ print and save all args """
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  with open(os.path.join(log_dir, 'args_log'), 'w') as f:
    lines = [' • {:<25}- {}\n'.format(key, val) for key, val in vars(args).items()]
    f.writelines(lines)
    for line in lines:
      print(line.rstrip())
  print('-------------------------------------------')


def get_args():
    parser = argparse.ArgumentParser()

    # BASICS
    parser.add_argument('--seed', type=int, default=0, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='imagenet', help='target dataset')
    parser.add_argument('--save-checkpoint', action='store_true', default=True,
                        help='save checkpoint')

    # OPTIMIZATION
    parser.add_argument('--batch-size', '-bs', type=int, default=2048, help='batch size during training linear weights')
    parser.add_argument('--epochs', '-ep', type=int, default=10)
    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--device', type=str, default='cuda:1', help='gpu or cpu')

    # Load a single model's features or all models' features
    parser.add_argument('--which-model-to-use', type=str, default='4',
                        help='Any combination among 0,1,2,3,4,5,6,7')

    parser.add_argument('--normalize-features', action='store_true', default=False,
                        help='normalize all features of all models')

    parser.add_argument('--method', type=str, default='map', help='map or mle')

    # if method=='map', then below matters.
    parser.add_argument('--pruning', type=str, default='element', help='channel or element')
    parser.add_argument('--prior-var', type=float, default=80, help='prior var')
    parser.add_argument('--selection-rate', type=float, default=0.6, help='1 means select all features, 0.01 means select 1 percent of top MLE features')


    ar = parser.parse_args()

    preprocess_args(ar)
    return ar


def preprocess_args(ar):
    if ar.seed is None:
        ar.seed = np.random.randint(0, 1000)


def main(**kwargs):

    ar = get_args()
    print(ar)

    torch.manual_seed(ar.seed)

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

    if ar.data_name == 'imagenet':
        numb_classes = 1000
        tot_train = 1281167
        tot_test = 50000
    elif ar.data_name=='camelyon17':
        numb_classes = 2
        tot_train = 302436
        tot_test = 34904
    elif ar.data_name =='sun397':
        numb_classes = 397
        tot_train =  19850
        tot_test =  19850
    elif ar.data_name =='flowers102':
        numb_classes = 102
        tot_train = 1020
        tot_test = 1020



    feat_dir = os.getcwd()
    feat_dir = feat_dir + f"/feature_representations/all_models_for_{ar.data_name}"

    feat_tot = torch.zeros(tot_train, tot_feat_dim)
    feat_test_tot = torch.zeros(tot_test, tot_feat_dim)

    for i in range(numb_candidates):
        # print(f'{i}th model')
        # i=4
        print(f'{i}th model')
        model_name = model_names[i]
        pretrained_with = datasets_pretrained_with[i]

        feat_train = torch.load(feat_dir+f"/feat_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
        print(f"dimension of features:{feat_train.shape}")
        print(f"pre-processed training data of {i}th model is loaded.")

        feat_test = torch.load(feat_dir+f"/feat_val_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")

        if i == 0:
            feat_tot[:, 0:dim_tot[0]] = feat_train
            feat_test_tot[:, 0:dim_tot[0]] = feat_test
            print(f'index from 0 to {dim_tot[0]}')
        else:
            feat_tot[:, sum(dim_tot[0:i]):sum(dim_tot[0:(i+1)])] = feat_train
            feat_test_tot[:, sum(dim_tot[0:i]):sum(dim_tot[0:(i + 1)])] = feat_test
            print(f'index from {sum(dim_tot[0:i])} to {sum(dim_tot[0:(i + 1)])}')

        print(f"pre-processed test or validation data of {i}th model is loaded.")
        del feat_train, feat_test

    if ar.normalize_features:
        print('we normalize the features of each datapoint')

        ## memory problem if more than 5 models were used.
        # norm = feat_tot.norm(p=2, dim=1, keepdim=True)  # tot_train by 1
        # feat_tot = feat_tot.div(norm)
        #
        # norm = feat_test_tot.norm(p=2, dim=1, keepdim=True)  # tot_test by 1
        # feat_test_tot = feat_test_tot.div(norm)

        for i in range(tot_train):
            l2norm_val = feat_tot[i,:].norm(p=2)
            feat_tot[i,:] = feat_tot[i,:]/l2norm_val
            # print(feat_tot[i,:].norm(p=2))
        for i in range(tot_test):
            l2norm_val_tst = feat_test_tot[i,:].norm(p=2)
            feat_test_tot[i, :] = feat_test_tot[i, :] / l2norm_val_tst
            # print(feat_test_tot[i,:].norm(p=2))


    ############### load labels ###############
    label_train = torch.load(feat_dir+f"/labels_data={ar.data_name}.pt")
    label_test = torch.load(
        feat_dir + f"/labels_val_data={ar.data_name}.pt")

    # Define a classifier
    classifier = Linear(in_features=tot_feat_dim, out_features=numb_classes, bias=False).to(ar.device)
    if ar.method=='map':
        # we start from mle
        model_dir = os.getcwd() + f"/imagenet_results"
        filename = fnmatch.filter(os.listdir(model_dir + f"/[{indices_for_selected_models[i]}]_model_mle"), 'At_*.pt')
        classifier.load_state_dict(torch.load(model_dir + f"/[{indices_for_selected_models[i]}]_model_mle/" + filename[0]))
        mle = classifier.weight

        ######################  Selection Criterion ############################
        if ar.pruning=='channel':
            sorted, indices = torch.sort(torch.sum(torch.abs(mle), dim=0), descending=True)
            sel_feat_dim = int(ar.selection_rate * tot_feat_dim)  # selected feature dimension
            sel_feat_ind = indices[:, 0:sel_feat_dim]
            classifier = Linear(in_features=sel_feat_dim, out_features=numb_classes, bias=False).to(ar.device)
        elif ar.pruning=='element':
            sorted, indices = torch.sort(torch.abs(mle), descending=True) # sort along the feature dimension
            sel_feat_dim = int(ar.selection_rate * tot_feat_dim)  # selected feature dimension
            sel_feat_ind = indices[:, 0:sel_feat_dim] # selected elements
            unsel_feat_ind = indices[:, sel_feat_dim:] # unselected elements

            # for pruning purpose, need to initialize at zero for unselected elements
            temp = classifier.weight.data.detach().clone()
            for i in range(numb_classes):
                temp[i, unsel_feat_ind[i]] = 0
            classifier.weight.data = temp

            # classifier.weight.data.fill_(0.0)
        #######################################s##############################


    optimizer = torch.optim.Adam(classifier.parameters(), lr=ar.lr)
    loss_fn = nn.CrossEntropyLoss() # default is mean


    print('training begins')

    how_many_steps_in_each_epoch = int(tot_train/ar.batch_size)

    ########### set up a directory to store results ###########
    home_dir = os.getcwd() + f"/{ar.data_name}_results"
    if numb_candidates==8:
        savedir =  f"/all_{numb_candidates}_models"
    else:
        savedir = f"/{indices_for_selected_models}_model_{ar.method}"
        # if ar.method =='map':
        #     savedir = f"/{indices_for_selected_models}_model_{ar.method}"
        # else:
        #     savedir = f"/{indices_for_selected_models}_models"

    savedir = home_dir + savedir
    os.makedirs(savedir, exist_ok=True)
    log_args(savedir, ar)

    for epoch in range(ar.epochs):

        classifier.train()

        # at every epoch, data comes in a different order
        train_idx = torch.randperm(tot_train)

        for i in range(how_many_steps_in_each_epoch):
            # print(f'{i}th step out of {how_many_steps_in_each_epoch}')

            optimizer.zero_grad()

            if i==(how_many_steps_in_each_epoch-1): # in case of imagenet: 1281000 + 167, bs==1000
                train_idx_per_batch = train_idx[ar.batch_size*i:]
                # print(f"batch index : from {ar.batch_size * i} till {ar.batch_size * i +  train_idx_per_batch.shape[0]}")
            else:
                train_idx_per_batch = train_idx[ar.batch_size*i:ar.batch_size*(i+1)]
                # print(f"batch index : from {ar.batch_size*i} till {ar.batch_size*(i+1)-1}")

            feat1 = feat_tot[train_idx_per_batch, :]
            labels = label_train[train_idx_per_batch]
            feat1 = feat1.to(ar.device)
            labels = labels.to(ar.device)

            if ar.method == 'map':
                if ar.pruning=='channel' and ar.selection_rate !=1:
                    feat1 = feat1[:,sel_feat_ind]

            outputs = classifier(feat1)
            loss = loss_fn(outputs, labels)

            if ar.method=='map':

                neg_log_prior_exponent = 0.5 * torch.sum((classifier.weight ** 2) / ar.prior_var)
                loss = loss*tot_train + neg_log_prior_exponent

            loss.backward()
            if ar.method=='map' and ar.pruning == 'element':
                # push gradients back to zero, so the corresponding parameters are not updated from the initial values.
                temp = classifier.weight.grad.detach().clone()
                for i in range(numb_classes):
                    temp[i, unsel_feat_ind[i]] = 0
                classifier.weight.grad.data = temp

            optimizer.step()


        classifier.eval()
        test_accs_top1 = []

        how_many_steps_in_test = int(tot_test/ar.batch_size)

        for i in range(how_many_steps_in_test):

            if i == (how_many_steps_in_test - 1):
                feat1 = feat_test_tot[ar.batch_size * i:, :]
                labels = label_test[ar.batch_size * i:]
                # print(f"batch index : from {ar.batch_size * i} till {ar.batch_size * i +  feat1.shape[0]}")
            else:
                feat1 = feat_test_tot[ar.batch_size*i:ar.batch_size * (i + 1),:]
                labels = label_test[ar.batch_size*i:ar.batch_size * (i + 1)]
                # print(f"batch index : from {ar.batch_size * i} till {ar.batch_size * (i + 1) - 1}")

            feat1 = feat1.to(ar.device)
            labels = labels.to(ar.device)

            if ar.method == 'map':
                if ar.pruning=='channel' and ar.selection_rate !=1:
                    feat1 = feat1[:,sel_feat_ind]

            outputs = classifier(feat1)
            top1_acc = accuracy(outputs, labels, topk=(1, ))  # is there a difference if taking outputs.softmax(dim=-1) then inputting it to acuracy?
            test_accs_top1.append(top1_acc[0])

        if epoch==0:
            best_val_acc_top1 = np.mean(test_accs_top1)
            best_epoch = 0
            print(f'at epoch {epoch}, top-1 val accuracy:{best_val_acc_top1:.3f}')

            if ar.method == 'map' and ar.which_model_to_use == '0' and ar.selection_rate==0.8:
                torch.save(classifier.state_dict(),
                           savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")

            if ar.method == 'map' and ar.which_model_to_use == '5' and ar.selection_rate==0.6:
                torch.save(classifier.state_dict(),
                           savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
            # if ar.save_checkpoint and ar.selection_rate > 0.3:
            #     torch.save(classifier.state_dict(),
            #                savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
        else:
            if np.mean(test_accs_top1) > best_val_acc_top1:
                best_val_acc_top1 = np.mean(test_accs_top1)
                best_epoch = epoch

                if ar.method=='mle': #mle or map
                    # torch.save(classifier.state_dict(), savedir + f"/At_{epoch}_val_acc={best_val_acc_top1}.pt")

                    if ar.which_model_to_use=='0' or ar.which_model_to_use=='1':
                        if epoch==39:
                            torch.save(classifier.state_dict(), savedir + f"/At_{epoch}_val_acc={best_val_acc_top1}.pt")
                    elif ar.which_model_to_use=='2':
                        if epoch==26:
                            torch.save(classifier.state_dict(), savedir + f"/At_{epoch}_val_acc={best_val_acc_top1}.pt")
                    elif ar.which_model_to_use=='3':
                        if epoch==47:
                            torch.save(classifier.state_dict(), savedir + f"/At_{epoch}_val_acc={best_val_acc_top1}.pt")
                    elif ar.which_model_to_use=='4':
                        if epoch==31:
                            torch.save(classifier.state_dict(), savedir + f"/At_{epoch}_val_acc={best_val_acc_top1}.pt")
                    elif ar.which_model_to_use=='5':
                        if epoch==38:
                            torch.save(classifier.state_dict(), savedir + f"/At_{epoch}_val_acc={best_val_acc_top1}.pt")
                    elif ar.which_model_to_use=='6':
                        if epoch==41:
                            torch.save(classifier.state_dict(), savedir + f"/At_{epoch}_val_acc={best_val_acc_top1}.pt")
                    elif ar.which_model_to_use=='7':
                        if epoch==34:
                            torch.save(classifier.state_dict(), savedir + f"/At_{epoch}_val_acc={best_val_acc_top1}.pt")


                elif ar.method=='map':
                    if ar.which_model_to_use == '0':
                        if ar.selection_rate==1.0 or ar.selection_rate==0.9 or ar.selection_rate==0.7:
                            if epoch==4:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.6:
                            if epoch==7:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.5:
                            if epoch==8:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.4:
                            if epoch==9:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                    elif ar.which_model_to_use == '1':
                        if ar.selection_rate==1.0 or ar.selection_rate==0.4:
                            if epoch==5:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.9 or ar.selection_rate==0.8 or ar.selection_rate==0.7:
                            if epoch==6:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.6:
                            if epoch==8:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.5:
                            if epoch==9:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                    elif ar.which_model_to_use == '2':
                        if ar.selection_rate==1.0 or ar.selection_rate==0.9 or ar.selection_rate==0.8:
                            if epoch==4:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.7 or ar.selection_rate==0.6 or ar.selection_rate==0.5:
                            if epoch==8:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.4:
                            if epoch==9:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                    elif ar.which_model_to_use == '3':
                        if ar.selection_rate==1.0:
                            if epoch==2:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.9 or ar.selection_rate==0.8 or ar.selection_rate==0.5:
                            if epoch==8:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.7 or ar.selection_rate==0.6 or ar.selection_rate==0.4:
                            if epoch==7:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                    elif ar.which_model_to_use == '4':
                        if ar.selection_rate==0.7:
                            if epoch==6:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==1.0 or ar.selection_rate==0.9 or ar.selection_rate==0.8:
                            if epoch==1:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.6:
                            if epoch==6:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.4:
                            if epoch==9:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.5:
                            if epoch==7:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")

                    elif ar.which_model_to_use == '5':
                        if ar.selection_rate == 0.8:
                            if epoch == 1:
                                torch.save(classifier.state_dict(),
                                           savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate == 1.0 or ar.selection_rate == 0.9 or ar.selection_rate == 0.7 or ar.selection_rate == 0.5:
                            if epoch == 5:
                                torch.save(classifier.state_dict(),
                                           savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate == 0.4:
                            if epoch == 8:
                                torch.save(classifier.state_dict(),
                                           savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")

                    elif ar.which_model_to_use == '6':
                        if ar.selection_rate==0.7:
                            if epoch==4:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==1.0 or ar.selection_rate==0.9 or ar.selection_rate==0.8 or ar.selection_rate==0.5:
                            if epoch==6:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.6:
                            if epoch==9:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.4:
                            if epoch==8:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")

                    elif ar.which_model_to_use == '7':
                        if ar.selection_rate==0.7 or ar.selection_rate==0.6:
                            if epoch==6:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==1.0 or ar.selection_rate==0.9 or ar.selection_rate==0.8:
                            if epoch==1:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.5:
                            if epoch==4:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")
                        elif ar.selection_rate==0.4:
                            if epoch==9:
                                torch.save(classifier.state_dict(),
                                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")

            print(f'at epoch {epoch}: top1-acc {np.mean(test_accs_top1):.3f}')
            print(f'so far the best accuracy is, top1:{best_val_acc_top1:.3f} at epoch: {best_epoch}')



    # save at the end
    if ar.method=='map' and ar.selection_rate<=0.3:
        torch.save(classifier.state_dict(),
                   savedir + f"/At_{epoch}_selrate={ar.selection_rate}_prvar={ar.prior_var}.pt")

    ######## write the results on a txt file ##########
    filename = f'{ar.method}_estimate_'+ar.data_name+'_model='+ar.which_model_to_use+'.csv'

    if ar.method=='mle':
        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                g = csv.writer(file, delimiter='\t')
                g.writerow(['model_ind', 'bs', 'lr', 'eps', 'top1_val_acc', 'best epoch'])

        with open(filename, 'a') as file:
            g = csv.writer(file, delimiter='\t')
            g.writerow(
                ['{:}'.format(ar.which_model_to_use), ar.batch_size, ar.lr, ar.epochs,
                 '{:.2f}'.format(best_val_acc_top1), best_epoch])
    else:
        if not os.path.exists(filename):
            with open(filename, 'w') as file:
                g = csv.writer(file,delimiter='\t')
                g.writerow(['model_ind','selection rate', 'prior var', 'top1_val_acc', 'best epoch'])


        with open(filename, 'a') as file:
            g = csv.writer(file, delimiter='\t')
            g.writerow(['{:}'.format(ar.which_model_to_use), '{:.2f}'.format(ar.selection_rate), '{:.1f}'.format(ar.prior_var), '{:.2f}'.format(best_val_acc_top1), best_epoch])


#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
