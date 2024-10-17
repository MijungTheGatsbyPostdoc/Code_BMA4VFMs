
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
import shap
import xgboost

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

class linear_model(nn.Module):
    def __init__(self, init_val, device):
        super(linear_model, self).__init__()
        self.parameter = Parameter(torch.Tensor(init_val), requires_grad=True)
        self.device = device
        self.total_num_features, self.output_dim = init_val.shape

    def forward(self, x): # shape(x) == batch_size x feature_dim
        output = torch.mm(x, self.parameter).to(self.device)
        return output




def log_args(log_dir, args):
  """ print and save all args """
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  with open(os.path.join(log_dir, 'args_log'), 'w') as f:
    lines = [' • {:<25}- {}\n'.format(key, val) for key, val in vars(args).items()]
    f.writelines(lines)
  #   for line in lines:
  #     print(line.rstrip())
  # print('-------------------------------------------')

def get_args():
    parser = argparse.ArgumentParser()

    # BASICS
    parser.add_argument('--seed', type=int, default=0, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='imagenet', help='target dataset')
    parser.add_argument('--save-checkpoint', action='store_true', default=True,
                        help='save checkpoint')

    # OPTIMIZATION
    parser.add_argument('--batch-size', '-bs', type=int, default=1000, help='batch size during training linear weights')
    parser.add_argument('--epochs', '-ep', type=int, default=100)
    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu or cpu')

    # Load a single model's features or all models' features
    parser.add_argument('--load-all', action='store_true', default=True,
                        help='load all features of candidate models')
    parser.add_argument('--normalize-features', action='store_true', default=False,
                        help='normalize all features of all models')

    # LOSS SPECIFICATION
    parser.add_argument('--method', type=str, default='map', help='map or mle')
    parser.add_argument('--prior-var', type=float, default=40., help='prior variance')
    parser.add_argument('--selection-rate', type=float, default=1.0, help='1 means select all features, 0.01 means select 1 percent of top MLE features')
    parser.add_argument('--select-by', type=str, default='mag', help='mag (magnitude) or grad (gradient)')

    ar = parser.parse_args()

    preprocess_args(ar)
    return ar


def preprocess_args(ar):
    if ar.seed is None:
        ar.seed = np.random.randint(0, 1000)


def main(**kwargs):

    ar = get_args()
    # print(ar)

    home_dir = os.getcwd() + f"/{ar.data_name}_results"
    if ar.load_all==False:
        savedir =  f"/{ar.model_name}_{ar.data_name}_{ar.method}_{ar.epochs}"
        savedir = home_dir + savedir
        os.makedirs(savedir, exist_ok=True)
        log_args(savedir, ar)

        # Load pre-processed data
        ## to load them
        # print('pre-processed data loading')

        data_dir =  f"feature_representations/prepro_dataset={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}"
        feat_mat = torch.load(data_dir+f"/feat_data={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}.pt",map_location=torch.device(ar.device), weights_only=True)
        label_mat = torch.load(data_dir+f"/labels_data={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}.pt",map_location=torch.device(ar.device), weights_only=True)
        if ar.data_name=='imagenet':
            feat_train = torch.concatenate(feat_mat[:-1]) # [(len(feat_mat) - 1 ) x batch_size] by feature_dim
            feat_train = torch.concatenate((feat_train, feat_mat[-1]), dim=0)
            label_train = torch.concatenate((label_mat[:-1]))
            label_train = torch.concatenate((label_train, label_mat[-1]), dim=0)
        else:
            feat_train = feat_mat
            label_train = torch.tensor(label_mat)
        tot_train = label_train.shape[0]
        # print(f"pre-processed training data loaded. The total number of training data is {tot_train}")

        feat_mat_val = torch.load(data_dir+f"/feat_val_data={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}.pt",map_location=torch.device(ar.device), weights_only=True)
        label_mat_val = torch.load(data_dir+f"/labels_val_data={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}.pt",map_location=torch.device(ar.device), weights_only=True)
        if ar.data_name=='imagenet':
            feat_test = torch.concatenate(feat_mat_val[:-1]) # [(len(feat_mat) - 1 ) x batch_size] by feature_dim
            feat_test = torch.concatenate((feat_test, feat_mat_val[-1]), dim=0)
            label_test = torch.concatenate((label_mat_val[:-1]))
            label_test = torch.concatenate((label_test, label_mat_val[-1]), dim=0)
        else:
            feat_test = feat_mat_val
            label_test = torch.tensor(label_mat_val)
        tot_test = label_test.shape[0]
        # print(f"pre-processed test or validation data loaded. The total number of test or validation data is {tot_test}")

        # prior over the weights in the classifier
        # print('load zeroshot_weights')
        zeroshot_mean = torch.load(
            home_dir + f"/zeroshot_weights/zero_shot_weights_{ar.data_name}_{ar.model_name}_{ar.pretrained_with}.pt",
            map_location=torch.device(ar.device), weights_only=True)

    else: # ar.load_all==True

        savedir =  f"/all_models_{ar.data_name}_{ar.method}_{ar.epochs}"
        savedir = home_dir + savedir
        os.makedirs(savedir, exist_ok=True)
        log_args(savedir, ar)

        # Load pre-processed data
        ## to load them
        # print('pre-processed data loading')
        model_names = ['ViT-H-14-378-quickgelu','ViT-bigG-14-CLIPA-336','ViT-SO400M-14-SigLIP-384','EVA02-E-14','ViT-H-14-quickgelu']
        datasets_pretrained_with = ['dfn5b','datacomp1b','webli','laion2b_s4b_b115k','metaclip_fullcc']

        numb_candidates = len(model_names)

        # to avoid memory overflow
        dim0 = 1024
        dim1 = 1280
        dim2 = 1152
        dim3 = 1024
        dim4 = 1024
        tot_feat_dim = dim0+dim1+dim2+dim3+dim4
        numb_classes = 1000
        tot_train = 1281167
        feat_tot = torch.zeros(tot_train, tot_feat_dim)
        tot_test = 50000
        feat_test_tot = torch.zeros(tot_test, tot_feat_dim)
        zeroshot_weights_tot = torch.zeros(tot_feat_dim,numb_classes)

        feat_dir = os.getcwd()
        feat_dir = feat_dir + f"/feature_representations/all_models_for_{ar.data_name}"

        for i in range(numb_candidates):
            # print(f'{i}th model')
            # i=4
            # print(f'{i}th model')
            model_name = model_names[i]
            pretrained_with = datasets_pretrained_with[i]

            feat_train = torch.load(feat_dir+f"/feat_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            label_train = torch.load(feat_dir+f"/labels_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            # print(f"dimension of features:{feat_train.shape}")

            # tot_train = label_train.shape[0]
            # print(f"pre-processed training data of {i}th model is loaded.")

            feat_test = torch.load(feat_dir+f"/feat_val_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            label_test = torch.load(feat_dir+f"/labels_val_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")


            # prior over the weights in the classifier
            # print('load zeroshot_weights')
            zeroshot_mean = torch.load(
                home_dir + f"/zeroshot_weights/zero_shot_weights_{ar.data_name}_{model_name}_{pretrained_with}.pt",
                map_location=torch.device(ar.device), weights_only=True)

            if i==0:
                feat_tot[:, :dim0]= feat_train
                feat_test_tot[:, :dim0] = feat_test
                zeroshot_weights_tot[0:dim0,:] = zeroshot_mean
            elif i==1:
                feat_tot[:, dim0:dim0+dim1] = feat_train
                feat_test_tot[:, dim0:dim0 + dim1] = feat_test
                zeroshot_weights_tot[dim0:dim0+dim1,:] = zeroshot_mean
            elif i==2:
                feat_tot[:, dim0+dim1:dim0+dim1+dim2] = feat_train
                feat_test_tot[:, dim0 + dim1:dim0 + dim1 + dim2] = feat_test
                zeroshot_weights_tot[dim0 + dim1:dim0 + dim1 + dim2,:] = zeroshot_mean
            elif i==3:
                feat_tot[:, dim0+dim1+dim2:dim0+dim1+dim2+dim3] = feat_train
                feat_test_tot[:, dim0 + dim1 + dim2:dim0 + dim1 + dim2 + dim3] = feat_test
                zeroshot_weights_tot[dim0 + dim1 + dim2:dim0 + dim1 + dim2+dim3, :] = zeroshot_mean
            elif i == 4:
                feat_tot[:, dim0 + dim1 + dim2 + dim3:] = feat_train
                feat_test_tot[:, dim0 + dim1 + dim2 + dim3:] = feat_test
                zeroshot_weights_tot[dim0 + dim1 + dim2 + dim3:, :] = zeroshot_mean

            # print(f"pre-processed test or validation data of {i}th model is loaded.")
            del feat_train, feat_test, zeroshot_mean


        # now we need to stack them together
        feat_train = feat_tot
        feat_test = feat_test_tot
        zeroshot_mean = zeroshot_weights_tot # feature dim by output dim

        del feat_tot, feat_test_tot, zeroshot_weights_tot

        if ar.normalize_features:
            # norm( every datapoint's feature ) == 1
            norm = feat_train.norm(p=2, dim=1, keepdim=True) # tot_train by 1
            feat_train = feat_train.div(norm)

            norm = feat_test.norm(p=2, dim=1, keepdim=True) # tot_test by 1
            feat_test = feat_test.div(norm)
            # for every output dim, weights need to be normalized
            norm = zeroshot_mean.norm(p=2, dim=0, keepdim=True)
            zeroshot_mean = zeroshot_mean.div(norm)

        zeroshot_mean = zeroshot_mean.to(ar.device) #tot_feat_dim by numb_classes




    # Define a classifier
    if ar.selection_rate == 1:
        classifier = Linear(in_features=tot_feat_dim, out_features=numb_classes, bias=False).to(ar.device)
    else:
        sel_feat_dim = int(ar.selection_rate*tot_feat_dim) # selected feature dimension
        # MLE magnitude based selection
        if ar.data_name=='imagenet': # later change where MLE is located
            mle_model = Linear(in_features=tot_feat_dim, out_features=numb_classes, bias=False).to(ar.device)
            mle_dir = f"/all_models_{ar.data_name}_mle_100"
            filename = home_dir + mle_dir + f"/{ar.data_name}_mle_18.pt"
            mle_model.load_state_dict(torch.load(filename))
            mle = mle_model.weight # numb_classes by tot_feat_dim

        # select the feature dimension to train
        if ar.select_by=='grad':
            indices = torch.load(savedir+f'/{ar.data_name}_hessian_based_selection_idx.pt')
            # How the indices computed is below:
            # loss_mle = nn.CrossEntropyLoss()  # default is mean
            # loss_mle = extend(loss_mle)
            # mle_model = extend(mle_model)
            #
            # train_idx = torch.randperm(tot_train)
            # train_idx_per_batch = train_idx[:2*ar.batch_size]
            # feat1 = feat_train[train_idx_per_batch, :]
            # labels = label_train[train_idx_per_batch]
            #
            # feat1 = feat1.to(ar.device)
            # labels = labels.to(ar.device)
            #
            # outputs = mle_model(feat1)
            # loss = loss_mle(outputs, labels)
            # # Now compute gradients:
            # with backpack(DiagHessian()):
            #     loss.backward()
            #
            # for name, param in mle_model.named_parameters():
            #     diag_h = tot_train*param.diag_h # numb_classes by feat_dim
            #
            # ######################  Selection Criterion ############################
            # # store these indices for a later use
            # sorted, indices = torch.sort(torch.sum((mle**2)*diag_h,dim=0), descending=True) # large change, we keep (small change, we discard)
            # torch.save(indices, savedir + f'/{ar.data_name}_hessian_based_selection_idx.pt')
            # #####################################################################
            # del diag_h, loss_mle, mle_model, outputs, feat1, labels
        else:
            ######################  Selection Criterion ############################
            sorted, indices = torch.sort(torch.sum(torch.abs(mle), dim=0), descending=True)
            # sorted, indices = torch.sort(torch.sum((mle**2),dim=0), descending=True)
            #####################################################################


        sel_feat_ind = indices[0:sel_feat_dim]
        classifier = Linear(in_features=sel_feat_dim, out_features=numb_classes, bias=False).to(ar.device)
        zeroshot_mean = zeroshot_mean[sel_feat_ind,:]

    # print('feature dimension:', zeroshot_mean.shape[0])

    # print('training begins')
    optimizer = torch.optim.Adam(classifier.parameters(), lr=ar.lr)
    loss_fn = nn.CrossEntropyLoss() # default is mean

    how_many_steps_in_each_epoch = int(tot_train/ar.batch_size)

    for epoch in range(ar.epochs):

        classifier.train()
        train_losses = []

        # at every epoch, data comes in a different order
        train_idx = torch.randperm(tot_train)

        for i in range(how_many_steps_in_each_epoch):

            optimizer.zero_grad()

            if i==(how_many_steps_in_each_epoch-1): # in case of imagenet: 1281000 + 167
                train_idx_per_batch = train_idx[ar.batch_size*i:]
                # print(f"batch index : from {ar.batch_size * i} till {ar.batch_size * i +  train_idx_per_batch.shape[0]}")
            else:
                train_idx_per_batch = train_idx[ar.batch_size*i:ar.batch_size*(i+1)]
                # print(f"batch index : from {ar.batch_size*i} till {ar.batch_size*(i+1)-1}")

            feat1 = feat_train[train_idx_per_batch, :]
            labels = label_train[train_idx_per_batch]
            feat1 = feat1.to(ar.device)
            labels = labels.to(ar.device)

            if ar.selection_rate !=1:
                feat1 = feat1[:,sel_feat_ind]

            outputs = classifier(feat1)
            neg_log_likelihood = loss_fn(outputs, labels)

            if ar.method=='map':
                neg_log_prior = torch.sum((zeroshot_mean - classifier.weight.t()) ** 2 /(2*ar.prior_var))
                loss = tot_train*neg_log_likelihood + neg_log_prior
                top1_acc = accuracy(outputs, labels, (1,))
                # print(f'neg log like : {tot_train*neg_log_likelihood:.6f} and neg log prior: {neg_log_prior:.6f} and top-1 training accuracy:{top1_acc[0]:.3f}')
            elif ar.method=='mle':
                loss = neg_log_likelihood
                top1_acc = accuracy(outputs, labels, (1,))
                # print(f'loss at training step {i} dataset: {loss.item():.3f} and top-1 training accuracy:{top1_acc[0]:.3f}')
            else:
                print('we do not support other than map and mle at this point')

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())


        # print(f'loss at at epoch {epoch}: {np.mean(train_losses):.3f}')
        if ar.save_checkpoint:
            if ar.method == 'map':
                if ar.select_by == 'grad':
                    torch.save(classifier.state_dict(),
                               savedir + f"/{ar.data_name}_{ar.method}_{epoch}_pv=={ar.prior_var}_select_by={ar.select_by}_sel_rate=={ar.selection_rate}.pt")
                else:
                    torch.save(classifier.state_dict(),
                               savedir + f"/{ar.data_name}_{ar.method}_{epoch}_prior_var=={ar.prior_var}_sel_rate=={ar.selection_rate}.pt")
            else:
                torch.save(classifier.state_dict(), savedir + f"/{ar.data_name}_{ar.method}_{epoch}.pt")


        classifier.eval()
        test_accs_top1 = []

        how_many_steps_in_test = int(tot_test/ar.batch_size)

        for i in range(how_many_steps_in_test):

            if i == (how_many_steps_in_test - 1):
                feat1 = feat_test[ar.batch_size * i:, :]
                labels = label_test[ar.batch_size * i:]
                # print(f"batch index : from {ar.batch_size * i} till {ar.batch_size * i +  feat1.shape[0]}")
            else:
                feat1 = feat_test[ar.batch_size*i:ar.batch_size * (i + 1),:]
                labels = label_test[ar.batch_size*i:ar.batch_size * (i + 1)]
                # print(f"batch index : from {ar.batch_size * i} till {ar.batch_size * (i + 1) - 1}")


            feat1 = feat1.to(ar.device)
            labels = labels.to(ar.device)

            if ar.selection_rate != 1:
                feat1 = feat1[:, sel_feat_ind]

            outputs = classifier(feat1)
            top1_acc = accuracy(outputs, labels, topk=(1, ))  # is there a difference if taking outputs.softmax(dim=-1) then inputting it to acuracy?
            test_accs_top1.append(top1_acc[0])

        if epoch==0:
            best_val_acc_top1 = np.mean(test_accs_top1)
            best_epoch = 0
        else:
            if np.mean(test_accs_top1) > best_val_acc_top1:
                best_val_acc_top1 = np.mean(test_accs_top1)
                best_epoch = epoch

    print(f'at sel_rate={ar.selection_rate} and prior_var={ar.prior_var}, top1:{best_val_acc_top1:.3f} at epoch: {best_epoch}')



#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------