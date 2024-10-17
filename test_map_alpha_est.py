
import torch
import os
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
import argparse
import backpack
from backpack import backpack, extend
from backpack.extensions import (
    DiagHessian, BatchDiagHessian
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


class log_evidence(nn.Module):
    def __init__(self, init_val, device):
        super(log_evidence, self).__init__()

        # init_val: alpha values before ReLU, i.e., values in real
        self.real_alpha = Parameter(torch.Tensor(init_val), requires_grad=True) # numb_classes by tot_feat_dim
        self.device = device
        self.total_num_features, self.output_dim = init_val.shape
        self.relu = nn.ReLU()

    def forward(self, w_map, diag_h, prev_alpha2, eps): # shape(x) == batch_size x feature_dim

        alpha2 = self.relu(self.real_alpha)

        # S = 1/alpha2, so S_inv = alpha2

        L_inv = diag_h - prev_alpha2 # Sig_inv - S_inv
        L_inv[L_inv<0] = eps # output dim by feat dim

        # # new S_inv and S based on new alpha values
        S_inv = alpha2
        S = 1/(alpha2+eps)
        Sig_inv = L_inv + S_inv

        # new w_map based on new alpha values
        m = Sig_inv*w_map/L_inv
        w_map = L_inv*m/Sig_inv # recompute w_map using moment matching

        S_inv_w_map = w_map * alpha2  # output dim by feat dim
        neg_log_likeli_proxy = 0.5*torch.sum(S_inv_w_map**2/L_inv)

        neg_log_prior = 0.5 * torch.sum((w_map ** 2) * alpha2)

        Sig_inv_S = L_inv*S + 1
        logdet_Sig_inv_S = 0.5*torch.sum(torch.log(Sig_inv_S)).to(self.device)

        # neg_log_marginal = neg_log_prior + logdet_Sig_inv_S

        neg_log_marginal = neg_log_likeli_proxy + neg_log_prior + logdet_Sig_inv_S
                            # + torch.sum(alpha2)) # penalize if alpha2 becomes smaller

        return neg_log_marginal, w_map

def get_args():
    parser = argparse.ArgumentParser()

    # BASICS
    parser.add_argument('--seed', type=int, default=0, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='imagenet', help='target dataset')
    parser.add_argument('--save-checkpoint', action='store_true', default=False,
                        help='save checkpoint')

    # OPTIMIZATION
    parser.add_argument('--batch-size', '-bs', type=int, default=1000, help='batch size during training linear weights')
    parser.add_argument('--epochs', '-ep', type=int, default=10)
    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu or cpu')

    parser.add_argument('--eps', type=float, default=1e-3, help='eps to avoid numerical issues')

    # Load a single model's features or all models' features
    parser.add_argument('--which-model-to-use', type=str, default='0',
                        help='Any combination among 0,1,2,3,4,5,6,7')

    parser.add_argument('--normalize-features', action='store_true', default=False,
                        help='normalize all features of all models')

    parser.add_argument('--method', type=str, default='map', help='map or mle')

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
        filename = fnmatch.filter(os.listdir(model_dir + f"/[{indices_for_selected_models[i]}]_models"), 'At_*.pt')
        classifier.load_state_dict(torch.load(model_dir + f"/[{indices_for_selected_models[i]}]_models/" + filename[0]))
        # mle = classifier.weight.detach().clone()

        init_val = torch.rand(numb_classes, tot_feat_dim, device=ar.device)
        log_evid = log_evidence(init_val=init_val, device=ar.device)
        optimizer_log_evid = torch.optim.Adam(log_evid.parameters(), lr=0.05)
        # optimizer_log_evid = torch.optim.Adam(log_evid.parameters(), lr=10*ar.lr)      #maybe low learning rate was the cause of alpha values being stuck in the initial values   #
        alpha2 = init_val

    optimizer = torch.optim.Adam(classifier.parameters(), lr=ar.lr)
    loss_fn = nn.CrossEntropyLoss(reduction='sum') # default is mean

    if ar.method=='map':
        # to use backpack to compute Hessian
        classifier = extend(classifier)
        loss_fn = extend(loss_fn)

    print('training begins')

    how_many_steps_in_each_epoch = int(tot_train/ar.batch_size)

    ########### set up a directory to store results ###########
    home_dir = os.getcwd() + f"/{ar.data_name}_results"
    if numb_candidates==8:
        savedir =  f"/all_{numb_candidates}_models"
    else:
        if ar.method =='map':
            savedir = f"/B_{indices_for_selected_models}_model_{ar.method}"
        else:
            savedir = f"/{indices_for_selected_models}_models"

    savedir = home_dir + savedir
    os.makedirs(savedir, exist_ok=True)
    log_args(savedir, ar)

    ##### one solution is, use backpack's backward to compute Hessian of log likelihood
    ##### but use another optimizer to update w when the loss function includes the neg log prior term

    m = nn.ReLU()

    for epoch in range(ar.epochs):

        classifier.train()

        # at every epoch, data comes in a different order
        train_idx = torch.randperm(tot_train)
        neg_log_likeli = 0
        diag_h = 0

        for i in range(how_many_steps_in_each_epoch):

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

            outputs = classifier(feat1)
            loss = loss_fn(outputs, labels)
            neg_log_prior = 0.5 * torch.sum(classifier.weight ** 2 * alpha2) / how_many_steps_in_each_epoch # S = 1/alpha2

            # print(f'{i}th/{how_many_steps_in_each_epoch}, neg likelihood:{loss} and neg_log_prior:{neg_log_prior}')


            neg_log_likeli = neg_log_likeli + loss.detach().clone()

            loss = loss + neg_log_prior # computing the MAP estimate

            optimizer.zero_grad()
            with backpack(DiagHessian()):
                loss.backward()
            for name, param in classifier.named_parameters():
                diag_h = diag_h + param.diag_h
                # diag_h_batch = param.diag_h_batch
            optimizer.step()


        ############### now update alpha based on the new map
        w_map = classifier.weight.detach().clone()  # batch size by feat_dim
        diag_h = diag_h.detach().clone()

        log_evid.train()
        optimizer_log_evid.zero_grad()
        neg_log_mar, w_map = log_evid(w_map, diag_h, alpha2, ar.eps)
        neg_log_mar.backward()
        optimizer_log_evid.step()  # update alpha

        # print(f'per-training data log marginal likelihood: {-neg_log_mar.data/data_seen_so_far}')
        alpha2 = m(log_evid.real_alpha.detach().clone())

        # diag_h is a posterior precision (diagonal) matrix
        # so if diag_h is large, larger than some threshold,
        # the corresponding w_map goes to zero
        pre_threshold = 5*1e2
        # if torch.max(diag_h) > pre_threshold:
        #     print('diag_h gets large, so we zero out the corresponding elements in w_map')
        #     w_map[diag_h>pre_threshold] = 0

        print(f'after updating, mean(alpha) = {torch.mean(alpha2)}, std(alpha) = {torch.std(alpha2)}, max(alpha) = {torch.max(alpha2)}, min(alpha) = {torch.min(alpha2)}')
        print(f'at epoch {epoch}, how many alphas are large (above {pre_threshold}):{torch.sum(alpha2>pre_threshold)}')
        print(f'at epoch {epoch}, per sample log marginal likelihood: {-neg_log_mar.data/tot_train}')
        log_mar_per_samp = -neg_log_mar.data/tot_train

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

            # outputs = classifier(feat1)
            outputs = torch.mm(feat1, w_map.t())
            top1_acc = accuracy(outputs, labels, topk=(1, ))  # is there a difference if taking outputs.softmax(dim=-1) then inputting it to acuracy?
            test_accs_top1.append(top1_acc[0])

        if epoch==0:
            best_val_acc_top1 = np.mean(test_accs_top1)
            best_epoch = 0
            print(f'at epoch {epoch}, top-1 val accuracy:{best_val_acc_top1:.3f}')
        else:
            if np.mean(test_accs_top1) > best_val_acc_top1:
                best_val_acc_top1 = np.mean(test_accs_top1)
                best_epoch = epoch
                # save
            print(f'at epoch {epoch}: top1-acc {np.mean(test_accs_top1):.3f}')
            print(f'so far the best accuracy is, top1:{best_val_acc_top1:.3f} at epoch: {best_epoch}')

    if ar.save_checkpoint:
        torch.save(w_map, savedir + f"/At_{epoch}.pt")
        torch.save(diag_h, savedir + f"/At_{epoch}_diag_h.pt")
        torch.save(alpha2, savedir + f"/At_{epoch}_alpha2.pt")
        torch.save(neg_log_likeli, savedir + f"/At_{epoch}_neg_log_likeli.pt")
        torch.save(log_mar_per_samp, savedir+f"/At_{epoch}_log_mar_per_samp.pt")

    ######## write the results on a txt file ##########
    filename = 'alpha_est_map_estimate_'+ar.data_name+'model='+ar.which_model_to_use+'.csv'

    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            g = csv.writer(file,delimiter='\t')
            g.writerow(['model_ind','top1_val_acc', 'best epoch'])


    with open(filename, 'a') as file:
        g = csv.writer(file, delimiter='\t')
        g.writerow(['{:}'.format(ar.which_model_to_use),'{:.2f}'.format(best_val_acc_top1), best_epoch])


#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
