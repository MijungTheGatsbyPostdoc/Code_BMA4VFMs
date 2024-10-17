# to create a dataloader that contains features and labels
# to expedite the linear weight training

import torch
import open_clip
import os
import torchvision.datasets as dset
import argparse
import time
from torchvision.datasets import Flowers102
from wilds import get_dataset
from mmpretrain.datasets import SUN397
from PIL import Image
from imagenetv2_pytorch import ImageNetV2Dataset

def maybe_dictionarize_batch(batch):
    if isinstance(batch, dict):
        return batch
    if len(batch) == 2:
        return {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        return {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

def model_to_feats(model, data, device):
    image_features = []
    image_labels = []

    data_len = data[0].shape[0]
    image_tensor = data[0] # bs x 3 x 32 x 32
    label_tensor = data[1] # bs

    for i in range(0, data_len):
        image = image_tensor[i]
        image_input = image.to(device)
        image_input = image_input[None, :, :, :] # it has to start with batch dimension

        class_id = label_tensor[i]

        with torch.no_grad():
            image_feature = model.encode_image(image_input)

        image_feature /= image_feature.norm()
        image_features.append(image_feature)
        image_labels.append(class_id)

    image_features = torch.stack(image_features, dim=1).to(device)
    image_features = image_features.squeeze()  # 10,000 by 1024

    image_labels = torch.stack(image_labels).to(device)

    return image_features, image_labels

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-name', type=str, default='imagenet', help='target dataset')
    parser.add_argument('--batch-size', '-bs', type=int, default=1000, help='batch size during training linear weights')
    parser.add_argument('--device', type=str, default='cuda:1', help='gpu or cpu')
    # parser.add_argument('--model-name', type=str, default='ViT-H-14-quickgelu', help='which VFM to load?')
    # parser.add_argument('--pretrained-with', type=str, default='dfn5b', help='which dataset is VFM trained with?')
    parser.add_argument('--seed', type=int, default=0, help='sets random seed')

    ar = parser.parse_args()

    # preprocess_args(ar)
    return ar



def main(**kwargs):

    ar = get_args()
    # print(ar)
    # torch.manual_seed(ar.seed) # test this if this changes anything in the order of the dataloader.
    home_dir = os.getcwd() + '/feature_representations'

    if ar.data_name=='imagenet':

        # model_names= ['ViT-H-14-378-quickgelu', 'ViT-H-14-quickgelu', 'EVA02-E-14-plus', 'ViT-SO400M-14-SigLIP-384']
        # datasets_pretrained_with = ['dfn5b', 'dfn5b', 'laion2b_s9b_b144k', 'webli']

        # model_names= ['ViT-bigG-14-CLIPA-336', 'EVA02-E-14', 'ViT-H-14-quickgelu', 'convnext_xxlarge']
        # datasets_pretrained_with = ['datacomp1b', 'laion2b_s4b_b115k', 'metaclip_fullcc','laion2b_s34b_b82k_augreg_soup']

        # model_names= ['ViT-SO400M-14-SigLIP-384']
        # datasets_pretrained_with = ['webli']
        model_names= ['EVA02-E-14-plus']
        datasets_pretrained_with = ['laion2b_s9b_b144k']

        numb_candidates = len(model_names)

        for i in range(numb_candidates):
            # print(f'{i}th model: {ar.data_name}')

            model_name = model_names[i]
            pretrained_with = datasets_pretrained_with[i]

            print(f'Feature extraction of {ar.data_name} starts for model:{model_name} pretrained with {pretrained_with}')

            savedir = f"/prepro_dataset={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}"
            savedir = home_dir + savedir
            os.makedirs(savedir, exist_ok=True)

            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_with,
                                                                         device=ar.device)
            model.requires_grad_(False)

            ## training data ##
            datadir_train = '../databank/imagenet/train/'  # 1,281,167 training datapoints
            dataset_train = dset.ImageFolder(root=datadir_train,
                                       transform=preprocess)

            g_cpu = torch.Generator()
            g_cpu.manual_seed(ar.seed)
            train_data = torch.utils.data.DataLoader(dataset_train, batch_size=ar.batch_size,
                                                  shuffle=False, generator=g_cpu, num_workers=int(1))

            tot_train = 1281167
            feat_dim = 1024 # if ['EVA02-E-14-plus']
            feat_mat = torch.zeros(tot_train, feat_dim)
            # last_step = int(tot_train / ar.batch_size)
            # label_mat = []
            for i, data in enumerate(train_data, 1):
                start = time.time()
                print(f'{i}th data')
                feats, labels = model_to_feats(model, data, ar.device)
                end = time.time()
                print('time for forward pass:', end - start)
                # feat_mat.append(feats)
                start_idx = (i-1)*ar.batch_size
                end_idx = start_idx + feats.shape[0]
                print(f'start ind:{start_idx} and end_ind:{end_idx}')
                feat_mat[start_idx:end_idx,:] = feats
                # label_mat.append(labels)
            torch.save(feat_mat,
                       savedir + f"/feat_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            # torch.save(label_mat,
            #            savedir + f"/labels_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            del train_data, feat_mat, model

            print(f'Feature extraction complete for model:{model_name} pretrained with {pretrained_with}')


            ## test data ##
            # datadir_test = '../databank/imagenet/CLS-LOC/test'
            # dataset_test = dset.ImageFolder(root=datadir_test,
            #                                transform=preprocess)
            # test_data = torch.utils.data.DataLoader(dataset_test, batch_size=ar.batch_size,
            #                                        shuffle=True, num_workers=int(1))
            #
            # feat_mat = []
            # label_mat = []
            # for i, data in enumerate(test_data, 1):
            #     # start = time.time()
            #     # print(f'{i}th data')
            #     feats, labels = model_to_feats(model, data, ar.device)
            #     # end = time.time()
            #     # print('time for forward pass:', end - start)
            #     feat_mat.append(feats)
            #     label_mat.append(labels)
            # torch.save(feat_mat,
            #            savedir + f"/feat_test_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            # torch.save(label_mat,
            #            savedir + f"/labels_test_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            #
            # del model, feat_mat, label_mat, test_data, dataset_test

            # ## validation data ##
            # datadir_val = '../databank/imagenet/val/'
            # dataset_val = dset.ImageFolder(root=datadir_val,
            #                                transform=preprocess)
            #
            # g_cpu = torch.Generator()
            # g_cpu.manual_seed(ar.seed)
            # val_data = torch.utils.data.DataLoader(dataset_val, batch_size=ar.batch_size,
            #                                        shuffle=False, generator=g_cpu, num_workers=int(1))
            #
            # feat_mat = []
            # label_mat = []
            # start = time.time()
            # for i, data in enumerate(val_data, 1):
            #     # start = time.time()
            #     feats, labels = model_to_feats(model, data, ar.device)
            #     feat_mat.append(feats)
            #     label_mat.append(labels)
            #     # end = time.time()
            #     # print('time for forward pass:', end - start)
            #
            # end = time.time()
            # print(f'time for forward pass of validation set of imagenet for model {model_name}:{end - start}')
            #
            # torch.save(feat_mat,
            #            savedir + f"/feat_val_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            # torch.save(label_mat,
            #            savedir + f"/labels_val_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            #
            # del model, feat_mat, label_mat, dataset_val, val_data

    elif ar.data_name=='flowers102':

        savedir = f"/prepro_dataset={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}"
        savedir = home_dir + savedir
        os.makedirs(savedir, exist_ok=True)

        home_dir = os.getcwd() + '/feature_representations'
        savedir = f"/prepro_dataset={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}"
        savedir = home_dir + savedir
        os.makedirs(savedir, exist_ok=True)

        home_dir = os.getcwd() + '/feature_representations'
        savedir = f"/prepro_dataset={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}"
        savedir = home_dir + savedir
        os.makedirs(savedir, exist_ok=True)

        # load feature extractor(s)
        model, _, preprocess = open_clip.create_model_and_transforms(ar.model_name, pretrained=ar.pretrained_with,
                                                                     device=ar.device)
        model.requires_grad_(False)

        train_data = Flowers102(root=os.path.expanduser("~/.cache"), download=True) # tot_train = 1020
        test_data = Flowers102(root=os.path.expanduser("~/.cache"), download=True, split='val')

        ## training data ##
        print('training data')
        image_features = []
        image_labels = []
        for image, class_id in train_data:
            image_input = preprocess(image).unsqueeze(0).to(ar.device)
            with torch.no_grad():
                # start = time.time()
                image_feature = model.encode_image(image_input)
                # end = time.time()
                # print('time for forward pass:', end - start)
            image_feature /= image_feature.norm()
            image_features.append(image_feature)
            image_labels.append(class_id)
        image_features = torch.stack(image_features, dim=1).to(ar.device)
        image_features = image_features.squeeze()  # this becomes datapoints x feature_dim
        torch.save(image_features,
                   savedir + f"/feat_data={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}.pt")
        torch.save(image_labels,
                   savedir + f"/labels_data={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}.pt")

        ## validation data ##
        print('validation data')
        image_features = []
        image_labels = []
        for image, class_id in test_data:
            image_input = preprocess(image).unsqueeze(0).to(ar.device)
            with torch.no_grad():
                image_feature = model.encode_image(image_input)
            image_feature /= image_feature.norm()
            image_features.append(image_feature)
            image_labels.append(class_id)
        image_features = torch.stack(image_features, dim=1).to(ar.device)
        image_features = image_features.squeeze()  # this becomes 10,000 x 1024
        torch.save(image_features,
                   savedir + f"/feat_val_data={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}.pt")
        torch.save(image_labels,
                   savedir + f"/labels_val_data={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}.pt")

    elif ar.data_name == 'camelyon17':

        model_names = ['ViT-H-14-378-quickgelu','ViT-bigG-14-CLIPA-336','ViT-SO400M-14-SigLIP-384','EVA02-E-14','ViT-H-14-quickgelu']
        datasets_pretrained_with = ['dfn5b','datacomp1b','webli','laion2b_s4b_b115k','metaclip_fullcc']

        numb_candidates = len(model_names)

        dataset = get_dataset(dataset="camelyon17", download=True)
        train_data = dataset.get_subset("train")
        # test_data = dataset.get_subset("test")
        val_data = dataset.get_subset("val")

        for i in range(numb_candidates):

            print(f'{i}th model: {ar.data_name}')
            model_name = model_names[i]
            pretrained_with = datasets_pretrained_with[i]

            savedir = f"/prepro_dataset={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}"
            savedir = home_dir + savedir
            os.makedirs(savedir, exist_ok=True)

            # load feature extractor(s)
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_with,
                                                                         device=ar.device)
            model.requires_grad_(False)

            ## training data ##
            print('training data')
            image_features = []
            image_labels = []
            for data_i in train_data:
                image = data_i[0]
                class_id = data_i[1]

                image_input = preprocess(image).unsqueeze(0).to(ar.device)
                with torch.no_grad():
                    # start = time.time()
                    image_feature = model.encode_image(image_input)
                    # end = time.time()
                    # print('time for forward pass of a single datapoint:', end - start)
                image_feature /= image_feature.norm()
                image_features.append(image_feature)
                image_labels.append(class_id)
            image_features = torch.stack(image_features, dim=1).to(ar.device)
            image_features = image_features.squeeze()  # this becomes 10,000 x 1024
            torch.save(image_features,
                       savedir + f"/feat_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            torch.save(image_labels,
                       savedir + f"/labels_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            del image_features, image_labels

            # ## training data ##
            # print('test data')
            # image_features = []
            # image_labels = []
            # for data_i in test_data:
            #     image = data_i[0]
            #     class_id = data_i[1]
            #
            #     image_input = preprocess(image).unsqueeze(0).to(ar.device)
            #     with torch.no_grad():
            #         # start = time.time()
            #         image_feature = model.encode_image(image_input)
            #         # end = time.time()
            #         # print('time for forward pass of a single datapoint:', end - start)
            #     image_feature /= image_feature.norm()
            #     image_features.append(image_feature)
            #     image_labels.append(class_id)
            # image_features = torch.stack(image_features, dim=1).to(ar.device)
            # image_features = image_features.squeeze()  # this becomes 10,000 x 1024
            # torch.save(image_features,
            #            savedir + f"/feat_test_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            # torch.save(image_labels,
            #            savedir + f"/labels_test_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            # del image_features, image_labels


            ## validation data ##
            print('validation data')
            image_features = []
            image_labels = []
            for data_i in val_data:
                image = data_i[0]
                class_id = data_i[1]

                image_input = preprocess(image).unsqueeze(0).to(ar.device)
                with torch.no_grad():
                    # start = time.time()
                    image_feature = model.encode_image(image_input)
                    # end = time.time()
                    # print('time for forward pass of a single datapoint:', end - start)
                image_feature /= image_feature.norm()
                image_features.append(image_feature)
                image_labels.append(class_id)
            image_features = torch.stack(image_features, dim=1).to(ar.device)
            image_features = image_features.squeeze()  # this becomes 10,000 x 1024

            torch.save(image_features,
                       savedir + f"/feat_val_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            torch.save(image_labels,
                       savedir + f"/labels_val_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")

            del model, image_features, image_labels

    elif ar.data_name == 'sun397':

        # First do this, so it's downloaded, and then untar in command
        # from torchvision.datasets import SUN397
        # data = SUN397(root="../databank/", download=True)

        train_data = SUN397(data_root='../databank/SUN397', split='train')
        test_data = SUN397(data_root='../databank/SUN397', split='test')

        # model_names = ['ViT-H-14-378-quickgelu','ViT-bigG-14-CLIPA-336','ViT-SO400M-14-SigLIP-384','EVA02-E-14','ViT-H-14-quickgelu']
        # datasets_pretrained_with = ['dfn5b','datacomp1b','webli','laion2b_s4b_b115k','metaclip_fullcc']
        model_names = ['ViT-H-14-quickgelu']
        datasets_pretrained_with = ['metaclip_fullcc']

        numb_candidates = len(model_names)

        for i in range(numb_candidates):

            print(f'{i}th model: {ar.data_name}')
            model_name = model_names[i]
            pretrained_with = datasets_pretrained_with[i]

            savedir = f"/prepro_dataset={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}"
            savedir = home_dir + savedir
            os.makedirs(savedir, exist_ok=True)

            # load feature extractor(s)
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_with,
                                                                         device=ar.device)
            model.requires_grad_(False)

            ## training data ##
            print('training data')
            image_features = []
            image_labels = []
            for data_i in train_data:
                image = Image.open(data_i['img_path'])
                class_id = data_i['gt_label']

                image_input = preprocess(image).unsqueeze(0).to(ar.device)
                with torch.no_grad():
                    image_feature = model.encode_image(image_input)
                image_feature /= image_feature.norm()
                image_features.append(image_feature)
                image_labels.append(class_id)
            image_features = torch.stack(image_features, dim=1).to(ar.device)
            image_features = image_features.squeeze()  # this becomes 10,000 x 1024

            torch.save(image_features,
                       savedir + f"/feat_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            torch.save(image_labels,
                       savedir + f"/labels_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            del image_features, image_labels

            ## validation data ##
            print('validation data')
            image_features = []
            image_labels = []
            for data_i in test_data:
                image = Image.open(data_i['img_path'])
                class_id = data_i['gt_label']

                image_input = preprocess(image).unsqueeze(0).to(ar.device)
                with torch.no_grad():
                    # start = time.time()
                    image_feature = model.encode_image(image_input)
                    # end = time.time()
                    # print('time for forward pass of a single datapoint:', end - start)
                image_feature /= image_feature.norm()
                image_features.append(image_feature)
                image_labels.append(class_id)
            image_features = torch.stack(image_features, dim=1).to(ar.device)
            image_features = image_features.squeeze()  # this becomes 10,000 x 1024

            torch.save(image_features,
                       savedir + f"/feat_val_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            torch.save(image_labels,
                       savedir + f"/labels_val_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")

            del image_features, image_labels, model

    elif ar.data_name=='imagenet_v2':

        # from imagenetv2_pytorch import ImageNetV2Dataset
        # Download the data from https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main

        # model_names = ['ViT-H-14-378-quickgelu','ViT-bigG-14-CLIPA-336','ViT-SO400M-14-SigLIP-384','EVA02-E-14','ViT-H-14-quickgelu']
        # datasets_pretrained_with = ['dfn5b','datacomp1b','webli','laion2b_s4b_b115k','metaclip_fullcc']

        model_names = [ar.model_name]
        datasets_pretrained_with = [ar.pretrained_with]

        numb_candidates = len(model_names)

        for i in range(numb_candidates):
            print(f'{i}th model: {ar.data_name}')
            model_name = model_names[i]
            pretrained_with = datasets_pretrained_with[i]

            savedir = f"/prepro_dataset={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}"
            savedir = home_dir + savedir
            os.makedirs(savedir, exist_ok=True)

            # load feature extractor(s)
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_with,
                                                                         device=ar.device)
            model.requires_grad_(False)

            # imagenet_v2 dataset has three subsets, matched-frequency, threshold-0.7, and top-images variants. Each of these contain 10,000 samples.
            dataset_1 = ImageNetV2Dataset("matched-frequency",
                                          transform=preprocess)
            # dataset_2 = ImageNetV2Dataset("threshold-0.7", transform=preprocess)
            # dataset_3 = ImageNetV2Dataset("top-images", transform=preprocess)
            # dataset = torch.utils.data.ConcatDataset([dataset_1, dataset_2, dataset_3])

            # g_cpu = torch.Generator()
            # g_cpu.manual_seed(ar.seed)
            # loader = torch.utils.data.DataLoader(dataset_1, batch_size=ar.batch_size, generator=g_cpu)

            g_cpu = torch.Generator()
            g_cpu.manual_seed(ar.seed)
            loader = torch.utils.data.DataLoader(dataset_1, batch_size=ar.batch_size,
                                                   shuffle=False, generator=g_cpu, num_workers=int(1))

            # images = ImageNetV2Dataset(transform=preprocess)
            # loader = torch.utils.data.DataLoader(images, batch_size=ar.batch_size, num_workers=1)

            feat_mat = []
            label_mat = []
            for i, data in enumerate(loader, 1):
                # print(f'{i}th val data')
                feats, labels = model_to_feats(model, data, ar.device)
                feat_mat.append(feats)
                label_mat.append(labels)
            torch.save(feat_mat,
                       savedir + f"/feat_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            torch.save(label_mat,
                       savedir + f"/labels_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")

            # del model, images, loader, feat_mat, label_mat
            # del model, loader, feat_mat, label_mat, dataset_1, dataset_2, dataset_3, dataset
            del model, loader, feat_mat, label_mat, dataset_1

    elif ar.data_name=='imagenet_sketch':

        # model_names = ['ViT-H-14-378-quickgelu','ViT-bigG-14-CLIPA-336','ViT-SO400M-14-SigLIP-384','EVA02-E-14','ViT-H-14-quickgelu']
        # datasets_pretrained_with = ['dfn5b','datacomp1b','webli','laion2b_s4b_b115k','metaclip_fullcc']

        model_names = [ar.model_name]
        datasets_pretrained_with = [ar.pretrained_with]

        numb_candidates = len(model_names)

        for i in range(numb_candidates):
            print(f'{i}th model: {ar.data_name}')
            model_name = model_names[i]
            pretrained_with = datasets_pretrained_with[i]

            savedir = f"/prepro_dataset={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}"
            savedir = home_dir + savedir
            os.makedirs(savedir, exist_ok=True)

            # load feature extractor(s)
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_with,
                                                                         device=ar.device)
            model.requires_grad_(False)

            datadir = '../databank/imagenet-sketch/sketch/'
            dataset = dset.ImageFolder(root=datadir,
                                       transform=preprocess)
            # train_data = torch.utils.data.DataLoader(dataset, batch_size=ar.batch_size,
            #                                       shuffle=True, num_workers=int(1))
            g_cpu = torch.Generator()
            g_cpu.manual_seed(ar.seed)
            train_data = torch.utils.data.DataLoader(dataset, batch_size=ar.batch_size,
                                                   shuffle=False, generator=g_cpu, num_workers=int(1))
            ## training data ##
            feat_mat = []
            label_mat = []
            for i, data in enumerate(train_data, 1):
                # start = time.time()
                # print(f'{i}th data')
                feats, labels = model_to_feats(model, data, ar.device)
                # end = time.time()
                # print('time for forward pass:', end - start)
                feat_mat.append(feats)
                label_mat.append(labels)
            # torch.save(feat_mat,
            #            savedir + f"/feat_data={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}.pt")
            # torch.save(label_mat,
            #            savedir + f"/labels_data={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}.pt")

            torch.save(feat_mat,
                       savedir + f"/feat_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            torch.save(label_mat,
                       savedir + f"/labels_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")

            del model, dataset, train_data, feat_mat, label_mat

    elif ar.data_name=='imagenet_a':

        # model_names = ['ViT-H-14-378-quickgelu','ViT-bigG-14-CLIPA-336','ViT-SO400M-14-SigLIP-384','EVA02-E-14','ViT-H-14-quickgelu']
        # datasets_pretrained_with = ['dfn5b','datacomp1b','webli','laion2b_s4b_b115k','metaclip_fullcc']

        model_names = [ar.model_name]
        datasets_pretrained_with = [ar.pretrained_with]

        numb_candidates = len(model_names)

        for i in range(numb_candidates):
            print(f'{i}th model: {ar.data_name}')
            model_name = model_names[i]
            pretrained_with = datasets_pretrained_with[i]

            savedir = f"/prepro_dataset={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}"
            savedir = home_dir + savedir
            os.makedirs(savedir, exist_ok=True)

            # load feature extractor(s)
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_with,
                                                                         device=ar.device)
            model.requires_grad_(False)

            datadir = '../databank/imagenet-a/'
            dataset = dset.ImageFolder(root=datadir,
                                       transform=preprocess)
            # train_data = torch.utils.data.DataLoader(dataset, batch_size=ar.batch_size,
            #                                       shuffle=True, num_workers=int(1))

            g_cpu = torch.Generator()
            g_cpu.manual_seed(ar.seed)
            train_data = torch.utils.data.DataLoader(dataset, batch_size=ar.batch_size,
                                                   shuffle=False, generator=g_cpu, num_workers=int(1))
            ## training data ##
            feat_mat = []
            label_mat = []
            for i, data in enumerate(train_data, 1):
                # start = time.time()
                # print(f'{i}th data')
                feats, labels = model_to_feats(model, data, ar.device)
                # end = time.time()
                # print('time for forward pass:', end - start)
                feat_mat.append(feats)
                label_mat.append(labels)
            # torch.save(feat_mat,
            #            savedir + f"/feat_data={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}.pt")
            # torch.save(label_mat,
            #            savedir + f"/labels_data={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}.pt")

            torch.save(feat_mat,
                       savedir + f"/feat_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            torch.save(label_mat,
                       savedir + f"/labels_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")

            del model, dataset, train_data, feat_mat, label_mat


    elif ar.data_name=='imagenet_r':


        # model_names = ['ViT-H-14-378-quickgelu','ViT-bigG-14-CLIPA-336','ViT-SO400M-14-SigLIP-384','EVA02-E-14','ViT-H-14-quickgelu']
        # datasets_pretrained_with = ['dfn5b','datacomp1b','webli','laion2b_s4b_b115k','metaclip_fullcc']

        model_names = [ar.model_name]
        datasets_pretrained_with = [ar.pretrained_with]

        numb_candidates = len(model_names)

        for i in range(numb_candidates):
            print(f'{i}th model: {ar.data_name}')
            model_name = model_names[i]
            pretrained_with = datasets_pretrained_with[i]

            savedir = f"/prepro_dataset={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}"
            savedir = home_dir + savedir
            os.makedirs(savedir, exist_ok=True)

            # load feature extractor(s)
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_with,
                                                                         device=ar.device)
            model.requires_grad_(False)

            datadir = '../databank/imagenet-r/'
            dataset = dset.ImageFolder(root=datadir,
                                       transform=preprocess)

            g_cpu = torch.Generator()
            g_cpu.manual_seed(ar.seed)
            train_data = torch.utils.data.DataLoader(dataset, batch_size=ar.batch_size,
                                                   shuffle=False, generator=g_cpu, num_workers=int(1))
            # train_data = torch.utils.data.DataLoader(dataset, batch_size=ar.batch_size,
            #                                       shuffle=True, num_workers=int(1))
            ## training data ##
            feat_mat = []
            label_mat = []
            for i, data in enumerate(train_data, 1):
                # start = time.time()
                # print(f'{i}th {ar.data_name}')
                feats, labels = model_to_feats(model, data, ar.device)
                # end = time.time()
                # print('time for forward pass:', end - start)
                feat_mat.append(feats)
                label_mat.append(labels)
            # torch.save(feat_mat,
            #            savedir + f"/feat_data={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}.pt")
            # torch.save(label_mat,
            #            savedir + f"/labels_data={ar.data_name}_with_model={ar.model_name}_pretrained_with_{ar.pretrained_with}.pt")
            torch.save(feat_mat,
                       savedir + f"/feat_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            torch.save(label_mat,
                       savedir + f"/labels_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")

            del model, dataset, train_data, feat_mat, label_mat

    elif ar.data_name == 'objectnet':

        # model_names = ['ViT-H-14-378-quickgelu','ViT-bigG-14-CLIPA-336','EVA02-E-14']
        # datasets_pretrained_with = ['dfn5b','datacomp1b','laion2b_s4b_b115k']

        # model_names = ['ViT-SO400M-14-SigLIP-384','ViT-H-14-quickgelu']
        # datasets_pretrained_with = ['webli','metaclip_fullcc']

        model_names = [ar.model_name]
        datasets_pretrained_with = [ar.pretrained_with]

        numb_candidates = len(model_names)

        data_dir = '../databank/'
        from model_soups.datasets.objectnet import ObjectNet
        from model_soups.datasets.imagenet_classnames import objectnet_classnames

        label_all = []

        for i in range(numb_candidates):
            print(f'{i}th model: {ar.data_name}')

            model_name = model_names[i]
            pretrained_with = datasets_pretrained_with[i]

            savedir = f"/prepro_dataset={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}"
            savedir = home_dir + savedir
            os.makedirs(savedir, exist_ok=True)

            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_with,
                                                                         device=ar.device)
            model.requires_grad_(False)


            ## data ##
            dataset = ObjectNet(location=data_dir, preprocess=preprocess, batch_size=ar.batch_size, num_workers=int(1),
                                classnames=objectnet_classnames)
            loader = dataset.test_loader

            feat_mat = []
            label_mat = []

            for i, data in enumerate(loader):
                # print(i)
                combined_data = []
                # batch = maybe_dictionarize_batch(data)
                # images = batch['images']
                # target = batch['labels']
                combined_data.append(data['images'])
                combined_data.append(data['labels'])
                # combined_data.append(images)
                # combined_data.append(target)
                feats, labels = model_to_feats(model, combined_data, ar.device)



                # feats, labels = model_to_feats(model, data, ar.device)

                feat_mat.append(feats)
                label_mat.append(labels)

            label_all.append(label_mat)

            torch.save(feat_mat,
                       savedir + f"/feat_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")
            torch.save(label_mat,
                       savedir + f"/labels_data={ar.data_name}_with_model={model_name}_pretrained_with_{pretrained_with}.pt")

            del model, feat_mat, label_mat



#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
