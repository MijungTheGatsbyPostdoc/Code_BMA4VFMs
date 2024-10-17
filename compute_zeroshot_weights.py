# compute zeroshot weights and save them in the correct repository
# to save time, we use pre-processed, feature_representation of validation or test dataset.

import open_clip
from classes_templates import flowers102_classes, camelyon17_classes, sun397_classes, imagenet_classes
from classes_templates import flowers102_templates, camelyon17_templates, sun397_templates, imagenet_templates
import torch
import os
import argparse

@torch.no_grad()
def extract_text_features(class_names, templates, model, tokenizer, device):
    # code borrowed from: https://github.com/openai/CLIP/blob/fcab8b6eb92af684e7ff0a904464be7b99b49b88/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    model.to(device)
    model.eval()

    zeroshot_mean = []
    zeroshot_var = []
    for classname in class_names:
        texts = [template.format(classname) for template in templates]
        texts = tokenizer(texts).to(device)
        class_embeddings = model.encode_text(texts)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # dim=-1 is last dimension, same as dim=1 in this 2d tensor case
        class_embedding_mean = class_embeddings.mean(dim=0) # average across templates
        class_embedding_var = class_embeddings.var(dim=0)  # average across templates
        zeroshot_mean.append(class_embedding_mean)
        zeroshot_var.append(class_embedding_var)
    zeroshot_mean = torch.stack(zeroshot_mean, dim=1).to(device)  # the size of zeroshot_weights is 1024 by 10
    zeroshot_var = torch.stack(zeroshot_var, dim=1).to(device)
    return zeroshot_mean, zeroshot_var

def get_args():
    parser = argparse.ArgumentParser()

    # BASICS
    parser.add_argument('--seed', type=int, default=0, help='sets random seed')
    parser.add_argument('--data-name', type=str, default='imagenet', help='target dataset: camelyon17, flowers102, sun397')
    parser.add_argument('--device', type=str, default='cuda:1', help='gpu or cpu')

    ar = parser.parse_args()

    return ar
def main(**kwargs):

    ar = get_args()
    print(ar)

    home_dir = os.getcwd()
    savedir =  f"/{ar.data_name}_results/zeroshot_weights"
    savedir = home_dir + savedir
    os.makedirs(savedir, exist_ok=True)

    print('computing zeroshot_weights')

    # model_names = ['ViT-H-14-378-quickgelu', 'ViT-bigG-14-CLIPA-336', 'ViT-SO400M-14-SigLIP-384', 'EVA02-E-14',
    #                'ViT-H-14-quickgelu']
    # datasets_pretrained_with = ['dfn5b', 'datacomp1b', 'webli', 'laion2b_s4b_b115k', 'metaclip_fullcc']

    # model_names = ['EVA02-E-14-plus', 'convnext_xxlarge']
    # datasets_pretrained_with = ['laion2b_s9b_b144k', 'laion2b_s34b_b82k_augreg_soup']

    model_names = ['ViT-H-14-quickgelu']
    datasets_pretrained_with = ['dfn5b']

    if ar.data_name == 'flowers102':
        class_map = flowers102_classes
        template_map = flowers102_templates

    elif ar.data_name =='imagenet':
        class_map = imagenet_classes
        template_map = imagenet_templates

    elif ar.data_name == 'camelyon17':
        class_map = camelyon17_classes
        template_map = camelyon17_templates


    elif ar.data_name == 'sun397':
        class_map = sun397_classes
        template_map = sun397_templates


    elif ar.data_name == 'imagenet_r':

        from imagenet_r import CLASS_SUBLIST
        selected_imagenet_classes = []
        for i in range(len(CLASS_SUBLIST)):
            selected_imagenet_classes.append(imagenet_classes[CLASS_SUBLIST[i]])
        class_map = selected_imagenet_classes


        template_map = imagenet_templates

    elif ar.data_name=='imagenet_a':

        from imagenet_a import CLASS_SUBLIST
        selected_imagenet_classes = []
        for i in range(len(CLASS_SUBLIST)):
            selected_imagenet_classes.append(imagenet_classes[CLASS_SUBLIST[i]])
        class_map = selected_imagenet_classes
        template_map = imagenet_templates

    else:
        print('oops, you have to add class names, templates, and candidate model information for your chosen data!')

    print(f'{ar.data_name}: number of classes is {len(class_map)}')

    numb_candidates = len(model_names)

    for i in range(numb_candidates):

        # print(f'{i}th model')
        # i=4
        print(f'{i}th model')
        model_name = model_names[i]
        pretrained_with = datasets_pretrained_with[i]

        # load feature extractor(s)
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_with,
                                                                     device=ar.device)
        model.requires_grad_(False)
        tokenizer = open_clip.get_tokenizer(model_name)

        zeroshot_mean, zeroshot_var = extract_text_features(class_map, template_map, model, tokenizer, ar.device)
        # store the zeroshot weights
        torch.save(zeroshot_mean, savedir + f"/zero_shot_weights_{ar.data_name}_{model_name}_{pretrained_with}.pt")
        del model

    print(f'size of zershot weight: {zeroshot_mean.shape}')


#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------