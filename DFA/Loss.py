import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def Entropy(input_):
    return torch.sum(-input_ * torch.log(input_ + 1e-5), dim=1)

def grl_hook(coeff):
    def fun1(grad):
        return - coeff * grad.clone()
    return fun1

def DANN(features, ad_net):

    '''
    Paper Link : https://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation.pdf
    Github Link : https://github.com/thuml/CDAN
    '''

    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    
    if dc_target.is_cuda:
        dc_target = dc_target.cuda()

    return nn.BCELoss()(ad_out, dc_target)

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):

    '''
    Paper Link : https://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation.pdf
    Github Link : https://github.com/thuml/CDAN
    '''

    feature = input_list[0]
    softmax_output = input_list[1].detach()
    
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))

    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    
    if feature.is_cuda:
        dc_target = dc_target.cuda()

    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)

        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy * source_mask

        target_mask = torch.ones_like(entropy)
        target_mask[:feature.size(0)//2] = 0
        target_weight = entropy * target_mask

        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()

        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 

def HAFN(features, weight_L2norm, radius):
    
    '''
    Paper Link : https://arxiv.org/pdf/1811.07456.pdf
    Github Link : https://github.com/jihanyang/AFN
    '''

    return weight_L2norm * (features.norm(p=2, dim=1).mean() - radius) ** 2

def SAFN(features, weight_L2norm, deltaRadius):

    '''
    Paper Link : https://arxiv.org/pdf/1811.07456.pdf
    Github Link : https://github.com/jihanyang/AFN
    '''

    radius = features.norm(p=2, dim=1).detach()

    assert radius.requires_grad == False, 'radius\'s requires_grad should be False'
    
    return weight_L2norm * ((features.norm(p=2, dim=1) - (radius+deltaRadius)) ** 2).mean()

def DFA(feature, source_info):

    '''
    Paper Link: https://ieeexplore.ieee.org/document/7550085
    '''

    def k(x, y, sigma=1):
        distance = torch.squeeze(torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)(x.unsqueeze(0), y.unsqueeze(0)))
        return torch.exp(-distance/(2*sigma*sigma))

    N_s, N_t, numOfPerClass = feature.size(0)//2, feature.size(0)//2, feature.size(0)//14

    # Gist
    gist_loss = k(torch.mean(feature[:N_s], 0), torch.mean(feature[N_s:], 0))    

    # Intra
    intra_loss = 0
    for Class in range(7):
        for i in range(Class * numOfPerClass, (Class+1) * numOfPerClass - 1):
            for j in range(i+1, (Class+1) * numOfPerClass):
                intra_loss += k(feature[i], feature[j]) / (numOfPerClass * (numOfPerClass-1) * 7)

    # Inter
    inter_loss = 0
    for Class in range(7):
        for i in range(Class * numOfPerClass, (Class+1) * numOfPerClass - 1):
            for j in range((Class+1) * numOfPerClass, N_s):
                inter_loss += k(feature[i], feature[j]) / (numOfPerClass * (N_s-numOfPerClass) * 7)

    return gist_loss, intra_loss, inter_loss
