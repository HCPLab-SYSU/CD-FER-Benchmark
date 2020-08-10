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

def Reweighted_MMD(feature, source_info, target_info):

    '''
    Paper Link : https://arxiv.org/pdf/1904.11150.pdf
    '''

    def k(x, y, sigma=1):
        distance = torch.squeeze(torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)(x.unsqueeze(0), y.unsqueeze(0)))
        # distance = distance * distance
        return torch.exp(-distance/(2*sigma*sigma))

    assert feature.size(0)%14==0, "Feature Size is wrong."
    assert len(source_info)==8 and source_info[0]==(source_info[1]+source_info[2]+source_info[3]+source_info[4]+source_info[5]+source_info[6]+source_info[7]), "Source Info is wrong."
    assert len(target_info)==8 and target_info[0]==(target_info[1]+target_info[2]+target_info[3]+target_info[4]+target_info[5]+target_info[6]+target_info[7]), "Target Info is wrong."

    N_s, N_t, numOfPerClass = feature.size(0)//2, feature.size(0)//2, feature.size(0)//14

    # Compute Alpha
    alpha = [ (target_info[i+1]/target_info[0])/(source_info[i+1]/source_info[0]) for i in range(7)]

    # Compute Alpha MMD Loss
    alpha_MMD = 0
   
    for i in range(N_s-1):
        for j in range(i+1, N_s):
            alpha_MMD+=k(feature[i],feature[j]) * alpha[i//numOfPerClass] * alpha[j//numOfPerClass] / (N_s * (N_s-1))

    for i in range(N_t-1):
        for j in range(i+1, N_t):
            alpha_MMD+=k(feature[i+N_s],feature[j+N_s]) / (N_t * (N_t-1))

    for i in range(N_s):
        for j in range(N_t):
            alpha_MMD-=k(feature[i], feature[j+N_s]) * alpha[i//numOfPerClass] * 2 / (N_s * N_t)

    # Compute Conditional MMD Loss
    conditional_MMD = 0
    
    for Class in range(7):
        for i in range(Class * numOfPerClass, (Class+1) * numOfPerClass - 1):
            for j in range(i+1, (Class+1) * numOfPerClass):
                conditional_MMD+=k(feature[i], feature[j])/ (numOfPerClass * (numOfPerClass-1))

    for Class in range(7):
        for i in range(Class * numOfPerClass, (Class+1) * numOfPerClass - 1):
            for j in range(i+1, (Class+1) * numOfPerClass):
                conditional_MMD+=k(feature[i+N_s], feature[j+N_s])/ (numOfPerClass * (numOfPerClass-1))

    for Class in range(7):
        for i in range(Class * numOfPerClass, (Class+1)* numOfPerClass):
            for j in range(Class * numOfPerClass, (Class+1) * numOfPerClass):
                conditional_MMD+=k(feature[i], feature[j+N_s]) * 2 / (numOfPerClass * numOfPerClass)

    return alpha_MMD, conditional_MMD
