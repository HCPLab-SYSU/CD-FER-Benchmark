import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, MaxPool2d, BatchNorm2d, ReLU

import numpy as np

class CountMeanOfFeature(nn.Module):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        
        super(CountMeanOfFeature, self).__init__()
        
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, input):

        assert self.training==True, 'Use CountMeanOfFreature in eval mode.'

        if self.track_running_stats:  
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

            self.running_mean = exponential_average_factor * torch.mean(input.detach(), 0) + \
                                (1-exponential_average_factor) * self.running_mean

        return input

    def init(self, static_mean):
        self.running_mean = static_mean

    def getSample(self, input):
        return self.running_mean.expand(input.size(0), -1)

class CountMeanAndCovOfFeature(nn.Module):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        
        super(CountMeanAndCovOfFeature, self).__init__()
        
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_cov', torch.zeros(num_features, num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
        self.temple_batch = None

    def forward(self, input):

        assert self.training==True, 'Use CountMeanAndCovOfFreature in eval mode.'

        if self.track_running_stats:  
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

            self.running_mean = exponential_average_factor * torch.mean(input.detach(), 0) + \
                                (1-exponential_average_factor) * self.running_mean

            if self.temple_batch is None:
                self.temple_batch = input.clone().detach()
            else:
                self.temple_batch = torch.cat((self.temple_batch, input.clone().detach()), 0)

            if int(self.num_batches_tracked)!=0 and int(self.num_batches_tracked)%8==0:
                cov_matrix = torch.mm((self.temple_batch-self.running_mean).transpose(0,1), self.temple_batch-self.running_mean) / \
                             (self.temple_batch.size(0)-1)
                self.running_cov  = exponential_average_factor * cov_matrix + \
                                    (1-exponential_average_factor) * self.running_cov
                self.temple_batch = None

        return input
    
    def init(self, static_mean, static_cov):
        self.running_mean = static_mean
        self.running_cov  = static_cov

    def getSample(self, input):

        if self.training:
            result = torch.FloatTensor(np.random.multivariate_normal(mean=self.running_mean.cpu().data.numpy(), 
                                                                     cov=self.running_cov.cpu().data.numpy(), 
                                                                     size=input.size(0))).to('cuda' if torch.cuda.is_available else 'cpu')
            return result

        return self.running_mean.expand(input.size(0), -1)

class CountMeanOfFeatureInCluster(nn.Module):
    def __init__(self, num_features, class_num=7, momentum=0.1, track_running_stats=True):
        super(CountMeanOfFeatureInCluster, self).__init__()
        
        self.class_num = class_num
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        self.register_buffer('running_mean', torch.zeros(class_num, num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
        self.temple_cluster = [ [] for i in range(class_num)]

    def forward(self, input):

        assert self.training==True, 'Use CountMeanOfFreatureInCluster in eval mode.'

        if self.track_running_stats:  
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

            data = input.clone().detach()

            for index in range(data.size(0)):
                self.temple_cluster[self.findClusterIndex(data[index])].append(data[index])

            for index in range(self.class_num):
                if len(self.temple_cluster[index]) > 32:
                    self.running_mean[index] = exponential_average_factor * torch.mean(torch.cat(self.temple_cluster[index], 0), 0) + \
                                               (1-exponential_average_factor) * self.running_mean[index]
                    self.temple_cluster[index] = []

        return input

    def init(self, static_mean):

        for index in range(self.class_num):
            self.running_mean[index], self.temple_cluster[index] = static_mean[index], []

    def getSample(self, input):

        res = []
        for index in range(input.size(0)):
            res.append( torch.unsqueeze(self.running_mean[self.findClusterIndex(input[index])], 0) )
        
        return torch.cat(res, dim=0)

    def findClusterIndex(self, input):

        # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        # cosineSimilarity = np.array([cos(input, self.running_mean[index]) for index in range(self.class_num)])

        # return np.argmax(cosineSimilarity)

        pairwiseDistance = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
        distance = np.array([pairwiseDistance(input.unsqueeze(0), self.running_mean[index].unsqueeze(0)) for index in range(self.class_num)])

        return np.argmin(distance)

