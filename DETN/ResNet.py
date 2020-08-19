import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module

import copy
import numpy as np
from collections import namedtuple

# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x

class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut

class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers):
    if num_layers == 18:  # (2+2+2+2)*2+2=18
        blocks = [
            get_block(in_channel=64, depth=64, num_units=2),
            get_block(in_channel=64, depth=128, num_units=2),
            get_block(in_channel=128, depth=256, num_units=2), 
            get_block(in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

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

class Backbone(nn.Module):
    def __init__(self, numOfLayer):   

        super(Backbone, self).__init__()

        unit_module = bottleneck_IR
        
        self.input_layer = Sequential(Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        blocks = get_blocks(numOfLayer)
        self.layer1 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[0]]) #get_block(in_channel=64, depth=64, num_units=3)])
        self.layer2 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[1]]) #get_block(in_channel=64, depth=128, num_units=4)])
        self.layer3 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[2]]) #get_block(in_channel=128, depth=256, num_units=14)])
        self.layer4 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[3]]) #get_block(in_channel=256, depth=512, num_units=3)])

        self.output_layer = Sequential(nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)), 
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1,1)))

        cropNet_modules = []
        cropNet_blocks = [get_block(in_channel=128, depth=256, num_units=2), get_block(in_channel=256, depth=512, num_units=2)]
        for block in cropNet_blocks:
            for bottleneck in block:
                cropNet_modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        cropNet_modules+=[nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)), nn.ReLU()]
        self.Crop_Net = nn.ModuleList([ copy.deepcopy(nn.Sequential(*cropNet_modules)) for i in range(5) ])

        self.fc = nn.Linear(64 + 320, 7)
        self.fc.apply(init_weights)

        self.loc_fc = nn.Linear(320, 7)
        self.loc_fc.apply(init_weights)

        self.GAP = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, imgs, locations):

        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1) # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2) # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3) # Batch * 512 * 7 * 7

        global_feature = self.output_layer(featureMap4).view(featureMap.size(0), -1) # Batch * 64
        loc_feature = self.crop_featureMap(featureMap2, locations)                   # Batch * 320
        feature = torch.cat((global_feature, loc_feature), 1)                        # Batch * (64+320)    
        
        pred = self.fc(feature)              # Batch * 7
        loc_pred = self.loc_fc(loc_feature)  # Batch * 7

        return feature, pred, loc_pred

    def output_num(self):
        return 64*6

    def get_parameters(self):
        parameter_list = [  {"params":self.input_layer.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer1.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer2.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer3.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer4.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.output_layer.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.loc_fc.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.Crop_Net.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            ]
        return parameter_list

    def crop_featureMap(self, featureMap, locations):

        batch_size = featureMap.size(0)
        map_ch = featureMap.size(1)
        map_len = featureMap.size(2)

        grid_ch = map_ch
        grid_len = 7 # 14, 6, 4

        feature_list = []
        for i in range(5):
            grid_list = []
            for j in range(batch_size):
                w_min = locations[j,i,0]-int(grid_len/2)
                w_max = locations[j,i,0]+int(grid_len/2)
                h_min = locations[j,i,1]-int(grid_len/2)
                h_max = locations[j,i,1]+int(grid_len/2)
                
                map_w_min = max(0, w_min)
                map_w_max = min(map_len-1, w_max)
                map_h_min = max(0, h_min)
                map_h_max = min(map_len-1, h_max)
                
                grid_w_min = max(0, 0-w_min)
                grid_w_max = grid_len + min(0, map_len-1-w_max)
                grid_h_min = max(0, 0-h_min)
                grid_h_max = grid_len + min(0, map_len-1-h_max)
                
                grid = torch.zeros(grid_ch, grid_len, grid_len)
                if featureMap.is_cuda:
                    grid = grid.cuda()

                grid[:, grid_h_min:grid_h_max+1, grid_w_min:grid_w_max+1] = featureMap[j, :, map_h_min:map_h_max+1, map_w_min:map_w_max+1] 

                grid_list.append(grid)

            feature = torch.stack(grid_list, dim=0)
            feature_list.append(feature)
 
        # feature list: 5 * [ batch_size * channel * 3 * 3 ]
        output_list = []
        for i in range(5):
            output = self.Crop_Net[i](feature_list[i])
            output = self.GAP(output)
            output_list.append(output)

        loc_feature = torch.stack(output_list, dim=1)  # batch_size * 5 * 64 * 1 * 1
        loc_feature = loc_feature.view(batch_size, -1) # batch_size * 320 

        return loc_feature

class Backbone_onlyGlobal(nn.Module):
    def __init__(self, numOfLayer):

        super(Backbone_onlyGlobal, self).__init__()

        unit_module = bottleneck_IR
        
        self.input_layer = Sequential(Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1,1), padding=(1,1), bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        blocks = get_blocks(numOfLayer)
        self.layer1 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[0]]) #get_block(in_channel=64, depth=64, num_units=3)])
        self.layer2 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[1]]) #get_block(in_channel=64, depth=128, num_units=4)])
        self.layer3 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[2]]) #get_block(in_channel=128, depth=256, num_units=14)])
        self.layer4 = Sequential(*[unit_module(bottleneck.in_channel,bottleneck.depth,bottleneck.stride) for bottleneck in blocks[3]]) #get_block(in_channel=256, depth=512, num_units=3)])

        self.output_layer = Sequential(nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)), 
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1,1)))

        self.fc = nn.Linear(64, 7)
        self.fc.apply(init_weights)
        
    def forward(self, imgs, locations):

        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1) # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2) # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3) # Batch * 512 * 7 * 7

        feature = self.output_layer(featureMap4).view(featureMap.size(0), -1) # Batch * 64

        pred = self.fc(feature)              # Batch * 7
        loc_pred = None

        return feature, pred, loc_pred

    def output_num(self):
        return 64

    def get_parameters(self):
        parameter_list = [  {"params":self.input_layer.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer1.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer2.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer3.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer4.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.output_layer.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            ]
        return parameter_list

def IR(numOfLayer):
    """Constructs a ir-50 model."""

    model = Backbone(numOfLayer)

    return model

def IR_onlyGlobal(numOfLayer):
    """Constructs a ir-50 model."""

    model = Backbone_onlyGlobal(numOfLayer)

    return model
