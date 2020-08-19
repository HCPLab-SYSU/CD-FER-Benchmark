import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module

import copy
import numpy as np
from collections import namedtuple

from GraphConvolutionNetwork import GCN, GCNwithIntraAndInterMatrix
from Model import CountMeanOfFeature, CountMeanAndCovOfFeature, CountMeanOfFeatureInCluster

# Support: ['IR_18', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']

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
    if num_layers == 18:
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

class Backbone(nn.Module):
    def __init__(self, numOfLayer, useIntraGCN=True, useInterGCN=True, useRandomMatrix=False, useAllOneMatrix=False, useCov=False, useCluster=False):   

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

        #self.GCN = GCN(64, 128, 64)
        self.GCN = GCNwithIntraAndInterMatrix(64, 128, 64, 
                                              useIntraGCN=useIntraGCN, useInterGCN=useInterGCN, 
                                              useRandomMatrix=useRandomMatrix, useAllOneMatrix=useAllOneMatrix)

        self.SourceMean = (CountMeanAndCovOfFeature(64+320) if useCov else CountMeanOfFeature(64+320)) if not useCluster else CountMeanOfFeatureInCluster(64+320)
        self.TargetMean = (CountMeanAndCovOfFeature(64+320) if useCov else CountMeanOfFeature(64+320)) if not useCluster else CountMeanOfFeatureInCluster(64+320)
 
        self.SourceBN = BatchNorm1d(64+320)
        self.TargetBN = BatchNorm1d(64+320)

    def classify(self, imgs, locations):

        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1) # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2) # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3) # Batch * 512 * 7 * 7

        global_feature = self.output_layer(featureMap4).view(featureMap.size(0), -1) # Batch * 64
        loc_feature = self.crop_featureMap(featureMap2, locations)                   # Batch * 320
        feature = torch.cat((global_feature, loc_feature), 1)                        # Batch * (64+320)    
        
        # GCN
        if self.training:
            feature = self.SourceMean(feature)
        feature = torch.cat( ( self.SourceBN(feature), self.TargetBN(self.TargetMean.getSample(feature.detach())) ), 1) # Batch * (64+320 + 64+320)
        feature = self.GCN(feature.view(feature.size(0), 12, -1))                                                       # Batch * 12 * 64

        feature = feature.view(feature.size(0), -1).narrow(1, 0, 64+320) # Batch * (64+320)
        loc_feature = feature.narrow(1, 64, 320)                         # Batch * 320

        pred = self.fc(feature)              # Batch * 7
        loc_pred = self.loc_fc(loc_feature)  # Batch * 7

        return feature, pred, loc_pred

    def transfer(self, imgs, locations, domain='Target'):

        assert domain in ['Source', 'Target'], 'Parameter domain should be Source or Target.'

        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1) # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2) # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3) # Batch * 512 * 7 * 7

        global_feature = self.output_layer(featureMap4).view(featureMap.size(0), -1)  # Batch * 64
        loc_feature = self.crop_featureMap(featureMap2, locations)                    # Batch * 320
        feature = torch.cat((global_feature, loc_feature), 1)                         # Batch * (64+320)

        if self.training:

            # Compute Feature
            SourceFeature = feature.narrow(0, 0, feature.size(0)//2)                  # Batch/2 * (64+320)
            TargetFeature = feature.narrow(0, feature.size(0)//2, feature.size(0)//2) # Batch/2 * (64+320)

            SourceFeature = self.SourceMean(SourceFeature) # Batch/2 * (64+320)
            TargetFeature = self.TargetMean(TargetFeature) # Batch/2 * (64+320)

            SourceFeature = self.SourceBN(SourceFeature)   # Batch/2 * (64+320)
            TargetFeature = self.TargetBN(TargetFeature)   # Batch/2 * (64+320)

            # Compute Mean
            SourceMean = self.SourceMean.getSample(TargetFeature.detach()) # Batch/2 * (64+320)
            TargetMean = self.TargetMean.getSample(SourceFeature.detach()) # Batch/2 * (64+320)

            SourceMean = self.SourceBN(SourceMean) # Batch/2 * (64+320)
            TargetMean = self.TargetBN(TargetMean) # Batch/2 * (64+320)

            # GCN
            feature = torch.cat( ( torch.cat((SourceFeature,TargetMean), 1), torch.cat((SourceMean,TargetFeature), 1) ), 0) # Batch * (64+320 + 64+320)
            feature = self.GCN(feature.view(feature.size(0), 12, -1))                                                       # Batch * 12 * 64

            feature = feature.view(feature.size(0), -1)                                                                     # Batch * (64+320 + 64+320)
            feature = torch.cat( (feature.narrow(0, 0, feature.size(0)//2).narrow(1, 0, 64+320), \
                                  feature.narrow(0, feature.size(0)//2, feature.size(0)//2).narrow(1, 64+320, 64+320) ), 0) # Batch * (64+320)
            loc_feature = feature.narrow(1, 64, 320)                                                                        # Batch * 320

            pred = self.fc(feature)             # Batch * 7
            loc_pred = self.loc_fc(loc_feature) # Batch * 7

            return feature, pred, loc_pred

        # Inference
        if domain=='Source':
            SourceFeature = feature                                         # Batch * (64+320)
            TargetMean = self.TargetMean.getSample(SourceFeature.detach())  # Batch * (64+320)

            SourceFeature = self.SourceBN(SourceFeature)                    # Batch * (64+320)
            TargetMean = self.TargetBN(TargetMean)                          # Batch * (64+320)
            
            feature = torch.cat((SourceFeature,TargetMean), 1)              # Batch * (64+320 + 64+320)
            feature = self.GCN(feature.view(feature.size(0), 12, -1))       # Batch * 12 * 64

        elif domain=='Target':
            TargetFeature = feature                                         # Batch * (64+320)
            SourceMean = self.SourceMean.getSample(TargetFeature.detach())  # Batch * (64+320)

            SourceMean = self.SourceBN(SourceMean)                          # Batch * (64+320)
            TargetFeature = self.TargetBN(TargetFeature)                    # Batch * (64+320)

            feature = torch.cat((SourceMean,TargetFeature), 1)              # Batch * (64+320 + 64+320)
            feature = self.GCN(feature.view(feature.size(0), 12, -1))       # Batch * 12 * 64
            
        feature = feature.view(feature.size(0), -1)      # Batch * (64+320 + 64+320)
        if domain=='Source':
            feature = feature.narrow(1, 0, 64+320)       # Batch * (64+320)
        elif domain=='Target':   
            feature = feature.narrow(1, 64+320, 64+320)  # Batch * (64+320)

        loc_feature = feature.narrow(1, 64, 320)         # Batch * 320

        pred = self.fc(feature)             # Batch * 7
        loc_pred = self.loc_fc(loc_feature) # Batch * 7

        return feature, pred, loc_pred

    def forward(self, imgs, locations, flag=True, domain='Target'):
        
        if flag:
            return self.classify(imgs, locations)

        return self.transfer(imgs, locations, domain)

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
                            {"params":self.GCN.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.SourceBN.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.TargetBN.parameters(), "lr_mult":10, 'decay_mult':2}, \
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
    def __init__(self):

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
        
    def classify(self, imgs, locations):

        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1) # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2) # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3) # Batch * 512 * 7 * 7

        feature = self.output_layer(featureMap4).view(featureMap.size(0), -1) # Batch * 64

        pred = self.fc(feature)              # Batch * 7
        loc_pred = None

        return feature, pred, loc_pred

    def transfer(self, imgs, locations, domain='Target'):

        assert domain in ['Source', 'Target'], 'Parameter domain should be Source or Target.'

        featureMap = self.input_layer(imgs)

        featureMap1 = self.layer1(featureMap)  # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1) # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2) # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3) # Batch * 512 * 7 * 7

        feature = self.output_layer(featureMap4).view(featureMap.size(0), -1)  # Batch * 64

        pred = self.fc(feature)  # Batch * 7
        loc_pred = None

        return feature, pred, loc_pred

    def forward(self, imgs, locations, flag=True, domain='Target'):
        
        if flag:
            return self.classify(imgs, locations)

        return self.transfer(imgs, locations, domain)

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

def IR(numOfLayer, useIntraGCN, useInterGCN, useRandomMatrix, useAllOneMatrix, useCov, useCluster):
    """Constructs a ir-18/ir-50 model."""

    model = Backbone(numOfLayer, useIntraGCN, useInterGCN, useRandomMatrix, useAllOneMatrix, useCov, useCluster)

    return model

def IR_onlyGlobal(numOfLayer):
    """Constructs a ir-18/ir-50 model."""

    model = Backbone_onlyGlobal(numOfLayer)

    return model
