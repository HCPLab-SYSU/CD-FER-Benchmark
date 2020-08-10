import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module

from GraphConvolutionNetwork import GCN, GCNwithIntraAndInterMatrix
from Model import CountMeanOfFeature, CountMeanAndCovOfFeature, CountMeanOfFeatureInCluster

class Backbone_VGG(nn.Module):
    def __init__(self, useIntraGCN=True, useInterGCN=True, useRandomMatrix=False, useAllOneMatrix=False, useCov=False, useCluster=False):

        super(Backbone_VGG, self).__init__()

        self.layer1 = Sequential(Conv2d(3, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU(inplace=True),
                                 Conv2d(64, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU(inplace=True),
                                 MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = Sequential(Conv2d(64, 128, kernel_size=3, padding=1), BatchNorm2d(128), ReLU(inplace=True),
                                 Conv2d(128, 128, kernel_size=3, padding=1), BatchNorm2d(128), ReLU(inplace=True),
                                 MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = Sequential(Conv2d(128, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU(inplace=True),
                                 Conv2d(256, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU(inplace=True),
                                 Conv2d(256, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU(inplace=True),
                                 MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = Sequential(Conv2d(256, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(inplace=True),
                                 Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(inplace=True),
                                 Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(inplace=True),

                                 Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(inplace=True),
                                 Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(inplace=True),
                                 Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(inplace=True),
                                 MaxPool2d(kernel_size=2, stride=2))

        self.output_layer = Sequential(nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1,1)))

        self.Crop_Net = nn.ModuleList([ Sequential(Conv2d(128, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU(inplace=True),
                                                   Conv2d(256, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU(inplace=True),
                                                   Conv2d(256, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(inplace=True),
                                                   Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(inplace=True),
                                                   nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)), nn.ReLU() ) for i in range(5) ])

        self.fc = nn.Linear(64 + 320, 7)
        self.loc_fc = nn.Linear(320, 7)

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

        featureMap1 = self.layer1(imgs)        # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1) # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2) # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3) # Batch * 512 * 7 * 7

        global_feature = self.output_layer(featureMap4).view(imgs.size(0), -1) # Batch * 64
        loc_feature = self.crop_featureMap(featureMap2, locations)             # Batch * 320
        feature = torch.cat((global_feature, loc_feature), 1)                  # Batch * (64+320)

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

        featureMap1 = self.layer1(imgs)        # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1) # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2) # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3) # Batch * 512 * 7 * 7

        global_feature = self.output_layer(featureMap4).view(imgs.size(0), -1)        # Batch * 64
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
        parameter_list = [  {"params":self.layer1.parameters(), "lr_mult":1, 'decay_mult':2}, \
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


class Backbone_VGG_onlyGlobal(nn.Module):
    def __init__(self):

        super(Backbone_VGG_onlyGlobal, self).__init__()

        self.layer1 = Sequential(Conv2d(3, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU(inplace=True), 
                                 Conv2d(64, 64, kernel_size=3, padding=1), BatchNorm2d(64), ReLU(inplace=True),
                                 MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = Sequential(Conv2d(64, 128, kernel_size=3, padding=1), BatchNorm2d(128), ReLU(inplace=True),
                                 Conv2d(128, 128, kernel_size=3, padding=1), BatchNorm2d(128), ReLU(inplace=True),
                                 MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = Sequential(Conv2d(128, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU(inplace=True),
                                 Conv2d(256, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU(inplace=True),
                                 Conv2d(256, 256, kernel_size=3, padding=1), BatchNorm2d(256), ReLU(inplace=True),
                                 MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = Sequential(Conv2d(256, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(inplace=True),
                                 Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(inplace=True),
                                 Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(inplace=True),

                                 Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(inplace=True),
                                 Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(inplace=True),
                                 Conv2d(512, 512, kernel_size=3, padding=1), BatchNorm2d(512), ReLU(inplace=True),
                                 MaxPool2d(kernel_size=2, stride=2))

        self.GAP = nn.AdaptiveAvgPool2d((1,1))

        self.output_layer = Sequential(nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1,1)))

        self.fc = nn.Linear(64, 7)

    def classify(self, imgs, locations):

        featureMap1 = self.layer1(imgs)        # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1) # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2) # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3) # Batch * 512 * 7 * 7

        feature = self.output_layer(featureMap4).view(imgs.size(0), -1) # Batch * 64

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
        parameter_list = [  {"params":self.layer1.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer2.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer3.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.layer4.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.output_layer.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            ]
        return parameter_list

def VGG(useIntraGCN, useInterGCN, useRandomMatrix, useAllOneMatrix, useCov, useCluster):
    """Constructs a vgg-16 model."""

    model = Backbone_VGG(useIntraGCN, useInterGCN, useRandomMatrix, useAllOneMatrix, useCov, useCluster)

    return model

def VGG_onlyGlobal():
    """Constructs a vgg-16 model."""

    model = Backbone_VGG_onlyGlobal()

    return model
