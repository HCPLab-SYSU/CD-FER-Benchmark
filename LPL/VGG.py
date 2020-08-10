import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module

class Backbone(nn.Module):
    def __init__(self):

        super(Backbone, self).__init__()

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
                                                   nn.Conv2d(in_channels=512, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)), nn.ReLU()) for i in range(5) ])

        self.fc = nn.Linear(64 + 320, 7)
        self.loc_fc = nn.Linear(320, 7)

        self.GAP = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, imgs, locations):

        featureMap1 = self.layer1(imgs)        # Batch * 64 * 56 * 56
        featureMap2 = self.layer2(featureMap1) # Batch * 128 * 28 * 28
        featureMap3 = self.layer3(featureMap2) # Batch * 256 * 14 * 14
        featureMap4 = self.layer4(featureMap3) # Batch * 512 * 7 * 7

        global_feature = self.output_layer(featureMap4).view(imgs.size(0), -1) # Batch * 64
        loc_feature = self.crop_featureMap(featureMap2, locations)                   # Batch * 320
        feature = torch.cat((global_feature, loc_feature), 1)                        # Batch * (64+320)

        pred = self.fc(feature)              # Batch * 7
        loc_pred = self.loc_fc(loc_feature)  # Batch * 7

        return feature, pred, loc_pred

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

def VGG():
    """Constructs a vgg-16 model."""

    model = Backbone()

    return model
