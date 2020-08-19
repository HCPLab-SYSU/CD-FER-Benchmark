import torch

from ResNet import IR, IR_onlyGlobal

numOfLayer = 50 # [18, 50]
model = IR(numOfLayer=numOfLayer, useIntraGCN=True, useInterGCN=True, useRandomMatrix=False, useAllOneMatrix=False, useCov=False, useCluster=True)
# model = IR_onlyGlobal(numOfLayer=numOfLayer)

model_dict = model.state_dict()
checkpoint = torch.load('../preTrainedModel/backbone_ir50_ms1m_epoch120.pth') if numOfLayer == 50 else \
             torch.load('../preTrainedModel/backbone_IR_18_HeadFC_Softmax_112_512_1.0_Epoch_156_lfw_112_0.994_X4_112_0.990_agedb_30_112_0.949.pth')

indexToLayer = {'0':'layer1.0.', '1':'layer1.1.', '2':'layer1.2.', \
                '3':'layer2.0.', '4':'layer2.1.', '5':'layer2.2.', '6':'layer2.3.',\
                '7':'layer3.0.', '8':'layer3.1.', '9':'layer3.2.', '10':'layer3.3.', '11':'layer3.4.', '12':'layer3.5.', '13':'layer3.6.', '14':'layer3.7.', '15':'layer3.8.', '16':'layer3.9.', '17':'layer3.10.', '18':'layer3.11.', '19':'layer3.12.', '20':'layer3.13.',\
                '21':'layer4.0.', '22':'layer4.1.', '23':'layer4.2.'} if numOfLayer == 50 else \
               {'0':'layer1.0.', '1':'layer1.1.', \
                '2':'layer2.0.', '3':'layer2.1.', \
                '4':'layer3.0.', '5':'layer3.1.', \
                '6':'layer4.0.', '7':'layer4.1.'}

newCheckpoint = {}
for key, value in checkpoint.items():
    subStr = key.split('.',2)
    if subStr[0]=='body':
        newKey = indexToLayer[subStr[1]] + subStr[2]
        newCheckpoint[newKey] = value
    elif subStr[0]=='output_layer':
        continue
    else:
        newCheckpoint[key] = value

for key, value in model_dict.items():
    subStr = key.split('.',2)
    if subStr[0]=='fc' or subStr[0]=='loc_fc' or \
       subStr[0]=='Crop_Net' or subStr[0]=='GCN' or \
       subStr[0]=='SourceMean' or subStr[0]=='TargetMean' or \
       subStr[0]=='SourceBN' or subStr[0]=='TargetBN' or \
       subStr[0]=='output_layer':
        newCheckpoint[key] = value

model.load_state_dict(newCheckpoint)
if numOfLayer == 50:
    torch.save(model.state_dict(), './preTrainedModel/ir50_ms1m_112_CropNet_GCNwithIntraMatrixAndInterMatrix_useCluster.pkl')
    # torch.save(model.state_dict(), './preTrainedModel/ir50_ms1m_112_onlyGlobal.pkl')
    
elif numOfLayer == 18:
    torch.save(model.state_dict(), './preTrainedModel/ir18_lfw_112_CropNet_GCNwithIntraMatrixAndInterMatrix_useCluster.pkl')
    # torch.save(model.state_dict(), './preTrainedModel/ir18_lfw_112_onlyGlobal.pkl')
