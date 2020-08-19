import torch

from VGG import VGG, VGG_onlyGlobal

model = VGG(useIntraGCN=True, useInterGCN=True, useRandomMatrix=False, useAllOneMatrix=False, useCov=False, useCluster=True)
# model = VGG_onlyGlobal()

model_dict = model.state_dict()
checkpoint = torch.load('../preTrainedModel/VGGNet_Checkpoint.pth.tar')['state_dict']

new_checkpoint = {} #OrderedDict()
for k,v in checkpoint.items():
    if k.startswith('module.classifier') or k.startswith('classifier'):
        continue

    name = k[7:] if k.startswith('module') else k 
    new_checkpoint[name] = v

model_dict.update(new_checkpoint)
model.load_state_dict(model_dict)

torch.save(model.state_dict(),'./preTrainedModel/vgg_ms1m_112_CropNet_GCNwithIntraMatrixAndInterMatrix_useCluster.pkl')
# torch.save(model.state_dict(),'./preTrainedModel/vgg_ms1m_112_onlyGlobal.pkl')
