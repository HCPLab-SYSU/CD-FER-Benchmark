import torch

from MobileNet import MobileNetV2, MobileNetV2_onlyGlobal

model = MobileNetV2(useIntraGCN=False, useInterGCN=False, useRandomMatrix=False, useAllOneMatrix=False, useCov=False, useCluster=False)
# model = MobileNetV2_onlyGlobal()

model_dict = model.state_dict()
checkpoint = torch.load('../preTrainedModel/MobileNetV2.pth') # torch.load('./preTrainedModel/backbone_vgg_ms1m_epochxxx.pth')

new_checkpoint = {} #OrderedDict()
for k,v in checkpoint.items():
    if k.startswith('module.classifier') or k.startswith('classifier'):
        continue

    name = k[7:] if k.startswith('module') else k 
    new_checkpoint[name] = v

model_dict.update(new_checkpoint)
model.load_state_dict(model_dict)

torch.save(model.state_dict(),'./preTrainedModel/mobilenetv2_ms1m_112_CropNet.pkl')
# torch.save(model.state_dict(),'./preTrainedModel/mobilenetv2_ms1m_112_onlyGlobal.pkl')
