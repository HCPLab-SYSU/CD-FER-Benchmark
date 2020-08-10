Log_Name='ResNet50_CropNet_withSAFN_trainOnSourceDomain_RAFtoAFED'
Resume_Model='../preTrainedModel/ir50_ms1m_112_CropNet.pkl'
OutputPath='.'
GPU_ID: 0
Backbone='ResNet50'
useAFN='True'
methodOfAFN='SAFN'
radius=40
deltaRadius=0.001
weight_L2norm=0.05
faceScale=112
sourceDataset='RAF'
targetDataset='AFED'
train_batch_size=32
test_batch_size=32
useMultiDatasets='False'
epochs=60
lr=0.0001
momentum=0.9
weight_decay=0.0001
isTest='False'
showFeature='False'
class_num=7
useIntraGCN='False'
useInterGCN='False'
useLocalFeature='True'
useRandomMatrix='False'
useAllOneMatrix='False'
useCov='False'
useCluster='False'

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=${GPU_ID} python3 TrainOnSourceDomain.py \ 
    --Log_Name ${Log_Name} \
    --OutputPath ${OutputPath} \ 
    --Resume_Model ${Resume_Model} \ 
    --GPU_ID ${GPU_ID} \ 
    --Backbone ${Backbone } \ 
    --useAFN ${useAFN} \ 
    --methodOfAFN ${methodOfAFN } \ 
    --radius ${radius} \ 
    --deltaRadius ${deltaRadius} \ 
    --weight_L2norm ${weight_L2norm} \ 
    --faceScale ${faceScale} \ 
    --sourceDataset ${sourceDataset} \ 
    --targetDataset ${targetDataset} \ 
    --train_batch_size ${train_batch_size}\ 
    --test_batch_size ${test_batch_size} \ 
    --useMultiDatasets ${useMultiDatasets} \ 
    --epochs ${epochs} \ 
    --lr ${lr} \ 
    --momentum ${momentum} \ 
    --weight_decay ${weight_decay} \ 
    --isTest ${isTest} \ 
    --showFeature ${showFeature} \ 
    --class_num ${class_num} \ 
    --useIntraGCN ${useIntraGCN} \ 
    --useInterGCN ${useInterGCN} \ 
    --useLocalFeature ${useLocalFeature} \ 
    --useRandomMatrix ${useRandomMatrix} \ 
    --useAllOneMatrix ${useAllOneMatrix} \ 
    --useCov ${useCov} \ 
    --useCluster={{ useCluster }}