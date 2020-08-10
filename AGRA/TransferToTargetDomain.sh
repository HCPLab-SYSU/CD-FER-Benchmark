Log_Name='ResNet50_CropNet_GCNwithIntraMatrixAndInterMatrix_useCluster_withoutAFN_transferToTargetDomain_RAFtoAFED'
Resume_Model='ResNet50_CropNet_GCNwithIntraMatrixAndInterMatrix_useCluster_withoutAFN_trainOnSourceDomain_RAFtoAFED.pkl'
OutputPath='.'
GPU_ID=0
Backbone='ResNet18'
useAFN='False'
methodOfAFN='SAFN'
radius=25
deltaRadius=1
weight_L2norm=0.05
useDAN='True'
methodOfDAN='CDAN-E'
faceScale=112
sourceDataset='RAF'
targetDataset='AFED'
train_batch_size=32
test_batch_size=32
useMultiDatasets='False'
epochs=60
lr=0.0001
lr_ad=0.001
momentum=0.9
weight_decay=0.0001
isTest='False'
showFeature='False'
class_num=7
useIntraGCN='True'
useInterGCN='True'
useLocalFeature='True'
useRandomMatrix='False'
useAllOneMatrix='False'
useCov='False'
useCluster='True'
    
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=${GPU_ID} python3 TransferToTargetDomain.py \ 
    --Log_Name ${Log_Name} \ 
    --OutputPath ${OutputPath} \ 
    --Backbone ${Backbone } \ 
    --useOtherExps ${useOtherExps} \ 
    --Resume_Model ${Resume_Model} \ 
    --GPU_ID ${GPU_ID} \ 
    --useAFN ${useAFN} \ 
    --methodOfAFN ${methodOfAFN} \ 
    --radius ${radius} \ 
    --deltaRadius ${deltaRadius} \ 
    --weight_L2norm ${weight_L2norm} \ 
    --useDAN ${useDAN} \ 
    --methodOfDAN ${methodOfDAN} \ 
    --faceScale ${faceScale} \ 
    --sourceDataset ${sourceDataset} \ 
    --targetDataset ${targetDataset} \ 
    --train_batch_size ${train_batch_size} \ 
    --test_batch_size ${test_batch_size} \ 
    --useMultiDatasets ${useMultiDatasets} \ 
    --epochs ${epochs} \ 
    --lr ${lr} \ 
    --lr_ad ${lr_ad} \ 
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
    --useCluster ${useCluster}