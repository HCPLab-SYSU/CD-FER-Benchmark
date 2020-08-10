Log_Name='ResNet50_CropNet_RAFtoAFED'
Resume_Model='../preTrainedModel/ir50_ms1m_112_CropNet.pkl'
OutputPath='.'
GPU_ID=0
Backbone='ResNet50'
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
class_num=7
     
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=${GPU_ID} python3 TrainOnSourceDomain.py \ 
    --Log_Name ${Log_Name} \
    --OutputPath ${OutputPath} \ 
    --Resume_Model ${Resume_Model} \ 
    --GPU_ID ${GPU_ID} \ 
    --Backbone ${Backbone } \ 
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
    --class_num ${class_num} 