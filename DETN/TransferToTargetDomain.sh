Log_Name='ResNet50_CropNet_transferToTargetDomain_RAFtoAFED'
Resume_Model='ResNet50_CropNet_trainOnSourceDomain_RAFtoAFED.pkl'
OutputPath='.'
GPU_ID=0
Backbone='ResNet50'
useMMD='True'
Gamma=1
Lambda=0
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
    
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=${GPU_ID} python3 TransferToTargetDomain.py \ 
    --Log_Name ${Log_Name} \ 
    --OutputPath ${OutputPath} \ 
    --Backbone ${Backbone } \ 
    --Resume_Model ${Resume_Model} \ 
    --GPU_ID ${GPU_ID} \ 
    --useMMD ${useMMD} \ 
    --Gamma ${Gamma} \ 
    --Lambda ${Lambda} \ 
    --faceScale ${faceScale} \ 
    --sourceDataset ${sourceDataset} \ 
    --targetDataset ${targetDataset} \ 
    --train_batch_size ${train_batch_size} \ 
    --test_batch_size ${test_batch_size} \ 
    --useMultiDatasets ${useMultiDatasets} \ 
    --epochs ${epochs} \ 
    --lr ${lr} \ 
    --momentum ${momentum} \ 
    --weight_decay ${weight_decay} \ 
    --isTest ${isTest} \ 
    --class_num ${class_num}