import torch

def getCenters(data_loader, model):

    features = [[] for i in range(7)]
    for step, (input, landmark, label) in enumerate(data_loader):

        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        with torch.no_grad():
            feature, output, loc_output = model(input, landmark)
        for index in range(feature.size(0)):
            features[label[index]].append(feature[index])

    centers = []
    for index in range(7):
        centers.append(torch.mean(torch.stack(features[index]), dim=0))

    return centers

def LP_Loss(feature, label, center):

    '''
    Paper Link : http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.pdf
    '''
    
    dis = []

    for index in range(feature.size(0)):
        dis.append(torch.mean(torch.pow(feature-center[label[index]], 2)))

    dis = torch.mean(torch.Tensor(dis))

    return dis

# def LP_Loss(feature_, label_, data_loader, model, k=10):

#     '''
#     Paper Link : http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.pdf
#     '''
    
#     model.eval()

#     distances = [[] for i in range(feature_.size(0))]

#     for step, (input, landmark, label) in enumerate(data_loader):

#         input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
#         with torch.no_grad():
#             feature, output, loc_output = model(input, landmark)

#             for index in range(feature.size(0)):
#                 for index_ in range(feature_.size(0)):
#                     if label[index]==label_[index_]:
#                         distances[index_].append( torch.mean( torch.pow(feature_[index_]-feature[index], 2) ) )

#     res = []
#     for index_ in range(feature_.size(0)):
#         res.append( torch.mean(torch.topk(torch.Tensor(distances[index_]), k, largest=False)[0]) )
#     res = torch.mean(torch.Tensor(res))

#     return res

# def LP_Loss(feature_, label_, data_loader, model, k=10):

#     '''
#     Paper Link : http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.pdf
#     '''
    
#     model.eval()

#     k_neighbors = [[] for i in range(feature_.size(0))]

#     for step, (input, landmark, label) in enumerate(data_loader):

#         input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
#         with torch.no_grad():
#             feature, output, loc_output = model(input, landmark)
        
#         for feature_index in range(feature.size(0)):

#             for feature_index_ in range(feature_.size(0)):
                
#                 if label[feature_index]==label_[feature_index_]:

#                     if len(k_neighbors[feature_index_]) < k:
#                         k_neighbors[feature_index_].append(feature[feature_index])

#                     elif len(k_neighbors[feature_index_])==k:
                        
#                         dis_ = torch.sum(torch.pow(feature_[feature_index_]-feature[feature_index], 2))

#                         index = 0
#                         dis = torch.sum(torch.pow(feature_[feature_index_]-k_neighbors[feature_index_][0], 2))

#                         for i in range(1,k):
#                             dis_i = torch.sum(torch.pow(feature_[feature_index_]-k_neighbors[feature_index_][i], 2))

#                             if dis < dis_i:
#                                 dis = dis_i
#                                 index = i

#                         if dis_ < dis:
#                             k_neighbors[feature_index_][index] = feature[feature_index]

#                 torch.cuda.empty_cache()

#     res = []
#     for index in range(feature_.size(0)):
        
#         res_ = 0
#         for i in range(k):
#             res_+=k_neighbors[index][i]

#         res.append(torch.unsqueeze(res_/k, 0))
    
#     res = torch.cat(res, dim=0)

#     model.train()

#     return torch.mean(torch.pow(feature_-res, 2))
