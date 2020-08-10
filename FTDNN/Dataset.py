import os
import copy
import random
import numpy as np

import cv2
from PIL import Image, ImageDraw

import torch
import torch.utils.data as data

cv2.setNumThreads(0)

def getRotationMatrix(leftEye, rightEye):
    x1, y1 = leftEye
    x2, y2 = rightEye
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    return cv2.getRotationMatrix2D(((x1+x2)/2, (y1+y2)/2), np.degrees(np.arctan((y2-y1)/(x2-x1+np.finfo(np.float32).eps))), 1)

def getRotationLandmark(landmark, rotationMatrix):
    for index in range(landmark.shape[0]):
        oriX, oriY = landmark[index]
        landmark[index, 0] = int(oriX * rotationMatrix[0,0] + oriY * rotationMatrix[0,1] + rotationMatrix[0,2])
        landmark[index, 1] = int(oriX * rotationMatrix[1,0] + oriY * rotationMatrix[1,1] + rotationMatrix[1,2])
    return landmark

def skew(img, landmark):

    width, height = img.size
    original_plane = [(0, 0), (width, 0), (width, height), (0, height)] 

    skew_direction = random.randint(0, 3)
    random_length_ratio = random.uniform(0.02, 0.15)

    if skew_direction == 0:   # Left
        skew_amount = random_length_ratio * height
        new_plane = [(0, 0-skew_amount), (width, 0), (width, height), (0, height+skew_amount)]

    elif skew_direction == 1: # Right
        skew_amount = random_length_ratio * height 
        new_plane = [(0, 0), (width, 0-skew_amount), (width, height+skew_amount), (0, height)]

    elif skew_direction == 2: # Top
        skew_amount = random_length_ratio * width
        new_plane = [(0, 0), (width, 0), (width+skew_amount, height), (0-skew_amount, height)]

    else:                     # Bottom
        skew_amount = random_length_ratio * width
        new_plane = [(0-skew_amount, 0), (width+skew_amount, 0), (width, height), (0, height)]

    matrix = []
    for p1, p2 in zip(new_plane, original_plane):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    A, B = np.matrix(matrix, dtype=np.float), np.array(original_plane).reshape(8)

    perspective_skew_cofficients_matrix = np.array(np.dot(np.linalg.pinv(A), B)).reshape(8)
    img = img.transform(img.size, Image.PERSPECTIVE, perspective_skew_cofficients_matrix, resample=Image.BICUBIC)

    a, b, c, d, e, f, g, h = perspective_skew_cofficients_matrix
    for index in range(landmark.shape[0]):
        oriX, oriY = landmark[index]
        landmark[index, 0] = int((a * oriX + b * oriY + c) / (g * oriX + h * oriY + 1))
        landmark[index, 1] = int((d * oriX + e * oriY + f) / (g * oriX + h * oriY + 1))

    return img, landmark

def L_loader(path):
    return Image.open(path).convert('L')

def RGB_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(data.Dataset):
    def __init__(self, imgs, labels, bboxs, landmarks, flag, transform=None, target_transform=None, loader=RGB_loader):
        self.imgs = imgs
        self.labels = labels
        self.bboxs = bboxs
        self.landmarks = landmarks
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.flag = flag
        
    def __getitem__(self, index):
        img, label, bbox, landmark = self.loader(self.imgs[index]), copy.deepcopy(self.labels[index]), copy.deepcopy(self.bboxs[index]), copy.deepcopy(self.landmarks[index])
        ori_img_w, ori_img_h = img.size

        # BoundingBox
        left   = bbox[0]
        top    = bbox[1]
        right  = bbox[2]
        bottom = bbox[3]

        enlarge_bbox = True

        if self.flag=='train':
            random_crop, random_flip, random_noise, random_skew = True, True, True, True
        elif self.flag=='test':
            random_crop, random_flip, random_noise, random_skew = False, False, False, False

        # Enlarge BoundingBox
        padding_w, padding_h = int( 0.5 * max( 0, int( 0.20 * (right-left) ) ) ), int( 0.5 * max( 0, int( 0.20 * (bottom-top) ) ) )
    
        if enlarge_bbox:
            left  = max(left - padding_w, 0)
            right = min(right + padding_w, ori_img_w)

            top = max(top - padding_h, 0)
            bottom = min(bottom + padding_h, ori_img_h)

        if random_crop:
            x_offset = random.randint(-padding_w, padding_w)
            y_offset = random.randint(-padding_h, padding_h)

            left  = max(left + x_offset, 0)
            right = min(right - x_offset, ori_img_w)

            top = max(top + y_offset, 0)
            bottom = min(bottom - y_offset, ori_img_h)

        img = img.crop((left,top,right,bottom))
        crop_img_w, crop_img_h = img.size

        landmark[:,0]-=left
        landmark[:,1]-=top

        if random_flip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            landmark[:,0] = (right - left) - landmark[:,0]

        if random_noise and random.random() > 0.7:
            distance = np.linalg.norm(landmark[0, :]-landmark[1, :])
            noise = int(np.random.normal(scale=0.1*distance))
            landmark[0] += noise
            landmark[1] += noise

        elif random_skew:
            img, landmark = skew(img, landmark)

        rotationMatrix = getRotationMatrix(landmark[0], landmark[1])
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        img = cv2.warpAffine(img, rotationMatrix, (crop_img_w, crop_img_h))
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
        landmark = getRotationLandmark(landmark, rotationMatrix)

        # Transform Image
        trans_img = self.transform(img)
        _, trans_img_w, trans_img_h = trans_img.size()

        inputSizeOfCropNet = 28
        landmark[:, 0] = landmark[:, 0] * inputSizeOfCropNet / crop_img_w
        landmark[:, 1] = landmark[:, 1] * inputSizeOfCropNet / crop_img_h
        landmark = landmark.astype(np.int)

        grid_len = 7 
        half_grid_len = int(grid_len/2)

        for index in range(landmark.shape[0]):
            if landmark[index,0] <= (half_grid_len - 1):
                landmark[index,0] = half_grid_len
            if landmark[index,0] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index,0] = inputSizeOfCropNet - half_grid_len - 1
            if landmark[index,1] <= (half_grid_len - 1):
                landmark[index,1] = half_grid_len
            if landmark[index,1] >= (inputSizeOfCropNet - half_grid_len):
                landmark[index,1] = inputSizeOfCropNet - half_grid_len - 1 
        
        return trans_img, landmark, label

    def __len__(self): 
        return len(self.imgs)
