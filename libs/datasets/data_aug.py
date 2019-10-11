import os
import cv2
import numpy as np
import sys
sys.path.insert(0,'/home/zhangming')
# sys.path.insert(0,'/home/remo/Desktop')
print(sys.path)
from MyAugmentor import Augmentor
from tqdm import tqdm

def aug(aug_pic,prb,sample_num):
    p = Augmentor.DataPipeline(aug_pic)
    p.rotate(prb['Rotate'],20,25)
    p.random_distortion(prb['Distortation'],3, 3, 1)
    # p.gaussian_distortion(1,3, 3, 1.0, method="in", corner="bell")
    # p.random_erasing(prb['RandomErasing'],0.2)
    auged_pic = p.sample(sample_num)
    return auged_pic


def pre_data(image_path,mask_path,sample_num,prb,flg_show,flg_save):
    count = 0
    for im in tqdm(os.listdir(image_path)):
        aug_pic = []
        img_p = os.path.join(image_path, im)
        mask_p = os.path.join(mask_path, im.replace('jpg', 'png'))
        img = cv2.imread(img_p)
        mask = cv2.imread(mask_p)
        aug_pic.append([img, mask])
        auged_pic = aug(aug_pic, prb, sample_num)
        count = show_or_save_pic(auged_pic,aug_pic, flg_show, flg_save,count)

def show_or_save_pic(auged_pic,aug_pic,flg_show,flg_save,count):
    raw_image = aug_pic[0][0]
    raw_mask = aug_pic[0][1]
    for group in auged_pic:
        image = group[0]
        mask = group[1]
        if flg_show:
            cv2.imshow('raw_image', raw_image)
            cv2.imshow('raw_mask', raw_mask * 100)
            cv2.imshow('image',image)
            cv2.imshow('mask',mask*100)
            cv2.waitKey(0)
        if flg_save:
            image_path_aug = image_path+'aug'
            mask_path_aug = mask_path+'aug'
            if not os.path.exists(image_path_aug):
                os.mkdir(image_path_aug)
            if not os.path.exists(mask_path_aug):
                os.mkdir(mask_path_aug)
            cv2.imwrite(image_path_aug+'/'+str(count)+'.jpg',image)
            cv2.imwrite(mask_path_aug+'/'+str(count)+'.png',mask)
            count += 1
        else:
            continue
    return count


if __name__ == "__main__":
    # image_path = '/home/remo/Desktop/seg_data/image_10_ratio1/img'
    image_path = '/home/zhangming/Datasets/mask/Kaggle/rematch/data/image_21_ratio1/img'
    # mask_path = '/home/remo/Desktop/seg_data/image_10_ratio1/label'
    mask_path = '/home/zhangming/Datasets/mask/Kaggle/rematch/data/image_21_ratio1/label'
    prb = {'Rotate':1.0,'Distortation':1.0,'RandomErasing':0.0}
    flg_show = False
    flg_save = not flg_show
    sample_num = 1
    pre_data(image_path, mask_path,sample_num,prb,flg_show,flg_save)
