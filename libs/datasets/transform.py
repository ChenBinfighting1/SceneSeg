# encoding: utf-8

import cv2
import numpy as np
import torch


# 常见的数据增广



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, is_continuous=False,fix=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size,output_size)
        else:
            self.output_size = output_size
        # 保证mask 的0,1 性质
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST
        self.fix = fix

    def __call__(self, sample):
        # if not self.flag_ts:
        image = sample['image']
        h, w = image.shape[:2]
        # 输入的尺寸为 H,W 的格式
        if self.output_size == (h,w):
            return sample

        if self.fix:
            h_rate = self.output_size[0]/h
            w_rate = self.output_size[1]/w
            min_rate = h_rate if h_rate < w_rate else w_rate
            new_h = h * min_rate
            new_w = w * min_rate
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        # 直接的resize的操作实现
        img = cv2.resize(image, dsize=(new_w,new_h), interpolation=cv2.INTER_CUBIC)

        top = (self.output_size[0] - new_h)//2
        bottom = self.output_size[0] - new_h - top
        left = (self.output_size[1] - new_w)//2
        right = self.output_size[1] - new_w - left
        if self.fix:
            img = cv2.copyMakeBorder(img,top,bottom,left,right, cv2.BORDER_CONSTANT, value=[0,0,0])

        if 'segmentation' in sample.keys():
            segmentation = sample['segmentation']
            seg = cv2.resize(segmentation, dsize=(new_w,new_h), interpolation=self.seg_interpolation)

            if self.fix:
                seg = cv2.copyMakeBorder(seg,top,bottom,left,right, cv2.BORDER_CONSTANT, value=[0])
            sample['segmentation'] = seg
        sample['image'] = img
        # else:
        #     T_image = sample['T_image']
        #     S_image = sample['S_image']
        #     T_h, T_w = T_image.shape[:2]
        #     S_h, S_w = S_image.shape[:2]
        #     # 输入的尺寸为 H,W 的格式
        #     if self.output_size_T == (T_h, T_w) and self.output_size_S == (S_h, S_w):
        #         return sample
        #
        #     if self.fix:
        #         T_h_rate = self.output_size_T[0] / T_h
        #         T_w_rate = self.output_size_T[1] / T_w
        #         S_h_rate = self.output_size_S[0] / S_h
        #         S_w_rate = self.output_size_S[1] / S_w
        #         T_min_rate = T_h_rate if T_h_rate < T_w_rate else T_w_rate
        #         S_min_rate = S_h_rate if S_h_rate < S_w_rate else S_w_rate
        #         T_new_h = T_h * T_min_rate
        #         T_new_w = T_w * T_min_rate
        #         S_new_h = S_h * S_min_rate
        #         S_new_w = S_w * S_min_rate
        #     else:
        #         T_new_h, T_new_w = self.output_size_T
        #         S_new_h, S_new_w = self.output_size_S
        #     T_new_h, T_new_w = int(T_new_h), int(T_new_w)
        #     S_new_h, S_new_w = int(S_new_h), int(S_new_w)
        #     # 直接的resize的操作实现
        #     T_img = cv2.resize(T_image, dsize=(T_new_w, T_new_h), interpolation=cv2.INTER_CUBIC)
        #     S_img = cv2.resize(S_image, dsize=(S_new_w, S_new_h), interpolation=cv2.INTER_CUBIC)
        #
        #     T_top = (self.output_size_T[0] - T_new_h) // 2
        #     T_bottom = self.output_size_T[0] - T_new_h - T_top
        #     T_left = (self.output_size_T[1] - T_new_w) // 2
        #     T_right = self.output_size_T[1] - T_new_w - T_left
        #     S_top = (self.output_size_S[0] - S_new_h) // 2
        #     S_bottom = self.output_size_S[0] - S_new_h - S_top
        #     S_left = (self.output_size_S[1] - S_new_w) // 2
        #     S_right = self.output_size_S[1] - S_new_w - S_left
        #     if self.fix:
        #         T_img = cv2.copyMakeBorder(T_img, T_top, T_bottom, T_left, T_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        #         S_img = cv2.copyMakeBorder(S_img, T_top, S_bottom, S_left, S_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        #
        #     if 'segmentation' in sample.keys():
        #         T_segmentation = sample['T_segmentation']
        #         S_segmentation = sample['S_segmentation']
        #         T_seg = cv2.resize(T_segmentation, dsize=(T_new_w, T_new_h), interpolation=self.seg_interpolation)
        #         S_seg = cv2.resize(S_segmentation, dsize=(S_new_w, S_new_h), interpolation=self.seg_interpolation)
        #
        #         if self.fix:
        #             T_seg = cv2.copyMakeBorder(T_seg, T_top, T_bottom, T_left, T_right, cv2.BORDER_CONSTANT, value=[0])
        #             S_seg = cv2.copyMakeBorder(S_seg, S_top, S_bottom, S_left, S_right, cv2.BORDER_CONSTANT, value=[0])
        #         sample['T_segmentation'] = T_seg
        #         sample['S_segmentation'] = S_seg
        #     sample['image'] = S_img
        #     sample['T_image'] = T_img
        #     sample['S_image'] = S_img
        return sample

class Centerlize(object):
    def __init__(self, output_size, is_continuous=False):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if self.output_size == (h,w):
            return sample

        if isinstance(self.output_size, int):
            new_h = self.output_size
            new_w = self.output_size
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        
        top = (new_h - h) // 2  
        bottom = new_h - h - top
        left = (new_w - w) // 2
        right = new_w - w -left
        img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])   
        if 'segmentation' in sample.keys():
            segmentation = sample['segmentation'] 
            seg=cv2.copyMakeBorder(segmentation,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0])
            sample['segmentation'] = seg
        sample['image'] = img
        
        return sample
                     
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']

        image = cv2.resize(image,self.output_size)
        segmentation = cv2.resize(segmentation,self.output_size,cv2.INTER_NEAREST)

        # 原始的crop方式有问题，存在random函数，会不定时的报出尺寸大小不匹配的错误

        # h, w = image.shape[:2]
        # new_h, new_w = self.output_size   # self.output_size = 512
        # new_h = h if new_h >= h else new_h
        # new_w = w if new_w >= w else new_w
        #
        # top = np.random.randint(0, h - new_h + 1)
        # left = np.random.randint(0, w - new_w + 1)
        #
        # image = image[top: top + new_h,
        #               left: left + new_w]
        #
        # segmentation = segmentation[top: top + new_h,
        #               left: left + new_w]
        sample['image'] = image
        sample['segmentation'] = segmentation


        return sample
class RandomHSV(object):
    """Generate randomly the image in hsv space."""
    def __init__(self, h_r, s_r, v_r):
        self.h_r = h_r
        self.s_r = s_r
        self.v_r = v_r

    def __call__(self, sample):
        image = sample['image']
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h = hsv[:,:,0].astype(np.int32)
        s = hsv[:,:,1].astype(np.int32)
        v = hsv[:,:,2].astype(np.int32)
        delta_h = np.random.randint(-self.h_r,self.h_r)
        delta_s = np.random.randint(-self.s_r,self.s_r)
        delta_v = np.random.randint(-self.v_r,self.v_r)
        h = (h + delta_h)%180
        s = s + delta_s
        s[s>255] = 255
        s[s<0] = 0
        v = v + delta_v
        v[v>255] = 255
        v[v<0] = 0
        hsv = np.stack([h,s,v], axis=-1).astype(np.uint8)	
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)
        sample['image'] = image
        return sample

class RandomFlip(object):
    """Randomly flip image"""
    def __init__(self, threshold):
        self.flip_t = threshold
    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        if np.random.rand() < self.flip_t:
            image_flip = np.flip(image, axis=1)
            segmentation_flip = np.flip(segmentation, axis=1)
            sample['image'] = image_flip
            sample['segmentation'] = segmentation_flip
        return sample

class RandomRotation(object):
    """Randomly rotate image"""
    # 训练没有使用旋转，效果差别不大，所有的涉及到mask形状的改变，使用的插值算法只能使用最近邻插值实现
    def __init__(self, angle_r, is_continuous=False):
        self.angle_r = angle_r
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        row, col, _ = image.shape
        rand_angle = np.random.randint(-self.angle_r, self.angle_r) if self.angle_r != 0 else 0
        m = cv2.getRotationMatrix2D(center=(col/2, row/2), angle=rand_angle, scale=1)
        new_image = cv2.warpAffine(image, m, (col,row), flags=cv2.INTER_CUBIC, borderValue=0)
        new_segmentation = cv2.warpAffine(segmentation, m, (col,row), flags=self.seg_interpolation, borderValue=0)
        sample['image'] = new_image
        sample['segmentation'] = new_segmentation
        return sample

class RandomScale(object):
    """Randomly scale image"""
    def __init__(self, scale_r, is_continuous=False):
        self.scale_r = scale_r
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        row, col, _ = image.shape
        rand_scale = np.random.rand()*(self.scale_r - 1/self.scale_r) + 1/self.scale_r
        img = cv2.resize(image, None, fx=rand_scale, fy=rand_scale, interpolation=cv2.INTER_CUBIC)
        seg = cv2.resize(segmentation, None, fx=rand_scale, fy=rand_scale, interpolation=self.seg_interpolation)
        sample['image'] = img
        sample['segmentation'] = seg
        return sample

class Multiscale(object):
    def __init__(self, rate_list):
        self.rate_list = rate_list

    def __call__(self, sample):
        image = sample['image']
        row, col, _ = image.shape
        image_multiscale = []
        for rate in self.rate_list:
            rescaled_image = cv2.resize(image, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
            sample['image_%f'%rate] = rescaled_image
        return sample


class GtPrecise:
    def __init__(self, edge_rate):
        self.edge_rate = edge_rate

    def __call__(self, sample):
        for i in range(self.edge_rate):
            mask = sample['segmentation']
            edges = cv2.Canny(mask, 0, 1)
            edges = edges / edges.max()
            sample['segmentation'] = (mask - edges).astype(np.uint8)

        return sample

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    image h,w,c
        1. np.float32
        2. /255

    """

    def __call__(self, sample):
        key_list = sample.keys()
        for key in key_list:
            if 'image' in key or 'T_image' in key:
                image = sample[key]
                # swap color axis because
                # numpy image: H x W x C
                # torch image: C X H X W
                image = image.transpose((2,0,1))
                sample[key] = torch.from_numpy(image.astype(np.float32)/255.0)
                #sample[key] = torch.from_numpy(image.astype(np.float32)/128.0-1.0)
            elif 'segmentation' == key or 'T_segmentation' == key:
                segmentation = sample[key]
                sample[key] = torch.from_numpy(segmentation.astype(np.float32))
            elif 'segmentation_onehot' == key:
                onehot = sample['segmentation_onehot'].transpose((2,0,1))
                sample['segmentation_onehot'] = torch.from_numpy(onehot.astype(np.float32))
            elif 'mask' == key:
                mask = sample['mask']
                sample['mask'] = torch.from_numpy(mask.astype(np.float32))

            elif 'edges' == key:
                edges = sample['edges']
                sample['edges'] = torch.from_numpy(edges.astype(np.float32))

        return sample

def onehot(label, num):
    m = label
    one_hot = np.eye(num)[m]
    return one_hot
