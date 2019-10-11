# encoding: utf-8
from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import multiprocessing
#from skimage import io
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from libs.datasets.transform import *
import sys
import matplotlib.pyplot as plt
import libs.utils.test_utils as func
from tqdm import tqdm
from tqdm import tqdm

flag_debug = False
# VOCDataset('VOC2012', cfg, 'train', aug)
class RemoDataset(Dataset):
    # def __init__(self, dataset_name, cfg, period, aug, img_dir=None, mask_dir=None, txt_f=None):
    def __init__(self, cfg, period="train"):

        self.img_dir, self.mask_dir = self._createData(cfg.root_dir, cfg.txt_f)
        if hasattr(cfg, 'val_txt_f') and hasattr(cfg, 'val_root_dir'):
            self.val_img_dir, self.val_mask_dir = self._createData(cfg.val_root_dir, cfg.val_txt_f)
        else:
            self.val_img_dir, self.val_mask_dir = [], []

        self.root_path = cfg.root_dir + '/'
        self.img_save_dir = self.img_dir[0].split(self.root_path)[-1].split('/')[0]
        self.mask_save_dir = self.mask_dir[0].split(self.root_path)[-1].split('/')[0]

        self.period = period

        self.rescale = None
        self.centerlize = None
        self.randomcrop = None
        self.randomflip = None
        self.randomrotation = None
        self.randomscale = None
        self.randomhsv = None
        self.multiscale = None
        self.gt_precise = None
        self.totensor = ToTensor()
        self.cfg = cfg

        # 定义各种不同的增广

        # 重新定义crop函数，现在使用的是直接resize操作

        if cfg.DATA_RESCALE > 0:
            self.rescale = Rescale(cfg.DATA_RESCALE,fix=False)
            #self.centerlize = Centerlize(cfg.DATA_RESCALE)
        if 'train' in self.period:
            if cfg.DATA_RANDOMCROP > 0:
                self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
            if cfg.DATA_RANDOMROTATION > 0:
                self.randomrotation = RandomRotation(cfg.DATA_RANDOMROTATION)
            if cfg.DATA_RANDOMSCALE != 1:
                self.randomscale = RandomScale(cfg.DATA_RANDOMSCALE)
            if cfg.DATA_RANDOMFLIP > 0:
                self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)
            if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
                self.randomhsv = RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V)
            if cfg.DATA_gt_precise >0:
                self.gt_precise = GtPrecise(cfg.DATA_gt_precise)
        else:
            self.multiscale = Multiscale(self.cfg.TEST_MULTISCALE)

    def _createData(self, root_dir, txt_f):

        img_dir = list()
        mask_dir = list()
        if isinstance(root_dir, str) and isinstance(txt_f, str):  # yml中只输入str xml_list xml_root
            txt_f = [txt_f]
            root_dir = [root_dir]
        elif isinstance(root_dir, str) and isinstance(txt_f, list):
            root_dir = [root_dir] * len(txt_f)
        for i in range(len(txt_f)):
            img_d, mask_d = self._check_txt_list_add_root(root_dir[i], txt_f[i])
            img_dir += img_d
            mask_dir += mask_d
        return img_dir, mask_dir

    def _check_txt_list_add_root(self, root_dir, txt_f):
        img_dirs = list()
        mask_dirs = list()
        for line in open(txt_f):
            img_dir, mask_dir  = line[:-1].split(' ')
            if not img_dir.endswith('jpg') or not mask_dir.endswith('png'):
                print("error bug :%s " % img_dir)
                print("error bug :%s " % mask_dir)
                continue
            i = os.path.join(root_dir, img_dir)
            m = os.path.join(root_dir, mask_dir)
            img_dirs.append(i)
            mask_dirs.append(m)
        return img_dirs, mask_dirs

    def __len__(self):
        if self.period == 'train':
            return len(self.img_dir)
        elif self.period == 'val':
            return len(self.val_img_dir)
        elif self.period == 'test':
            return len(self.img_dir)


    def __getitem__(self, idx):

        if 'train' in self.period:
            name = self.img_dir[idx].split('/')[-1]
            image = cv2.imread(self.img_dir[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = np.array(io.imread(img_file),dtype=np.uint8)
            h,w,_ = image.shape
            # print(image.shape)
            sample = {'image': image, 'raw': image, 'name': name, 'row': h, 'col': w}


            segmentation = np.array(Image.open(self.mask_dir[idx]))
            m_h, m_w = segmentation.shape
            # if h != m_h or w != m_w:
                # print('shape is not same %s', name)

            # seg = cv2.imread(seg_file,0)
            # print(seg==segmentation)
            # sys.exit(1)
            # print(np.min(segmentation),segmentation.shape)
            # print(segmentation)

            sample['segmentation'] = segmentation

            if self.cfg.DATA_RANDOM_H>0 or self.cfg.DATA_RANDOM_S>0 or self.cfg.DATA_RANDOM_V>0:
                sample = self.randomhsv(sample)
            if self.cfg.DATA_RANDOMFLIP > 0:
                sample = self.randomflip(sample)
            if self.cfg.DATA_RANDOMROTATION > 0:
                sample = self.randomrotation(sample)
            if self.cfg.DATA_RANDOMSCALE != 1:
                sample = self.randomscale(sample)
            if self.cfg.DATA_RANDOMCROP > 0:
                sample = self.randomcrop(sample)
            if self.cfg.DATA_RESCALE > 0:
                #sample = self.centerlize(sample)
                sample = self.rescale(sample)
            if self.cfg.DATA_gt_precise >0 :
                sample = self.gt_precise(sample)

            # 产生边缘图
            if hasattr(self.cfg, 'edge_width'):
                mask = sample['segmentation']
                edges = cv2.Canny(mask, 0, 1)
                edges_o = edges
                for i in range(self.cfg.edge_width - 1): # 产生宽边缘图
                    edges = edges / edges.max()
                    mask = (mask - edges).astype(np.uint8)
                    edges = cv2.Canny(mask, 0, 1)
                    edges_o += edges
                sample["edges"] = edges_o

        elif 'test' in self.period:
            name = self.img_dir[idx].split('/')[-1]
            image = cv2.imread(self.img_dir[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = np.array(io.imread(img_file),dtype=np.uint8)
            h, w, _ = image.shape
            # print(image.shape)
            sample = {'image': image, 'raw': image, 'name': name, 'row': h, 'col': w}
            if self.cfg.DATA_RESCALE > 0:
                sample = self.rescale(sample)
            sample = self.multiscale(sample)

        elif 'val' in self.period:
            name = self.val_img_dir[idx].split('/')[-1]
            image = cv2.imread(self.val_img_dir[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = np.array(io.imread(img_file),dtype=np.uint8)
            h, w, _ = image.shape
            # print(image.shape)
            sample = {'image': image, 'raw': image, 'name': name, 'row': h, 'col': w}
            segmentation = np.array(Image.open(self.val_mask_dir[idx]))

            sample['segmentation'] = segmentation

            if self.cfg.DATA_RESCALE > 0:
                sample = self.rescale(sample)
            sample = self.multiscale(sample)

        if not flag_debug:
            sample = self.totensor(sample)


        return sample

    def collate_fn(self, batch):
        images = []
        seg = []
        edg = []
        cs = []
        rs = []
        names = []
        for _, sample in enumerate(batch):
            images.append(sample['image'])
            seg.append(sample['segmentation'])
            rs.append(sample['row'])
            cs.append(sample['col'])
            names.append(sample['name'])
        if hasattr(self.cfg, 'edge_loss_weight'):
            edg.append(sample['edges'])

            return {
                'image': torch.stack(images, 0),
                'segmentation': torch.stack(seg, 0),
                'edges': torch.stack(edg, 0),
            }
        else:
            return {
                'row': rs,
                'col': cs,
                'names': names,
                'image': torch.stack(images, 0),
                'segmentation': torch.stack(seg, 0),
            }

    def __colormap(self, N):
        """Get the map from label index to color

        Args:
            N: number of class

            return: a Nx3 matrix

        """
        cmap = np.zeros((N, 3), dtype = np.uint8)

        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

        for i in range(N):
            r = 0
            g = 0
            b = 0
            idx = i
            for j in range(7):
                str_id = uint82bin(idx)
                r = r ^ ( np.uint8(str_id[-1]) << (7-j))
                g = g ^ ( np.uint8(str_id[-2]) << (7-j))
                b = b ^ ( np.uint8(str_id[-3]) << (7-j))
                idx = idx >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        return cmap

    def label2colormap(self, label):
        m = label.astype(np.uint8)
        r,c = m.shape
        cmap = np.zeros((r,c,3), dtype=np.uint8)
        cmap[:,:,0] = (m&1)<<7 | (m&8)<<3
        cmap[:,:,1] = (m&2)<<6 | (m&16)<<2
        cmap[:,:,2] = (m&4)<<5
        return cmap

    def save_result(self, result_list, model_id):
        """Save test results

        Args:
            result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

        """
        test_img_path = self.img_dir
        i = 1
        folder_path = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for sample in result_list:
            if not sample["name"].endswith("jpg"):
                img_path = os.path.join(test_img_path, '%s.jpg'%sample['name'])
            else:
                img_path = os.path.join(test_img_path, '%s'%sample['name'])

            img = cv2.imread(img_path)
            file_path = os.path.join(folder_path, '%s.png'%sample['name'])
            # predict_color = self.label2colormap(sample['predict'])
            # p = self.__coco2voc(sample['predict'])

            # cv2.imwrite(file_path, sample['predict'])
            # mask = cv2.imread(file_path)
            mask_img = sample['predict']
            mask_img_n = np.stack((mask_img, mask_img, mask_img), axis=2)
            print(img.shape)
            print(mask_img_n.shape)
            mix_img = cv2.addWeighted(img, 0.5, mask_img_n * 255,0.5,1)
            # cv2.imwrite(file_path,mix_img)
            cv2.imshow("img", mix_img)
            k = cv2.waitKey(0)
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            # cv2.imwrite(file_path, sample['predict'])


            # print('[%d/%d] %s saved'%(i,len(result_list),file_path))
            i+=1

    def do_python_eval(self, result_list, flag_process=False):
        flag_read_gt = False
        if isinstance(result_list[0]['gt'], str):
            flag_read_gt = True

        TP = []
        P = []
        T = []
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            TP.append(multiprocessing.Value('i', 0, lock=True))
            P.append(multiprocessing.Value('i', 0, lock=True))
            T.append(multiprocessing.Value('i', 0, lock=True))
        # gt 包含与pred 比例不对的 mask, 需要重新resize
        # TODO: 写到日志里面
        def compare(start, step, TP, P, T):
            for idx in tqdm(range(start, len(result_list), step), desc='cmp miou'):
                name = result_list[idx]
                predict = name['predict']
                h, w = predict.shape

                if flag_read_gt:
                    gt_file = os.path.join(os.path.split(self.val_mask_dir[0])[0] , name['gt'].replace('jpg', 'png'))
                    gt = cv2.imread(gt_file, 0)
                    h_g, w_g = gt.shape
                    if h_g != h or w_g != w:
                        gt = cv2.resize(gt, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    gt = name['gt']


                cal = gt < 255
                mask = (predict == gt) * cal

                for i in range(self.cfg.MODEL_NUM_CLASSES):
                    P[i].acquire()
                    P[i].value += np.sum((predict == i) * cal)
                    P[i].release()
                    T[i].acquire()
                    T[i].value += np.sum((gt == i) * cal)
                    T[i].release()
                    TP[i].acquire()
                    TP[i].value += np.sum((gt == i) * mask)
                    TP[i].release()

        p_list = []
        num_process = 4
        for i in range(num_process ):
            p = multiprocessing.Process(target=compare, args=(i, num_process, TP, P, T))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()
        IoU = []
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
        return IoU


    def getTestData(self, test_ratio=0.2, save_path=None, test_name='test.txt', train_name='train.txt'):
        '''
        获得测试集 训练集。
        :param test_ratio: 切分数据集比例
        :param save_path:  保存路径

        使用方法： RemoDataset.getTestData()

        '''
        if save_path is None:
            save_path = '/home/xjx/data/mask/Kaggle/data_crop/layout'

        root_path = self.cfg.root_dir + '/'
        data_list = np.asarray(self.img_dir)



        test_dataset_num = int(len(data_list) * test_ratio)
        ind = np.random.randint(0, len(data_list), test_dataset_num).astype(np.int)
        testDataset = data_list[ind]
        trainDataset = list(set(data_list) - set(testDataset))

        with open('%s/%s' % (save_path, test_name), '+w') as f:
            for l in tqdm(testDataset):
                print(l)
                file_name = l.split('/')[-1]
                img_path = '%s/%s' % (self.img_save_dir, file_name)
                mask_path = '%s/%s' % (self.mask_save_dir, file_name.replace('jpg', 'png'))
                f.write('%s %s\n' % (img_path, mask_path))

        with open('%s/%s' % (save_path, train_name), '+w') as f:
            for l in tqdm(trainDataset):
                file_name = l.split('/')[-1]
                img_path = '%s/%s' % (self.img_save_dir, file_name)
                mask_path = '%s/%s' % (self.mask_save_dir, file_name.replace('jpg', 'png'))
                f.write('%s %s\n' % (img_path, mask_path))


def maskAddImg(img, mask):
    mask_img_n = np.stack((mask, mask, mask), axis=2)
    mix_img = cv2.addWeighted(img, 0.5, mask_img_n * 255, 0.5, 1)
    return mix_img

def check_data():
    ''' !!!!!!!!!!!!! 使用checkdata 需要关闭dataset中的toTensor !!!!!!!!!!!!!! '''
    global flag_debug
    flag_debug = True
    import config.remo_unet as config
    import cv2
    cfg = config.Configuration()
    cfg.txt_f = [
                #"/home/zhangming/Datasets/mask/Kaggle/rematch/data_s256/image_10_ratio1/layout.txt",
'/home/zhangming/Datasets/mask/Kaggle/rematch/data_s256/image_11_ratio1/layout.txt', 
"/home/zhangming/Datasets/mask/Kaggle/rematch/data_s256/image_20_ratio1/layout.txt", 
"/home/zhangming/Datasets/mask/Kaggle/rematch/data_s256/image_21_ratio1/layout.txt"
              
                 ]

    cfg.root_dir = '/home/zhangming/Datasets/mask/Kaggle/rematch/data_s256'
    # cfg.root_dir = '/home/xjx/data/mask/Kaggle/data'
    # cfg.txt_f = "/home/xjx/data/mask/seg_coco_LIP/layout/train_mosaic.txt"
    datasets = RemoDataset(cfg, 'train')
    datasets.getTestData(0.2, save_path='/home/zhangming/Datasets/mask/Kaggle/rematch/data_s256', test_name='val_1.txt', train_name='train_1.txt')
    flag_no_totensor = False
    if flag_no_totensor:
        for indx in range(len(datasets)):
            sample = datasets[indx]  # __getitem__
            img = sample['image']
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mask = sample['segmentation']

            # sample = {'image': image, 'raw': image, 'name': name, 'row': r, 'col': c}
            edges = sample['edges']

            edges_t = cv2.Canny(mask, 0, 1)
            edges_t = (edges_t / edges_t.max()).astype(np.uint8)
            mask_t = (mask - (edges / edges.max())).astype(np.uint8)
            try:
                mix_img = maskAddImg(img, mask)
                mix_img_edge = maskAddImg(img, mask_t)
            except:
                print(sample['name'])

            # cv2.imshow('mix_img', mix_img)

            mix_img_s = np.concatenate((img, mix_img, mix_img_edge), 1)

            # save = "/home/xjx/data/mask/Remo_seg_coco_Dataset/" + sample['name']
            # cv2.imwrite(save, mix_img_s)
            cv2.imshow('mix_img', mix_img_s )
            # cv2.imshow('edges', edges )
            k = cv2.waitKey(0)
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
        cv2.destroyAllWindows()




if __name__ == '__main__':

    check_data()

