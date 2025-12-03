import random
import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
from openpyxl import load_workbook
import pickle
import pandas as pd
from base_tools import *
from scipy.spatial.distance import cdist, pdist, squareform
from DDCUp import upsample


class LIVEFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num):

        refpath = os.path.join(root, 'refimgs')
        refname = getFileName(refpath, '.bmp')

        jp2kroot = os.path.join(root, 'jp2k')
        jp2kname = self.getDistortionTypeFileName(jp2kroot, 227)

        jpegroot = os.path.join(root, 'jpeg')
        jpegname = self.getDistortionTypeFileName(jpegroot, 233)

        wnroot = os.path.join(root, 'wn')
        wnname = self.getDistortionTypeFileName(wnroot, 174)

        gblurroot = os.path.join(root, 'gblur')
        gblurname = self.getDistortionTypeFileName(gblurroot, 174)

        fastfadingroot = os.path.join(root, 'fastfading')
        fastfadingname = self.getDistortionTypeFileName(fastfadingroot, 174)

        imgpath = jp2kname + jpegname + wnname + gblurname + fastfadingname

        info = scipy.io.loadmat(os.path.join(root, 'dmos_realigned.mat'))
        dmos = info['dmos_new'].astype(np.float32)[0]
        labels = normalize_labels(dmos, flip=True)

        orgs = info['orgs']
        refnames_all = scipy.io.loadmat(os.path.join(root, 'refnames_all.mat'))
        refnames_all = refnames_all['refnames_all']

        sample = []
        for i in range(0, len(index)):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = train_sel * ~orgs.astype(np.bool_)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[1].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((imgpath[item], labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return path, sample, target

    def __len__(self):
        length = len(self.samples)
        return length

    def getDistortionTypeFileName(self, path, num):
        filename = []
        index = 1
        for i in range(0, num):
            name = '%s%s%s' % ('img', str(index), '.bmp')
            filename.append(os.path.join(path, name))
            index = index + 1
        return filename


class LIVEChallengeFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num):

        imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        imgpath = imgpath['AllImages_release']
        imgpath = imgpath[7:1169]
        info = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        mos = info['AllMOS_release'].astype(np.float32)
        mos = mos[0][7:1169]
        labels = normalize_labels(mos)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'Images', imgpath[item][0][0]), labels[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return path, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class CSIQFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num):

        data_info = scipy.io.loadmat(os.path.join(root, 'CSIQ_info.mat'))
        std_info = scipy.io.loadmat(os.path.join(root, 'csiq_std_mos.mat'))

        dst_name_array = data_info['dst_name'][:, 0]
        dmos = data_info['mos'][:, 0].astype(np.float32)
        labels = normalize_labels(dmos, flip=True)

        refpath = os.path.join(root, 'src_imgs')
        refname = getFileName(refpath, '.png')

        imgnames = []
        refnames_all = []
        for i in range(dst_name_array.size):
            dst_name = dst_name_array[i][0]
            imgnames.append(dst_name)
            ref_temp = dst_name.split('/')[-1].split(".")
            refnames_all.append(ref_temp[0] + '.' + ref_temp[-1])
        refnames_all = np.array(refnames_all)

        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append((os.path.join(root, imgnames[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)

        return path, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class Koniq_10kFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        data = pd.read_csv(os.path.join(root, 'koniq10k_scores_and_distributions.csv'))
        imgname = data['image_name'].tolist()
        mos = data['MOS'].values.astype(np.float32)  # to_numpy()
        labels = normalize_labels(np.array(mos))

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                # sample.append((os.path.join(root, '1024x768', imgname[item]), mos_all[item]))
                sample.append((os.path.join(root, '512x384', imgname[item]), labels[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return path, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class BIDFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        info = pd.read_excel(os.path.join(root, 'DatabaseGrades.xlsx'))
        img_num = info['Image Number'].tolist()  # old version: tolist(); new version: to_list()
        imgname = ["DatabaseImage%04d.JPG" % (i) for i in img_num]

        mos = info['Average Subjective Grade'].values.astype(
            np.float32)  # old version: value(); new version: to_numpy()
        labels = normalize_labels(mos)

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                sample.append((os.path.join(root, imgname[item]), labels[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return path, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class TID2013Folder(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        refpath = os.path.join(root, 'reference_images')
        refname = getTIDFileName(refpath, '.bmp.BMP')

        imgnames = []
        target = []
        refnames_all = []
        with open(os.path.join(root, 'mos_with_names.txt'), 'r') as f:
            content = f.readlines()
        for line in content:
            line = line.split('\n')
            words = line[0].split()
            imgnames.append((words[1]))
            target.append(words[0])
            ref_temp = words[1].split("_")
            refnames_all.append(ref_temp[0][1:])

        refnames_all = np.array(refnames_all)
        mos = np.array(target).astype(np.float32)
        labels = normalize_labels(mos)

        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append(
                        (os.path.join(root, 'distorted_images', imgnames[item]), labels[item]))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return path, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class KADID_10kFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num, filtered_idx=None, test_db=None):
        refname = ['I%02d.png' % i for i in range(1, 82)]
        data = pd.read_csv(os.path.join(root, 'dmos.csv'))

        dist_lvs = np.array(range(1, 6)).reshape([1, -1])

        if test_db in ['live', 'csiq', 'tid2013', 'kadid-10k']:
            self.sel_types = np.array(range(1, 26)).reshape([-1, 1])
        elif test_db == 'livec':
            self.sel_types = np.array([1, 3, 10, 17, 18, 25]).reshape([-1, 1])
        elif test_db == 'koniq':
            self.sel_types = np.array([1, 3, 16, 17, 18, 25]).reshape([-1, 1])
        elif test_db == 'bid':
            self.sel_types = np.array([1, 3, 9, 21, 23, 25]).reshape([-1, 1])
        elif test_db == 'pipal1':
            self.sel_types = np.array([1, 2, 3, 9, 17, 18, 25]).reshape([-1, 1])
        elif test_db == 'pipal2':
            self.sel_types = np.array([1, 2, 3, 9, 17, 23]).reshape([-1, 1])
        elif test_db == 'pipal3':
            self.sel_types = np.array([1, 3, 16, 17, 18, 19, 25]).reshape([-1, 1])
        elif test_db == 'pipal4':
            self.sel_types = np.array([1, 3, 7, 9, 10, 17, 18, 25]).reshape([-1, 1])
        elif test_db == 'pipal5':
            self.sel_types = np.array([1, 2, 3, 4, 9]).reshape([-1, 1])
        elif test_db == 'pipal6':
            self.sel_types = np.array([1, 2, 3, 9, 10, 11, 12, 16, 17, 18, 19, 21, 22, 25]).reshape([-1, 1])

        sel_imgs = np.array(range(81)).reshape([-1, 1])
        sel_dists = dist_lvs + (self.sel_types - 1) * 5 - 1
        sel_idx = sel_dists.reshape([1, -1]) + sel_imgs * 125
        sel_idx = sel_idx.flatten().tolist()

        if filtered_idx is not None:
            sel_idx = np.array(sel_idx)[filtered_idx].tolist()

        imgnames = data.loc[sel_idx, 'dist_img'].tolist()
        refnames_all = data.loc[sel_idx, 'ref_img'].values
        mos = data.loc[sel_idx, 'dmos'].values.astype(np.float32)  # mos
        labels = normalize_labels(mos)

        sample = []
        for i, item in enumerate(index):
            train_sel = (refname[index[i]] == refnames_all)
            train_sel = np.where(train_sel == True)
            train_sel = train_sel[0].tolist()
            for j, item in enumerate(train_sel):
                for aug in range(patch_num):
                    sample.append(
                        (imgnames[item], os.path.join(root, 'images', imgnames[item]), labels[item]))

        self.samples = sample
        self.transform = transform

        if len(index) > 20 and filtered_idx is not None and test_db is not None:
            cache_path = 'additional_samples.pkl'

            additional_samples = upsample(
                res_path='RefSet',
                kadid_root=root,
                sel_types=self.sel_types,
                patch_num=patch_num,
                cache_path=cache_path,
            )
            self.samples += additional_samples

    def __getitem__(self, index):
        imgname, path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return imgname, sample, target

    def __len__(self):
        length = len(self.samples)
        return length


class PIPALFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num, sel_types=[0, 1, 2, 3, 4, 5, 6]):
        # dist_dict = {'trad': range(12), 'trad_SR': range(16), 'PSNR_SR': range(10), 'SR_mismatch': range(24),
        #              'GAN_SR': range(13), 'Denoising': range(14), 'SR_Denoising': range(27)}
        # dist_sub_type = {0: 'trad', 1: 'trad_SR', 2: 'PSNR_SR', 3: 'SR_mismatch', 4: 'GAN_SR', 5: 'Denoising',
        #                  6: 'SR_Denoising'}
        info_root = os.path.join(root, 'train', 'Train_Label')
        info_txt = [os.path.join(info_root, file) for file in sorted(os.listdir(info_root))]

        names = []
        scores = []
        for i in index:
            with open(info_txt[i], 'r') as f:
                content = f.readlines()
            for line in content:
                name, score = line.strip().split(',')
                _, dis_type, _ = name.split('_')
                if int(dis_type) in sel_types:
                    names.append(name)
                    scores.append(score)

        mos = np.array(scores).astype(np.float32)
        labels = normalize_labels(mos)

        sample = []
        for i, name in enumerate(names):
            for aug in range(patch_num):
                sample.append((os.path.join(root, 'train', 'Distortion', name), labels[i], mos[i], 0))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target, mos, std = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return path, sample, target, mos, std

    def __len__(self):
        length = len(self.samples)
        return length


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


