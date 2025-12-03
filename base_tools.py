import os
import pickle
import sys
import shutil

import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def seed_torch(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    cudnn.deterministic = True
    cudnn.benchmark = False


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def dump_pkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def dir_init(config):
    makedirs(config.logging_dir)

    if config.model:
        makedirs(config.model_dir)

def normalize_labels(ys, flip=False):
    assert type(ys) == np.ndarray
    y_max = np.max(ys)
    y_min = np.min(ys)
    ys_norm = (ys - y_min) / (y_max - y_min)
    if flip:
        ys_norm = 1 - ys_norm
    return ys_norm * 10.


class Logger(object):

    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def Logger_init(filename):
    sys.stdout = Logger(filename, sys.stdout)
    sys.stderr = Logger(filename, sys.stderr)


def get_db_base_info(root, dataset):
    folder_path = {
        'live': '/LIVE/',
        'csiq': '/CSIQ/',
        'tid2013': '/TID2013/',
        'kadid-10k': '/KADID_10k/',
        'livec': '/ChallengeDB_release/',
        'koniq-10k': '/KonIQ_10k/',
        'bid': '/BID_512/',
        'pipal': '/PIPAL/',
        'spaq': '/SPAQ_384/',
    }
    folder_path = {key: root + val for key, val in folder_path.items()}[dataset]

    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'tid2013': list(range(0, 25)),
        'kadid-10k': sorted(list(set(list(range(0, 81))) - {5, 6, 17, 18, 52, 57, 69})),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'bid': list(range(0, 586)),
        'pipal': list(range(0, 200)),
        'spaq': list(range(0, 11125)),
    }
    sel_num = img_num[dataset]
    return folder_path, sel_num


def save_checkpoint(state, is_best, checkpoint, model_best):
    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, model_best)
