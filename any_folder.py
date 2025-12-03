import os
import torchvision
import torch.utils.data as data
from PIL import Image


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(os.path.join(path, i))
    return filename


class AnyFolder(data.Dataset):
    def __init__(self, path, suffix, patch_size):
        if type(path) == list:
            self.samples = path
        else:
            self.samples = getFileName(path, suffix)
        assert len(self.samples) > 0
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(size=patch_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        path = self.samples[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        return path.split('/')[-1], self.transform(sample)

    def __len__(self):
        length = len(self.samples)
        return length

