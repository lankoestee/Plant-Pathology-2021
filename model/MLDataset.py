from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MultiLabelDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(dataset_path + '/labels.csv')
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, item):
        # 读取dataset中images列的图片，其label为后面所有列的值
        img = Image.open(self.dataset_path + '/images/' + self.dataset.iloc[item, 0])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.dataset.iloc[item, 0]))
        label = self.dataset.iloc[item, 1:].values.astype('float32')
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.from_numpy(np.array(labels))
        return images, labels
        
