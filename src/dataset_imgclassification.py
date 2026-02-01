from torch.utils.data import Dataset
import os
import pickle
import numpy as np
class Dataset_Classification(Dataset):
    def __init__(self,root='data',train = True,transform = None):
        data_path = os.path.join(root,'dataset')
        if train:
            data_file = [os.path.join(data_path,'data_batch_{}'.format(i)) for i in range(1,6)]
        else:
            data_file = [os.path.join(data_path,"test_batch")]
        self.images = []
        self.labels = []
        for data_file in data_file:
            with open(data_file,"rb") as fo:
                dict = pickle.load(fo,encoding = 'bytes')
                self.images.extend(dict[b'data'])
                self.labels.extend(dict[b'labels'])
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        image = self.images[idx].reshape((3,32,32)).astype(np.float32)
        label = self.labels[idx]
        return image/255.,label