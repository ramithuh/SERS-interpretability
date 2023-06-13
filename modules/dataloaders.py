import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import glob
import torchvision
import matplotlib.pyplot as plt

def get_sers_dataloaders(torch_seed=10, data_dir= '/n/holyscratch01/wadduwage_lab/ramith/SERS'):
    '''
        Function to return train, validation,test SERS Bacteria dataloaders
        Args:
            train_batch_size : batch size for training
            torch_seed       : seed
            data_dir         : data directory which has the data hierachy as `./test/klebsiella pneumoniae/klebsiella pneumoniae_1.npy`

        Returns:
            train_loader : Data loader for training
            val_loader   : Data loader for validation
            test_loader  : Data loader for testing
    '''
    
    torch.manual_seed(torch_seed)

    train_data = sers_bacteria_dataset(data_dir = data_dir, type_= 'train')
    val_data   = sers_bacteria_dataset(data_dir = data_dir, type_= 'val')
    test_data  = sers_bacteria_dataset(data_dir = data_dir, type_= 'test')
    
    train_loader = DataLoader(train_data, batch_size = 4, shuffle=True, drop_last= True, num_workers=2)
    val_loader   = DataLoader(val_data, batch_size  = 2, shuffle=True, drop_last= True, num_workers=2)
    test_loader  = DataLoader(test_data, batch_size = 2, shuffle=True, drop_last= True, num_workers=2)

    return train_loader, val_loader, test_loader


class sers_bacteria_dataset(torch.utils.data.Dataset):
    '''
        A standard dataset class to get the bacteria dataset
        
        Args:
            data_dir   : data directory which contains data hierarchy
            type_      : whether dataloader if train/ val 
            transform  : torchvision.transforms
            label_type : There are multiple types of classification in bacteria dataset
                         therefore, specify which label you need as follows:

                            | label_type              | Description
                            |------------------------ |---------------
                            | 'class' (default)       | Strain (0-20)
                            | 'antibiotic_resistant'  | Non wild type (1) / Wild type (0)
                            | 'gram_strain'           | Gram Positive (1) / Gram Negative (0)
                            | 'species'               | Species (0-4)

            balance_data    : If true, dataset will be balanced by the minimum class count (default: False)
            expand_channels : If true, bacteria image will be copied to 3 channels  (default: False)
                              (used for some predefined backbones which need RGB images)
    '''
    
    def __init__(self, data_dir, type_):
    
        self.type_ = type_

        self.img_dirs =  glob.glob(f'{data_dir}/{type_}/*/*')

        print(f"Dataset type {type_}", end = " -> ")
        print(f"Loaded {len(self.img_dirs)} images")

        
    def __len__(self):
        return len(self.img_dirs)
        
    def __getitem__(self, idx): 
        wave  = np.load(self.img_dirs[idx], allow_pickle=True).astype(float)

        label = self.img_dirs[idx].split('/')[-2]
        
        return wave, label


