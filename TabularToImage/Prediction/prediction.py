import torch
#device = torch.device("cuda") #device = 'cuda'
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from autogluon.TablarToImage import  Utils

class Predictions:
    def init(self,**kwargs):
        self._validate_init_kwargs(kwargs)
        Utils_type = kwargs.pop('Utils_type', Utils)
        Utils_kwargs = kwargs.pop('Utils_kwargs', dict())
        X_train_img  = kwargs.get('X_train_img', None)
        X_val_img = kwargs.get('X_val_img', None)
        X_test_img = kwargs.get('X_test_img', None)
        y_train = kwargs.get('y_train', None)
        y_val = kwargs.get('y_val', None)
        y_test = kwargs.get('y_test', None)
        
        self._Utils: Utils = Utils_type(X_train_img =X_train_img ,X_val_img=X_val_img ,X_test_img=X_test_img
                                   ,y_train=y_train,y_val=y_val,y_test=y_test,**Utils_kwargs)
        self._Utils_type = type(self._Utils)
        #self._trainer = None
    @property
    def X_train_img(self):
        return self._Utils.X_train_img    
    @property
    def X_val_img(self):
        return self._Utils.X_val_img  
    @property
    def X_test_img(self):
        return self._Utils.X_test_img   
    @property
    def y_train(self):
        return self._Utils.y_train  
    @property
    def y_val(self):
        return self._Utils.y_val
    @property
    def y_test(self):
        return self._Utils.y_test
    
    @staticmethod
    def _validate_init_kwargs(kwargs):
        valid_kwargs = {
            'Utils_type',
            'Utils_kwargs',
            'X_train_img',
            'X_val_img',
            'X_test_img',
            'y_train',
            'y_val',
            'y_test'
        }
        invalid_keys = []
        for key in kwargs:
            if key not in valid_kwargs:
                invalid_keys.append(key)
        if invalid_keys:
            raise ValueError(f'Invalid kwargs passed: {invalid_keys}\nValid kwargs: {list(valid_kwargs)}')
        