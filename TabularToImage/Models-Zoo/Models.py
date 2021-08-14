import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision.transforms import *
from torch.utils.data import DataLoader
import torch
import numpy as np
from collections import namedtuple
import pandas as pd
import time
import os
import copy
#from autogluon.TablarToImage import  Utils
class TablarToImage():  
    def __init__(self, ImageShape,model_type, num_classes, pretrained=True):  
        self.ImageShape = ImageShape 
        self.model_type=model_type
        self.num_classes=num_classes
        self.pretrained=True
        use_gpu = torch.cuda.is_available() 
    
    
      
    def create_model(self,self.model_type, self.num_classes, self.pretrained=True):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if   self.ImageShape=='224':
            if 'resnet18' == self.model_type:
                model = models.resnet18(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, num_classes).double()
            elif 'resnet34' == self.model_type:
                model = models.resnet34(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, num_classes).double()
            elif 'resnet50' == self.model_type:
                model = models.resnet50(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, num_classes).double()
            elif 'resnet101' == self.model_type:
                model = models.resnet101(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, num_classes).double()
            elif 'resnet152' == self.model_type:
                model = models.resnet152(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, num_classes).double()
            elif 'alexnet' == self.model_type:
                model = models.alexnet(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[6] = nn.Linear(4096, num_classes).double()
            elif 'vgg11' == self.model_type:
                model = models.vgg11(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes).double()
            elif 'vgg11_bn' == self.model_type:
                model = models.vgg11_bn(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes).double()
            elif 'vgg13' == self.model_type:
                model = models.vgg13(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False    
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes).double()
            elif 'vgg13_bn' == self.model_type:
                model = models.vgg13_bn(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes).double()
            elif 'vgg16' == self.model_type:
                model = models.vgg16(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes).double()
            elif 'vgg16_bn' == self.model_type:
                model = models.vgg16_bn(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes).double()
            elif 'vgg19' == self.model_type:
                model = models.vgg19(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes).double()
            elif 'vgg19_bn' == self.model_type:
                model = models.vgg19_bn(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes).double()
            elif 'densenet121' == self.model_type:
                model = models.densenet121(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier = nn.Linear(model.classifier.in_features, num_classes).double()
            elif 'densenet161' == self.model_type:
                model = models.densenet161(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier = nn.Linear(model.classifier.in_features, num_classes).double()
            elif 'densenet169' == self.model_type:
                model = models.densenet169(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier = nn.Linear(model.classifier.in_features, num_classes).double()
            elif 'densenet201' == self.model_type:
                model = models.densenet201(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier = nn.Linear(model.classifier.in_features, num_classes).double()
            elif 'googlenet' == self.model_type:
                model = models.googlenet(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.fc = nn.Linear(model.fc.in_features, num_classes).double()
            elif 'shufflenet_v2_x0_5' == self.model_type:
                model = models.shufflenet_v2_x0_5(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.fc = nn.Linear(model.fc.in_features, num_classes).double()
            elif 'shufflenet_v2_x1_0' == self.model_type:
                model = models.shufflenet_v2_x0_0(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.fc = nn.Linear(model.fc.in_features, num_classes).double()
            elif 'mobilenet_v2' == self.model_type:
                model = models.mobilenet_v2(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes).double()
            elif 'wide_resnet50_2' == self.model_type:
                model = models.wide_resnet50_2(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.fc = nn.Linear(model.fc.in_features, num_classes).double()
            elif 'wide_resnet101_2' == self.model_type:
                model = models.wide_resnet101_2(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.fc = nn.Linear(model.fc.in_features, num_classes).double
            elif 'mnasnet0_5' == self.model_type:
                model = models.wide_resnet0_5(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            elif 'mnasnet1_0' == self.model_type:
                model = models.mnasnet1_0(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes).double()         
        elif self.ImageShape=='227':
            if 'squeezenet1_0' == self.model_type:
                model = models.squeezenet1_0(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1)).double()
                    model.num_classes = num_classes
            elif 'squeezenet1_1' == self.model_type:
                model = models.squeezenet1_1(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1)).double()
                model.num_classes = num_classes
        elif self.ImageShape=='256':
            if 'resnext50_32x4d' == self.model_type:
                model = models.resnext50_32x4d(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes).double()
            elif 'resnext101_32x8d' == self.model_type:
                model = models.resnext50_32x8d(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False 
                model.fc = nn.Linear(model.fc.in_features, num_classes).double()         
        elif self.ImageShape=='299':
            if 'inception_v3' == self.model_type:
                model = models.inception_v3(pretrained=pretrained).to(device).double()
                for param in model.parameters():
                    param.requires_grad = False
                model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes).double()
                model.fc = nn.Linear(model.fc.in_features, num_classes).double()
        return model.double().to(device)
    
    def train(self,dataloader, model, num_epochs=20):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Rprop(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

        model.train(True)
        results = []
        for epoch in range(num_epochs):
            optimizer.step()
            scheduler.step()
            model.train()

            running_loss = 0.0
            running_corrects = 0

            n = 0
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                n += len(labels)

            epoch_loss = running_loss / float(n)
            epoch_acc = running_corrects.double() / float(n)

            print(f'epoch {epoch}/{num_epochs} : {epoch_loss:.5f}, {epoch_acc:.5f}')
            results.append(EpochProgress(epoch, epoch_loss, epoch_acc.item()))
        return pd.DataFrame(results)

    def optimizer(self):
        if self.model_type in []:
            #optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
            optimizer = optim.SGD(self.create_model(self.model_type.self.num_classess,True).parameters(), lr=0.001, momentum=0.9)
            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif self.model_type in []:
            optimizer=torch.optim.RMSprop(self.create_model(self.model_type.self.num_classess,True), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            # Decay LR by a factor of 0.1 every 7 epochs
            exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        elif self.model_type in []:
            optimizer = optim.Adam(self.create_model(self.model_type.self.num_classess,True).parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
            exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience =  5, mode = 'max', verbose=True)    
        criterion = nn.CrossEntropyLoss()    
  
    def train_model(self,model, num_epochs=3):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Rprop(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        avg_loss = 0
        avg_acc = 0
        avg_loss_val = 0
        avg_acc_val = 0
        
        train_batches = len(trainloader)
        val_batches = len(valloader)
        
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs))
            print('-' * 10)
            
            loss_train = 0
            loss_val = 0
            acc_train = 0
            acc_val = 0
            
            model.train(True)
            
            for i, data in enumerate(trainloader):
                if i % 100 == 0:
                    print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)
                    
                # Use half training dataset
                #if i >= train_batches / 2:
                #    break
                    
                inputs, labels = data
                
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                #loss_train += loss.data[0]
                loss_train += loss.item() * inputs.size(0)
                acc_train += torch.sum(preds == labels.data)
                
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
            
            print()
            # * 2 as we only used half of the dataset
            avg_loss = loss_train * 2 /68154#len(X_train_img) #dataset_sizes[TRAIN]
            avg_acc = acc_train * 2 /68154#len(X_train_img)#dataset_sizes[TRAIN]
            
            model.train(False)
            model.eval()
                
            for i, data in enumerate(valloader):
                if i % 100 == 0:
                    print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)
                    
                inputs, labels = data
                
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
                else:
                    inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                #loss_val += loss.data[0]
                loss_val += loss.item() * inputs.size(0)
                acc_val += torch.sum(preds == labels.data)
                
                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()
            
            avg_loss_val = loss_val /22718#len(X_val_img) #dataset_sizes[VAL]
            avg_acc_val = acc_val /22718#len(X_val_img) #dataset_sizes[VAL]
            
            print()
            print("Epoch {} result: ".format(epoch))
            print("Avg loss (train): {:.4f}".format(avg_loss))
            print("Avg acc (train): {:.4f}".format(avg_acc))
            print("Avg loss (val): {:.4f}".format(avg_loss_val))
            print("Avg acc (val): {:.4f}".format(avg_acc_val))
            print('-' * 10)
            print()
            
            if avg_acc_val > best_acc:
                    best_acc = avg_acc_val
                    best_model_wts = copy.deepcopy(model.state_dict())
                
            elapsed_time = time.time() - since
            print()
            print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
            print("Best acc: {:.4f}".format(best_acc))
            
            model.load_state_dict(best_model_wts)
            return model
    
    def plot_results(df, figsize=(10, 5)):
        fig, ax1 = plt.subplots(figsize=figsize)

        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color='tab:red')
        ax1.plot(df['epoch'], df['loss'], color='tab:red')

        ax2 = ax1.twinx()
        ax2.set_ylabel('accuracy', color='tab:blue')
        ax2.plot(df['epoch'], df['accuracy'], color='tab:blue')

        fig.tight_layout()

np.random.seed(37)
torch.manual_seed(37)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_classes = 3
pretrained = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

EpochProgress = namedtuple('EpochProgress', 'epoch, loss, accuracy')

transform = transforms.Compose([Resize(224), ToTensor()])
image_folder = datasets.ImageFolder('./shapes/train', transform=transform)
dataloader = DataLoader(image_folder, batch_size=4, shuffle=True, num_workers=4)
    

