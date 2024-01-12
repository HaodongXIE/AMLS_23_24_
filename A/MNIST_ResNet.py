#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy,BinaryAccuracy

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# # Load dataset

# In[2]:


def concate_xy(x,y):
    train_data = []
    for x_,y_ in zip(x,y):
       train_data.append([x_[None,:]/255, y_])
    return train_data
dataset = np.load('pneumoniamnist.npz')
keys = list(dataset.keys())
data_params = {'batch_size': 64,
               'shuffle': True,
               'num_workers': 4}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_epochs = 10
X_train,X_val,X_test,y_train,y_val,y_test, = dataset[keys[0]],dataset[keys[1]],dataset[keys[2]],dataset[keys[3]],dataset[keys[4]],dataset[keys[5]]
# Generators
training_generator = DataLoader(concate_xy(X_train,y_train), **data_params)
validation_generator = DataLoader(concate_xy(X_val,y_val), **data_params)
test_generator = DataLoader(concate_xy(X_test,y_test), **data_params)


# In[3]:


#check the training test validation set
print(y_train.mean(),y_val.mean(),y_test.mean())
X_train.shape,X_val.shape,X_test.shape


# In[4]:


pos_weight = 0.2
print('using class weights:',pos_weight)


# In[5]:


plt.imshow(dataset[keys[0]][2],cmap='gray', vmin=0, vmax=255)


# # Build Models

# In[42]:


# In[67]:


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
#Explain

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 1)
        #self.layer2 = self._make_layer(block, 256, layers[2], stride = 1)
        #self.layer3 = self._make_layer(block, 512, layers[3], stride = 1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(8192, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# In[68]:

print('Training model on PneumoniaMNST:')

torch.manual_seed(0) 
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor((float(pos_weight),)).to(device))
model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes =1).to(device)


optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
max_epoch = 50
# Loop over epochs
loss_hist = []
loss_val_hist = []
acc_hist = []
for epoch in tqdm(range(max_epoch)):
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        optimizer.zero_grad()
        local_batch, local_labels = local_batch.to(device).float(), local_labels.to(device).float()
        # Model computations
        loss = loss_fn(model(local_batch),local_labels)
        loss.backward()
        optimizer.step()
        # Loop over epochs
        loss_hist.append(loss.item())
    # Validation
    loss =0 
    total_size = 0
    acc  = 0
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device).float(), local_labels.to(device).float()
            pred_raw = model(local_batch)
            size = local_batch.shape[0]
            loss += loss_fn(pred_raw,local_labels).item()*size
            total_size += size 
            acc += ((F.sigmoid(pred_raw)>0.5)==local_labels).sum().item()
    acc_hist.append(acc/total_size)
    loss_val_hist.append(loss/total_size)
            # Model computations


# In[69]:


_, ax = plt.subplots(1,2,figsize = (16,6))
ax[0].plot(np.linspace(0,max_epoch,len(loss_hist)),loss_hist,label='test')
ax[0].plot(np.linspace(0,max_epoch,len(loss_val_hist)),loss_val_hist,label='val')
ax[0].legend()
ax[0].grid()

ax[1].plot(np.linspace(0,max_epoch,len(acc_hist)),acc_hist,label='val_acc')
ax[1].legend()
ax[1].grid()

plt.savefig('pneumonia_Residual.pdf',dpi=100)
plt.show()


# In[70]:


total_size = 0
acc  = 0
with torch.set_grad_enabled(False):
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device).float(), local_labels.to(device).float()
        pred_raw = model(local_batch)
        size = local_batch.shape[0]
        loss += loss_fn(pred_raw,local_labels).item()*size
        total_size += size 
        acc += ((F.sigmoid(pred_raw)>0.5)==local_labels).sum().item()
print('Accuracy on the train_set using Residual Net: ', acc/total_size)


# In[71]:


total_size = 0
acc  = 0
with torch.set_grad_enabled(False):
    for local_batch, local_labels in validation_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device).float(), local_labels.to(device).float()
        pred_raw = model(local_batch)
        size = local_batch.shape[0]
        loss += loss_fn(pred_raw,local_labels).item()*size
        total_size += size 
        acc += ((F.sigmoid(pred_raw)>0.5)==local_labels).sum().item()
print('Accuracy on the validation_set using Residual Net: ', acc/total_size)


# In[72]:


total_size = 0
acc  = 0
with torch.set_grad_enabled(False):
    for local_batch, local_labels in test_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device).float(), local_labels.to(device).float()
        pred_raw = model(local_batch)
        size = local_batch.shape[0]
        loss += loss_fn(pred_raw,local_labels).item()*size
        total_size += size 
        acc += ((F.sigmoid(pred_raw)>0.5)==local_labels).sum().item()
print('Accuracy on the test_set using Residual Net: ', acc/total_size)


# # PathMNIST

# In[74]:


def concate_xy(x,y):
    train_data = []
    for x_,y_ in zip(x,y):
       train_data.append([x_.T/255, y_])
    return train_data
dataset = np.load('Datasets/pathmnist.npz')
keys = list(dataset.keys())
data_params = {'batch_size': 64,
               'shuffle': True,
               'num_workers': 4}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_epochs = 10
X_train,X_val,X_test,y_train,y_val,y_test, = dataset[keys[0]],dataset[keys[1]],dataset[keys[2]],dataset[keys[3]],dataset[keys[4]],dataset[keys[5]]
# Generators
training_generator = DataLoader(concate_xy(X_train,y_train), **data_params)
validation_generator = DataLoader(concate_xy(X_val,y_val), **data_params)
test_generator = DataLoader(concate_xy(X_test,y_test), **data_params)


# In[75]:


#check the training test validation set
X_train.shape,X_val.shape,X_test.shape


# In[76]:


# compute balance
from collections import Counter
cnt = Counter(y_train[:,0])
print(cnt)
total = sum(cnt.values())
cnt = {key: total/val for key,val in cnt.items()}
cnt = dict(sorted(cnt.items()))
cnt


# In[77]:


pos_weight = torch.Tensor(list(cnt.values())).cuda()
pos_weight


# In[78]:


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
#Explain

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 1)
        #self.layer2 = self._make_layer(block, 256, layers[2], stride = 1)
        #self.layer3 = self._make_layer(block, 512, layers[3], stride = 1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(8192, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# In[79]:
print('Training model on PathMNST')

torch.manual_seed(0) 
loss_fn = torch.nn.CrossEntropyLoss(weight =pos_weight )
model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes =9).to(device)#CNNClassifier().cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
max_epoch = 60
metric = MulticlassAccuracy( num_classes=9)
# Loop over epochs
loss_hist = []
loss_val_hist = []
acc_hist = []
for epoch in tqdm(range(max_epoch)):
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        optimizer.zero_grad()
        local_batch, local_labels = local_batch.to(device).float(), local_labels[:,0].to(device).long()
        # Model computations
        loss = loss_fn(model(local_batch),local_labels)
        loss.backward()
        optimizer.step()
        # Loop over epochs
        loss_hist.append(loss.item())
    # Validation
    loss =0 
    total_size = 0
    acc  = 0
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device).float(), local_labels[:,0].to(device).long()
            pred_raw = model(local_batch)
            size = local_batch.shape[0]
            loss += loss_fn(pred_raw,local_labels).item()*size
            total_size += size 
            metric.update(pred_raw,local_labels)
            acc += metric.compute().item() *size
    acc_hist.append(acc/total_size)
    loss_val_hist.append(loss/total_size)
    if (epoch+1)%10==0:
        print('Val acc:',acc_hist[-1])
        print('Saving model:')
        torch.save(model.state_dict(), 'new_model_%s.pt'%epoch)
            # Model computations


# In[12]:


local_labels.device


# In[ ]:


torch.save(model.state_dict(), 'new_model_%s.pt'%epoch)


# In[64]:


#model = CNNClassifier().cuda()
#model.load_state_dict(torch.load('model_59.pt'))


# In[81]:


_, ax = plt.subplots(1,2,figsize = (16,6))
ax[0].plot(np.linspace(0,max_epoch,len(loss_hist)),loss_hist,label='test')
ax[0].plot(np.linspace(0,max_epoch,len(loss_val_hist)),loss_val_hist,label='val')
ax[0].legend()
ax[0].grid()

ax[1].plot(np.linspace(0,max_epoch,len(acc_hist)),acc_hist,label='val_acc')
ax[1].legend()
ax[1].grid()
plt.savefig('path_residual.pdf',dpi=100)
plt.show()


# In[69]:


model = CNNClassifier().cuda()
model.load_state_dict(torch.load('model_59.pt'))


# In[82]:


total_size = 0
acc  = 0
metric = MulticlassAccuracy( num_classes=9)
with torch.set_grad_enabled(False):
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device).float(), local_labels[:,0].to(device).long()
        pred_raw = model(local_batch)
        size = local_batch.shape[0]
        total_size += size 
        metric.update(pred_raw,local_labels)
        acc += metric.compute().item() *size
print('Accuracy on the train_set using Residual Net: ', acc/total_size)


# In[83]:


total_size = 0
acc  = 0
metric = MulticlassAccuracy( num_classes=9)
with torch.set_grad_enabled(False):
    for local_batch, local_labels in validation_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device).float(), local_labels[:,0].to(device).long()
        pred_raw = model(local_batch)
        size = local_batch.shape[0]
        total_size += size 
        metric.update(pred_raw,local_labels)
        acc += metric.compute().item() *size
print('Accuracy on the validation_set using Residual Net: ', acc/total_size)


# In[84]:


total_size = 0
acc  = 0
metric = MulticlassAccuracy( num_classes=9)
with torch.set_grad_enabled(False):
    for local_batch, local_labels in test_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device).float(), local_labels[:,0].to(device).long()
        pred_raw = model(local_batch)
        size = local_batch.shape[0]
        total_size += size 
        metric.update(pred_raw,local_labels)
        acc += metric.compute().item() *size
print('Accuracy on the test_set  using Residual Net: ', acc/total_size)


# In[ ]:





# In[ ]:




