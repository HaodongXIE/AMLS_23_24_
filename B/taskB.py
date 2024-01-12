
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# # Load dataset

# In[2]:

# # PathMNIST

# In[2]:

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

# In[3]:


#check the training test validation set
X_train.shape,X_val.shape,X_test.shape


# In[4]:


# compute balance
from collections import Counter
cnt = Counter(y_train[:,0])
print(cnt)
total = sum(cnt.values())
cnt = {key: total/val for key,val in cnt.items()}
cnt = dict(sorted(cnt.items()))


# In[13]:


pos_weight = torch.Tensor(list(cnt.values())).cuda()



# In[6]:

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier,self).__init__()
        self.cov1 = nn.Conv2d(in_channels=3,out_channels=32, kernel_size = (3,3))
        self.cov2 = nn.Conv2d(in_channels=32,out_channels=64, kernel_size = (3,3))
        self.maxpool = nn.MaxPool2d(kernel_size = [2,2])
        self.linear1 = nn.Linear(9216,128)
        self.linear2 = nn.Linear(128,9)
        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self,x):
        h = F.leaky_relu(self.cov1(x))
        h = F.leaky_relu(self.cov2(h))
        h =  self.maxpool(h)
        h = h.flatten(1,3)
        h = F.leaky_relu(self.linear1(h))
        y = self.linear2(h)#F.sigmoid(
        return F.softmax(y)


# In[ ]:


torch.manual_seed(0) 
loss_fn = torch.nn.CrossEntropyLoss(weight =pos_weight )
model = CNNClassifier().cuda()
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



torch.save(model.state_dict(), 'new_model_%s.pt'%epoch)



# In[ ]:


_, ax = plt.subplots(1,2,figsize = (16,6))
ax[0].plot(np.linspace(0,max_epoch,len(loss_hist)),loss_hist,label='test')
ax[0].plot(np.linspace(0,max_epoch,len(loss_val_hist)),loss_val_hist,label='val')
ax[0].legend()
ax[0].grid()

ax[1].plot(np.linspace(0,max_epoch,len(acc_hist)),acc_hist,label='val_acc')
ax[1].legend()
ax[1].grid()
plt.savefig('path.pdf',dpi=100)
plt.show()


# In[13]:


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
print('Accuracy on the train_set: ', acc/total_size)


# In[ ]:



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
print('Accuracy on the val_set: ', acc/total_size)


# In[15]:


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
print('Accuracy on the test_set: ', acc/total_size)