
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


# In[9]:


device


# In[4]:


#check the training test validation set
print(y_train.mean(),y_val.mean(),y_test.mean())
X_train.shape,X_val.shape,X_test.shape


# In[12]:


pos_weight = 0.2
print('using class weights:',pos_weight)


# In[13]:


plt.imshow(dataset[keys[0]][2],cmap='gray', vmin=0, vmax=255)


# # Build Models

# In[14]:


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier,self).__init__()
        self.cov = nn.Conv2d(in_channels=1,out_channels=32, kernel_size = (3,3))
        self.maxpool = nn.MaxPool2d(kernel_size = [2,2])
        self.linear1 = nn.Linear(5408,64)
        self.linear2 = nn.Linear(64,1)
        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self,x):
        h = F.relu(self.cov(x))
        h =  self.maxpool(h)
        h = h.flatten(1,3)
        h = F.relu(self.linear1(h))
        y = self.linear2(h)#F.sigmoid(
        return y
    
def accuracy(raw,labels):
    raw,labels = raw.cpu(),labels.cpu()
    pred = (F.sigmoid(raw).numpy()>0.5).astype(float)
    return (pred == labels.numpy()).mean()


# In[15]:


torch.manual_seed(0) 
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor((float(pos_weight),)).to(device))
model = CNNClassifier().cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=2e-4)
max_epoch = 50
# Loop over epochs
loss_hist = []
loss_val_hist = []
acc_hist = []
print('Training CNN on Pneumonia')
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


# In[32]:


_, ax = plt.subplots(1,2,figsize = (16,6))
ax[0].plot(np.linspace(0,max_epoch,len(loss_hist)),loss_hist,label='test')
ax[0].plot(np.linspace(0,max_epoch,len(loss_val_hist)),loss_val_hist,label='val')
ax[0].legend()
ax[0].grid()

ax[1].plot(np.linspace(0,max_epoch,len(acc_hist)),acc_hist,label='val_acc')
ax[1].legend()
ax[1].grid()

plt.savefig('pneumonia.pdf',dpi=100)
plt.show()


# In[27]:


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
print('Accuracy on the train_set: ', acc/total_size)


# In[28]:


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
print('Accuracy on the validation_set: ', acc/total_size)


# In[29]:


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
print('Accuracy on the test_set: ', acc/total_size)
