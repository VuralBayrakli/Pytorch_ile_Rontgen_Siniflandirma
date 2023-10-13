#!/usr/bin/env python
# coding: utf-8

# ### Modül ve Kütüphanelerin Import Edilmesi

# In[11]:


import torch
from torch import nn


import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor


import matplotlib.pyplot as plt
import torchvision.transforms as transforms


# ### Veri Yüklenmesi ve Dönüşümleri

# In[17]:


from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
data_dir = 'data/'  # Veri klasörünün yolu
batch_size = 128  # Mini-batch boyutu

# Veri dönüşümleri ve etiketleme işlemi
transform = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),
])

dataset = ImageFolder(root=data_dir, transform=transform)


# ###  Veri Yükleyici Oluşturma

# In[18]:


dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ### Bir Mini-Batch Çekme

# In[19]:


train_batch, label_batch = next(iter(dataloader))




import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1), 
            nn.Sigmoid()
        )
        
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        class_output = self.classification_head(features)
        return class_output

model = CustomModel()
model


# ### Doğruluk Fonksiyonunun İmport Edilmesi
# ### Kayıp Fonksiyonunun Tanımlanması
# ### Optimizasyon Stratejisinin Tanımlanması

# In[23]:


from helper_functions import accuracy_fn
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)


# ### Cihaz Seçimi ve Modelin Cihaza Taşınması

# In[24]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        
        X, y = X.to(device), y.to(device)
        y = y.to(torch.float)
        
        y_pred_logits = model(X)
        y_pred = torch.round(torch.sigmoid(y_pred_logits)).squeeze().to(device)
        
        loss = loss_fn(y_pred_logits.squeeze(), y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred) 

        
        optimizer.zero_grad()

        
        loss.backward()

        
        optimizer.step()

    
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() 
    
    with torch.inference_mode(): 
        for X, y in data_loader:
            
            X, y = X.to(device), y.to(device)
            y = y.to(torch.float)
            
            test_pred_logits = model(X)
            test_pred = torch.round(torch.sigmoid(test_pred_logits)).squeeze().to(device)
            
            test_loss += loss_fn(test_pred_logits.squeeze(), y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred 
            )
        
       
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


# ### Eğitim Döngüsü

# In[41]:


torch.manual_seed(2)
from tqdm import tqdm

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=new_dataloader, 
        model=model, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )


# ###  Eğitimden Sonra Mini-Batch Üzerinde Modelin Değerlendirilmesi


torch.save(model, 'custom_model2.pth')



