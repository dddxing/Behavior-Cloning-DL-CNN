from base import RobotPolicy
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
torch.manual_seed(0)


class MyCNN(nn.Module):

     def __init__(self):
          super(MyCNN, self).__init__()
          self.conv1 = nn.Conv2d(3,8,kernel_size=5,stride=1, padding=2)
          self.conv2 = nn.Conv2d(8,16,kernel_size=5,stride=1, padding=2)

          self.pool = nn.MaxPool2d(2,2)

          self.bn1d = nn.BatchNorm1d(64)
          self.bn2d1 = nn.BatchNorm2d(8)
          self.bn2d2 = nn.BatchNorm2d(16)

          self.fc1 = nn.Linear(16*4*4, 64)
          self.fc2 = nn.Linear(64, 4)
          
     def forward(self,x):
          x = self.pool(F.relu(self.bn2d1(self.conv1(x))))
          x = self.pool(x)
          x = self.pool(F.relu(self.bn2d2(self.conv2(x))))
          x = self.pool(x)

          x = x.reshape(x.size(0), -1)

          x = F.relu(self.bn1d(self.fc1(x)))
          x = self.fc2(x)
          return x

     def predict(self, features):       
          self.eval()        
          features = torch.from_numpy(features).float()  
          pred = self.forward(features.permute(0,3,2,1)).detach().numpy()
          
          return np.argmax(pred)


class MyDataset(Dataset):    
     def __init__(self, labels, features):        
          super(MyDataset, self).__init__()        
          self.labels = labels        
          self.features = features    
     def __len__(self):        
          return self.features.shape[0]    
     def __getitem__(self, idx):          
          feature = self.features[idx]        
          label = self.labels[idx]        
          return {'feature': feature, 'label': label}


class TrainCNN():

     def __init__(self):
          self.network = MyCNN()
          self.learning_rate = 0.01
          self.optimizer = torch.optim.Adam(self.network.parameters(),lr=self.learning_rate)
          self.criterion = nn.CrossEntropyLoss()
          self.num_epochs = 100
          self.batchsize = 500
          self.shuffle = True
          
     def train(self,labels,features):
          self.network.train()
          dataset = MyDataset(labels,features)
          loader = DataLoader(dataset,shuffle = self.shuffle, batch_size = self.batchsize)
          for epoch in range(self.num_epochs):
               print(epoch)
               self.train_epoch(loader)

     def train_epoch(self,loader):
          total_loss = 0.0
          for i,data in enumerate(loader,0):
               features = data['feature'].float()
            
               labels = data['label'].long()
               self.optimizer.zero_grad()
               
               predictions = self.network(features.permute(0,3,2,1))     
               loss = self.criterion(predictions, labels)            
               loss.backward()            
               total_loss += loss.item()
               self.optimizer.step()    
          print('loss', total_loss/i)
          
     def get_action(self,features):
          return self.network.predict(features)


class RGBBCRobot1(RobotPolicy):

    trainer = TrainCNN()
    def train(self, data):
        print("Using solution for RGBBCRobot1")
        features = data["obs"]
        labels = data["actions"]
        RGBBCRobot1.trainer.train(labels,features)

    def get_action(self, obs):
        
        obs = obs[None]
        pred_action = RGBBCRobot1.trainer.get_action(obs)
        return pred_action

