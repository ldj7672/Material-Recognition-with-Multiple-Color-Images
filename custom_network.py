import torch
import numpy as np
import cv2
import copy
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b,v,c = x.size()
        x_ = torch.mean(x,dim=1).view(b,1,c)
        x_ = self.excitation(x_)
        x = x*x_
        return x

class oneStream_multiView_Net(nn.Module):
    def __init__(self,numClass):
        super(oneStream_multiView_Net, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512,128)

        self.lstm_1 = nn.LSTM(input_size=128, hidden_size=128,num_layers=1,batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=128, hidden_size=75,num_layers=1,batch_first=True)
        self.fc1 = nn.Linear(75,numClass)

    def forward(self, x):
        b_size, numView, C, H, W = x.size()
        x = x.view(b_size*numView, C, H, W) 
        x = self.model(x)
        x = x.view(b_size, numView, -1) # reshape input for LSTM

        x,(_,_) = self.lstm_1(x)
        x,(_,_) = self.lstm_2(x)

        x = self.fc1(x[:,-1,:])
        return x

class twoStream_multiView_Net(nn.Module):
    def __init__(self,numClass):
        super(twoStream_multiView_Net, self).__init__()
        self.colorFeatNet = models.resnet18(pretrained=True)
        self.brightnessVarNet = models.resnet18(pretrained=True)

        self.colorFeatNet.fc = nn.Linear(512,128)
        self.brightnessVarNet.fc = nn.Linear(512,64)

        self.lstm = nn.LSTM(input_size=192, hidden_size=128,num_layers=1,batch_first=True)
        self.last_fc = nn.Linear(128,numClass)

        self.bn_color = nn.BatchNorm1d(128)
        self.bn_brightness = nn.BatchNorm1d(64)

        self.Attblock = AttentionBlock(in_channels=128+64)

    def forward(self, x):
        b_size, numView, C, H, W = x.size()
        #===========Patch Sorting===========#
        x_1 = copy.copy(x)
        m = torch.zeros((b_size,numView))
        c = torch.zeros((b_size,numView),dtype =torch.int8)

        for i in range(b_size):
            for j in range(numView):
                m[i,j] = torch.mean(x[i,j,:,:,:])
        for i in range(b_size):
            for j in range(numView):
                r=0
                for k in range(numView):
                    if m[i,j] < m[i,k] : r+=1
                c[i,j] = r
        for i in range(b_size):
            for j in range(numView):
                x[i,c[i,j]] = x_1[i,j]
        #==================================#

        x = x.reshape(b_size*numView, C, H, W) # reshape input for CNN: rnnBatch*25 = cnnBatch

        #=========Making reflectance Patch=========#
        x_r = copy.copy(x).reshape(b_size,numView,C,H,W) 
        x_avg = torch.zeros((b_size,C,H,W))
        for i in range(b_size):
            x_avg[i] = torch.mean(x_r[i],dim=0)
        x_avg = x_avg.reshape(b_size,C,H,W)

        for a in range(numView):    
            x_r[:,a] = torch.from_numpy(cv2.absdiff(np.array(x_avg[:].cpu()),np.array(x_r[:,a].cpu())))
        x_r = x_r.reshape(b_size*numView, C, H, W)
        #============================+++============#

        x = self.bn_color(self.colorFeatNet(x))
        x_r = self.bn_brightness(self.brightnessVarNet(x_r))
        x = x.view(b_size, numView, -1) 
        x_r = x_r.view(b_size, numView, -1) 
        x = torch.cat([x,x_r],dim=2)
        x = self.Attblock(x)

        del x_r
        x = x.view(b_size, numView, -1) # reshape input for LSTM: cnnBatch/25 = rnnBatch

        x,(_,_) = self.lstm(x)
        x = self.last_fc(x[:,-1,:])
        
        return x

if __name__ == '__main__':
    # model = oneStream_multiView_Net(numClass=30)
    model = twoStream_multiView_Net(numClass=30)
    model.eval()

    x = torch.rand((10,9,3,224,224)) # batch size, num of views, C, H, W
    print(model(x).shape)
