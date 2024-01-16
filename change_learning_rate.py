#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/6/2 22:22
# @Author  : LiQinfeng
# @Software: PyCharm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms,datasets
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import xlwt


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
trainset = datasets.CIFAR10(root='\data',train=True,download=True,transform=transform)
testset = datasets.CIFAR10(root='\data',train=False,download=True,transform=transform)
BATCH_SIZE=32
# print(trainset)
train_loader= DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=True)        # 下载数据集
test_loader= DataLoader(testset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=True)
print(train_loader)
classes = ('plane','car','bird','cat','deer',                                                       # 标明类别
           'dog','frog','horse','ship','truck')

imgs , labs= next(iter(train_loader))                                                           # 生成迭代器\

class Net(nn.Module):                                                                           # 建立模型
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5,padding=2)                                                          #卷积层 1 输入特征3，输出特征6，卷积核5，下同
        self.pool = nn.MaxPool2d(2,2)
        self.avpool = nn.AvgPool2d(2,2)# 池化层，采用最大池化结构，模板为2*2，下同
        self.conv2 = nn.Conv2d(6, 12, 5,padding=2)
        self.conv3 = nn.Conv2d(12, 16, 5,padding=2)
        self.fc1 = nn.Linear(16*4*4,120)#,nn.Softmax()                                          # 线性层
        self.fc2 = nn.Linear(120,84)#,nn.Softmax()
        self.fc3 = nn.Linear(84, 10)                                                            # 输出为10类别
        # nn.Softmax(dim=1)
    def forward (self,x):
        x = F.relu(self.pool((self.conv1(x))))
        #print(x.shape)# 采用relu为激活函数
        x = self.avpool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = self.avpool(F.relu(self.conv3(x)))
        #print(x.shape)
        x = x.view(-1,16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net().to(device)
criterion = nn.CrossEntropyLoss()            # 采用交叉熵损失函数
lr = 0.1

conv1_params = list(map(id, net.conv1.parameters()))
conv2_params = list(map(id, net.conv2.parameters()))
conv3_params = list(map(id, net.conv3.parameters()))
base_params = filter(lambda p: id(p) not in conv1_params + conv2_params + conv3_params, net.parameters())
params = [{'params': base_params},
          {'params': net.conv1.parameters(), 'lr': lr * 0.001},
          {'params': net.conv2.parameters(), 'lr': lr * 0.1},
          {'params': net.conv3.parameters(), 'lr': lr * 0.1}]
optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

EPOCHS = 100                                                 # 迭代次数为200次
print(train_loader)
loss_dex = []
acc_dex = []
for epoch in range(EPOCHS):
    train_loss=0.0
    for i,(datas,labels) in enumerate(train_loader):
        datas,labels =datas.to(device),labels.to(device)
        # 梯度置零
        optimizer.zero_grad()
        # 训练
        outputs = net(datas)
        # 计算损失值
        loss = criterion(outputs,labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累计损失
        train_loss += loss.item()
    los = train_loss / len(train_loader.dataset)
    loss_dex.append(los.__float__())
    print("EPOCH : %d , Batch : %5d ,Loss : %.3f"% (epoch+1, i+1 ,los))
    PATH = './model/cifar_net_change_lr_'+str(epoch)+'.pth'
    torch.save(net.state_dict(), PATH)
for i in range(100):
    print(i)
    PATH = './model/cifar_net_change_lr_' + str(i) + '.pth'
    model = Net().to(device)  # 加载模型
    model.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (datas, labels) in enumerate(test_loader):
            outputs = model(datas.to(device))  # 输出 类型大小是 【32,10】
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)  # 累计数据量
            for t, (label) in enumerate(labels.to(device)):
                if predicted[t] == label:
                    correct += 1
        eacc =  correct / total
        print("在10000张测试照片上的准确率是: {:.3f}%".format(correct /  total * 100))
        acc_dex.append(eacc)

    class_correct = list(0. for i in range(10))
    class_recall = list(0. for i in range(10))
    total = list(0. for i in range(10))
    with torch.no_grad():
        for (images, labels) in test_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, dim=1)
            c = (predicted == labels.to(device)).squeeze()
            d = (predicted == predicted).squeeze()
            d1 = labels
            # print(d1[0].item(),d[0].item())
            if labels.shape[0] == 32:
                for i in range(BATCH_SIZE):
                    label = labels[i]
                    predict = predicted[i]
                    class_correct[label] += c[i].item()
                    class_recall[predict] += d[i].item()
                    total[label] += 1
    # print(labels)
    for i in range(10):
        print("正确率 : %5s : %2d %%" % (classes[i], 100 * class_correct[i] / total[i]))
        print("召回率 : %5s : %2d %%" % (classes[i], 100 * class_correct[i] / class_recall[i]))

workbook = xlwt.Workbook(encoding='utf-8')
sheet = workbook.add_sheet('结果')
title = ['迭代轮次', '准确率','训练集误差']
i = 0
for header in title:
    sheet.write(i, 0, header)
    i += 1
for col in range(100):
    sheet.write(0 , col +1, str(col))
for col in range(100):
    sheet.write(1 , col +1, acc_dex[col])
for col in range(100):
    sheet.write(2 , col +1, loss_dex[col])
workbook.save('结果1.xls')
print("导出成功！")
