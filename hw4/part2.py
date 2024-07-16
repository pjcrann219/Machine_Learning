## import libraries
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# print(torch.__version__)

args={}
kwargs={}
args['batch_size']=32
args['test_batch_size']=32
args['epochs']=1  #The number of Epochs is the number of times you go through the full dataset.
args['lr']=0.01 #Learning rate is how fast it will decend.
args['momentum']=0.5 #SGD momentum (default: 0.5) Momentum is a moving average of our gradients (helps to keep direction).

args['seed']=1 #random seed
args['log_interval']= 250
args['cuda']=True #if the computer has a GPU, type True, otherwise, False

## transformations
transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

## download and load training dataset
trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, **kwargs)

## download and load testing dataset
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args['test_batch_size'], shuffle=True, **kwargs)

import matplotlib.pyplot as plt
import numpy as np

## functions to show an image
def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

## get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# ## show images
# imshow(torchvision.utils.make_grid(images))

# for images, labels in train_loader:
#     print("Image batch dimensions:", images.shape)
#     print("Image label dimensions:", labels.shape)
#     break

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.name = 'Original Net'
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x=x.view(-1,784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class Net_1(nn.Module):

    def __init__(self):
        super(Net_1, self).__init__()
        self.name = '1 CNN Layer'
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(14*14*16, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(-1,14*14*16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

class Net_2_0(nn.Module):

    def __init__(self):
        super(Net_2_0, self).__init__()
        self.name = '2 CNN Layers'
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*32, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1,7*7*32)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

class Net_2_1(nn.Module):

    def __init__(self):
        super(Net_2_1, self).__init__()
        self.name = '2 CNN Layers, double kernels'
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1,7*7*64)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x
    
class Net_2_2(nn.Module):

    def __init__(self):
        super(Net_2_2, self).__init__()
        self.name = '2 CNN Layers, size 7 kernel'
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*32, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1,7*7*32)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

# ## test the model with 1 batch
# model = Net_2_2()
# #print(model)
# for images, labels in train_loader:
#     print("batch size:", args['batch_size'])
#     out = model(images)
#     print(out.shape)
#     break

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args['cuda']:
            data, target = data.cuda(), target.cuda()
        #Variables in Pytorch are differenciable.
        data, target = Variable(data), Variable(target)
        #This will zero out the gradients for this batch.
        optimizer.zero_grad()
        output = model(data)
        # Calculate the loss The negative log likelihood loss. It is useful to train a classification problem with C classes.
        loss = F.nll_loss(output, target)
        #dloss/dx for every Variable
        loss.backward()
        #to do a one-step update on our parameter.
        optimizer.step()
        #Print out the loss periodically.
        # if batch_idx % args['log_interval'] == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.data.item()))
            

def test():
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
      for data, target in test_loader:
          if args['cuda']:
              data, target = data.cuda(), target.cuda()
          data, target = Variable(data), Variable(target)
          output = model(data)
          test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
          pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
          correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('{:}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        model.name, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

epoch = 1
print(f"\nNumber of epochs: {epoch}")
model = Net()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
for i in range(epoch):
    train(i)
test()
del(model)

model = Net_1()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
for i in range(epoch):
    train(i)
test()
del(model)

model = Net_2_0()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
for i in range(epoch):
    train(i)
test()

model = Net_2_1()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
for i in range(epoch):
    train(i)
test()

model = Net_2_2()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
for i in range(epoch):
    train(i)
test()

epoch = 10
print(f"Number of epochs: {epoch}")
model = Net()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
for i in range(epoch):
    train(i)
test()

model = Net_1()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
for i in range(epoch):
    train(i)
test()

model = Net_2_0()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
for i in range(epoch):
    train(i)
test()

model = Net_2_1()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
for i in range(epoch):
    train(i)
test()

model = Net_2_2()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
for i in range(epoch):
    train(i)
test()