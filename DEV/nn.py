import numpy as np
from datetime import datetime
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as fun
import torchvision.transforms as transforms
import torch.optim as optim
import tensorflow as tf

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

# Method for reading in the "pickled" object images
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def preprocess(x):
    min_val = np.matrix.min(x)
    max_val = np.matrix.max(x)
    x = (x - min_val) / (max_val - min_val)
    x = (x * 2) - 1
    return x

# Read in the datasets 5 training batches and 1 test batch, each has 10,000 images
# data_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
# data_batch_2 = unpickle('cifar-10-batches-py/data_batch_2')
# data_batch_3 = unpickle('cifar-10-batches-py/data_batch_3')
# data_batch_4 = unpickle('cifar-10-batches-py/data_batch_4')
# data_batch_5 = unpickle('cifar-10-batches-py/data_batch_5')
# data_batch_6 = unpickle('cifar-10-batches-py/test_batch')

# Each data_batch is a dictionary with the following items
# b'batch_label --> specifies which batch it is
# b'labels --> array of 10,000 labels 0-9 correspoding to the correct classification
# b'data --> 10,000 x 3072 array of uint8 pixels, each rows is a 32x32 image with the first 1024 entries being the red,
#            the second 1024 entries being the green, and the last 1024 entries being the blue

#Read in the batch data and perform pre-processing
# db1_labels = data_batch_1[b'labels']
# db1_data = data_batch_1[b'data'].reshape((len(data_batch_1[b'data']), 3, 32, 32))#.transpose(0, 2, 3, 1)
# db2_labels = data_batch_2[b'labels']
# db2_data = data_batch_2[b'data'].reshape((len(data_batch_2[b'data']), 3, 32, 32))#.transpose(0, 2, 3, 1)
# db3_labels = data_batch_3[b'labels']
# db3_data = data_batch_3[b'data'].reshape((len(data_batch_3[b'data']), 3, 32, 32))#.transpose(0, 2, 3, 1)
# db4_labels = data_batch_4[b'labels']
# db4_data = data_batch_4[b'data'].reshape((len(data_batch_4[b'data']), 3, 32, 32))#.transpose(0, 2, 3, 1)
# db5_labels = data_batch_5[b'labels']
# db5_data = data_batch_5[b'data'].reshape((len(data_batch_5[b'data']), 3, 32, 32))#.transpose(0, 2, 3, 1)
# db6_labels = data_batch_6[b'labels']
# db6_data = data_batch_6[b'data'].reshape((len(data_batch_6[b'data']), 3, 32, 32))#.transpose(0, 2, 3, 1)


# Each new image has the form num_chanels (RGB) x width x height = 3 x 32 x 32
# 10,000 images in each data segment so 10000 x 3 x 32 x 32

# print(len(db1_data[:,0,0,0])) # col size
# print(len(db1_data[0,:,0,0])) # row size
# print(len(db1_data[0,0,:,0])) # row size
# print(len(db1_data[0,0,0,:])) # row size

# print(len(db1_labels)) # col size

# for i in range(0,10000):
    # db1_data[i,:,:,:] = preprocess(np.matrix(db1_data[i,:,:,:]))
    # db2_data[i,:,:,:] = preprocess(db2_data[i,:,:,:])
    # db3_data[i,:,:,:] = preprocess(db3_data[i,:,:,:])
    # db4_data[i,:,:,:] = preprocess(db4_data[i,:,:,:])
    # db5_data[i,:,:,:] = preprocess(db5_data[i,:,:,:])
    # db6_data[i,:,:,:] = preprocess(db6_data[i,:,:,:])
    # for j in range(0,3):
    #     db1_data[i,j,:,:] = preprocess(np.matrix(db1_data[i,j,:,:]))
        # db2_data[i,j,:,:] = preprocess(db2_data[i,j,:,:])
        # db3_data[i,j,:,:] = preprocess(db3_data[i,j,:,:])
        # db4_data[i,j,:,:] = preprocess(db4_data[i,j,:,:])
        # db5_data[i,j,:,:] = preprocess(db5_data[i,j,:,:])
        # db6_data[i,j,:,:] = preprocess(db6_data[i,j,:,:])

# print(np.min(db1_data[500,1,:,:]))
# print(np.max(db1_data[500,1,:,:]))
    
    
# One Hot Encoding
# db1_labels = np.to_categorical(db1_labels)
# db2_labels = np.to_categorical(db2_labels)
# db3_labels = np.to_categorical(db3_labels)
# db4_labels = np.to_categorical(db4_labels)
# db5_labels = np.to_categorical(db5_labels)
# db6_labels = np.to_categorical(db6_labels)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(fun.relu(self.conv1(x)))
        x = self.pool(fun.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = fun.relu(self.fc1(x))
        x = fun.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
# print(net)

criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
# criterion = nn.L1Loss()


# lr = learning rate, momentum = helps avoid local minimum
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# tensor_db1 = torch.from_numpy(db1_data)
# tensor_db2 = torch.from_numpy(db2_data)
# tensor_db3 = torch.from_numpy(db3_data)
# tensor_db4 = torch.from_numpy(db4_data)
# tensor_db5 = torch.from_numpy(db5_data)
# tensor_db6 = torch.from_numpy(db6_data)
# print(trainloader.size())

# out = net(tensor_db1)
# input = torch.randn(1, 1, 32, 32)
# out = net(input)
# print(out)

# NN Training Loop
count = 0
total_loss = 0
for trial in range(0,2):
    for idx, curr in enumerate(trainloader,0):
        count += 1
        data, label = curr
        # label = torch.from_numpy(tf.keras.utils.to_categorical(label, num_classes=10))
        optimizer.zero_grad()

        out = net(data)

        if idx < 5:
            print(out)
            print(label)
        
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if idx % 10000 == 9999:
            print("Iteration "+str(trial+1)+": current loss = "+str(total_loss/count))

        # if idx % 2000 == 1999:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (trial + 1, idx + 1, total_loss / 2000))
        #     total_loss = 0.0

print(count)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
