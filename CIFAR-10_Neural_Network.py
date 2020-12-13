import numpy as np
from datetime import datetime
import torch
import torchvision
import torch.nn as n_net
import torch.nn.functional as func
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

# Normalize the data
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


# tensor_db1 = torch.from_numpy(db1_data)
# tensor_db2 = torch.from_numpy(db2_data)
# tensor_db3 = torch.from_numpy(db3_data)
# tensor_db4 = torch.from_numpy(db4_data)
# tensor_db5 = torch.from_numpy(db5_data)
# tensor_db6 = torch.from_numpy(db6_data)

# PyTorch built in method for reading th CIFAR-10 dataset, performs pre-processing as well
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=0)

nn_width = 10

class Net(n_net.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = n_net.Conv2d(3, nn_width, 5)
        self.pool = n_net.MaxPool2d(2, 2)
        self.conv2 = n_net.Conv2d(nn_width, 16, 5)
        
        self.fc1 = n_net.Linear(16 * 5 * 5, 120)
        self.fc2 = n_net.Linear(120, 84)
        self.fc3 = n_net.Linear(84, 10)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# criterion = n_net.MSELoss()
# criterion = n_net.CrossEntropyLoss()
criterion = n_net.L1Loss()


# lr = learning rate, momentum = helps avoid local minimum
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# NN Training Loop
num_iterations = 1
count = 0
total_loss = 0

for trial in range(0,num_iterations):
    for idx, curr in enumerate(trainloader,0):
        count += 1
        data, label = curr
        label = torch.from_numpy(tf.keras.utils.to_categorical(label, num_classes=10))
        optimizer.zero_grad()

        out = net(data)

        # if idx < 5:
        #     print(out)
        #     print(label)
        
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

# correct = 0
test_size = 10000
error_count = 0
squared_error = 0
training_errors = np.zeros(test_size)
training_sqs = np.zeros(test_size)

with torch.no_grad():
    for curr in testloader:
        data, labels = curr
        out = net(data)
        b, predicted = torch.max(out.data, 1)
        # predicted = torch.max(out.data, 1)

        error_count += (predicted != labels).sum().item()
        squared_error += np.linalg.norm(predicted - labels)**2
        

error_rate = error_count / test_size
mse = squared_error / test_size

        # correct += (predicted == labels).sum().item()

print("Neural Network trained with " + str(num_iterations) + " iterations")
print("Error Rate: " + str(round(error_rate*100,3)) + ", Mean Sqaured Error: " + str(round(mse,3)))
print()

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)