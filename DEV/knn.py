import numpy as np
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import statistics 

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

# Method for reading in the "pickled" object images
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Preprocessing- convert to greyscale
def rgb2gray(im):
    col_size = len(im[:,0])
    im_out = np.empty([col_size,1024])
    
    for i in range(0,col_size):
        for j in range(0,1024):
            r = im[i,j] 
            g = im[i,j+1024]
            b = im[i,j+2048]
            im_out[i,j] = (0.299 * r + 0.587 * g + 0.114 * b)
    
    return im_out

# Read in the datasets 5 training batches and 1 test batch, each has 10,000 images
data_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
data_batch_2 = unpickle('cifar-10-batches-py/data_batch_2')
data_batch_3 = unpickle('cifar-10-batches-py/data_batch_3')
data_batch_4 = unpickle('cifar-10-batches-py/data_batch_4')
data_batch_5 = unpickle('cifar-10-batches-py/data_batch_5')
data_batch_6 = unpickle('cifar-10-batches-py/test_batch')

# Each data_batch is a dictionary with the following items
# b'batch_label --> specifies which batch it is
# b'labels --> array of 10,000 labels 0-9 correspoding to the correct classification
# b'data --> 10,000 x 3072 array of uint8 pixels, each rows is a 32x32 image with the first 1024 entries being the red,
#            the second 1024 entries being the green, and the last 1024 entries being the blue

db1_labels = data_batch_1[b'labels']
db1_data = data_batch_1[b'data']
db2_labels = data_batch_2[b'labels']
db2_data = data_batch_2[b'data']
db3_labels = data_batch_3[b'labels']
db3_data = data_batch_3[b'data']
db4_labels = data_batch_4[b'labels']
db4_data = data_batch_4[b'data']
db5_labels = data_batch_5[b'labels']
db5_data = data_batch_5[b'data']
db6_labels = data_batch_6[b'labels']
db6_data = data_batch_6[b'data']


db1_data = rgb2gray(db1_data)
db2_data = rgb2gray(db2_data)
db3_data = rgb2gray(db3_data)
db4_data = rgb2gray(db4_data)
db5_data = rgb2gray(db5_data)
db6_data = rgb2gray(db6_data)

A = np.vstack((db1_data, db2_data, db3_data, db4_data, db5_data)) #Training matrix
d = np.column_stack((np.array(db1_labels), np.array(db2_labels), np.array(db3_labels), np.array(db4_labels), np.array(db5_labels))).reshape(50000,1) #Known classifiers
T = db6_data
y = db6_labels

train_size = len(A[:,0])
test_size = len(T[:,0])

ks = [1,5,10,20,45,70,100]
training_errors = np.zeros(len(ks))
training_sqs = np.zeros(len(ks))

for idx in range(0,len(ks)):
    print(ks[idx])

    # Uses l2 norm
    knn = KNeighborsClassifier(n_neighbors=ks[idx],algorithm='ball_tree',p=2)
    
    # Uses l1 norm
    # knn = KNeighborsClassifier(n_neighbors=3,algorithm='kd_tree',p=1)
    
    knn.fit(A, d)

    error_count = 0
    labels = np.zeros(test_size)

    for i in range(0,test_size):
        test = T[i,:].reshape((1,-1))
        y_hat = knn.predict(test)
        labels[i] = y_hat
    #     y_hat = knn.predict(T[i,:])

        if y_hat != y[i]:
            error_count += 1

    error_rate = error_count / test_size
    training_errors[idx] = error_rate
    
    squared_error = np.linalg.norm(labels - y)**2
    training_sqs[idx] = squared_error

#     for i in range(0,20):
#         print(labels[i])

# Determine which k gave the lowest error rates
min_idx = 0
min_error = 50000

for i in range(0,len(ks)):
    if training_errors[i] < min_error:
        min_idx = i
        min_error = training_errors[i]  
        
error_rate = training_errors[min_idx]
mse = training_sqs[min_idx] / test_size
        
print("Optimal Lambda Chosen: " + str(ks[min_idx]))
print()

print("Error Rate: " + str(round(error_rate*100,3)) + ", Mean Sqaured Error: " + str(round(mse,3)))
print()

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
print()
print()