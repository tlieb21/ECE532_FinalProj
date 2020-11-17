###################################################################################################
#
# ECE 532 Final Project Fall 2020
# Timothy Lieb tlieb@wisc.edu
#

import numpy as np

# Method for reading in the "pickled" object images
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# Read in the datasets 5 training batches and 1 test batch, each has 10,000 images
data_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
data_batch_2 = unpickle('cifar-10-batches-py/data_batch_2')
data_batch_3 = unpickle('cifar-10-batches-py/data_batch_3')
data_batch_4 = unpickle('cifar-10-batches-py/data_batch_4')
data_batch_5 = unpickle('cifar-10-batches-py/data_batch_5')
test_batch = unpickle('cifar-10-batches-py/test_batch')

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
db6_labels = test_batch[b'labels']
db6_data = test_batch[b'data']
# tb_labels = test_batch[b'labels']
# tb_data = test_batch[b'data']

# print(db1_data[0,:]) # --> first image
# print(db1_data[:,0]) # -->first column


###################################################################################################
#
# Ridge Regression
#
def ridge_regression(A1, A2, A3, A4, A5, d1, d2, d3, d4, d5, T1, T2, y1, y2, lambdas):
    print("Running Ridge Regression")
    A = np.vstack((A1, A2, A3, A4, A5)) #Training matrix
    d = np.vstack((d1, d2, d3, d4, d5)) #Known classifiers

    print(len(A[:,0]))
    print(len(A[0,:]))

    num_iterations = len(lambdas)
    training_errors = np.zeros(num_iterations)

    # Perform the training over all the different lambdas
    for lam in range(0, num_iterations):
        print("Using lambda = " + str(lambdas[lam]))
        w = np.linalg.inv(A.T @ A + lambdas[lam] * np.indentity(len(A))) @ A.T @ d

        # Find the predictions for the first test set
        t_hat = T1 @ w
        error_count = 0

        # Record the number of errors
        for i in range(0, len(t_hat)):
            if t_hat[i] != y1[i]:
                error_count += 1
        
        training_errors[lam] = error_count
    
    # Determine which lambda gave the lowest error rates
    min_idx = 0
    min_error = 50000

    for i in range(0,num_iterations):
        if training_errors[i] < min_error:
            min_idx = i
            min_error = training_errors[i]
    

    # Use the selected lambda with the rest of the training data to get w
    lam = lamdas[min_idx]

    w = np.linalg.inv(A.T @ A + lam * np.indentity(len(A))) @ A.T @ d

    # Find the predictions for the second test set
    y_hat = T2 @ w
    error_count = 0

    # Record the number of errors
    for i in range(0, len(y_hat)):
        if y_hat[i] != y2[i]:
            error_count += 1

    # Calculate the errors and return them
    error_rate = error_count / len(y2)
    squared_error = np.linalg.norm(y_hat - y2)**2

    return ([error_rate, squared_error])



###################################################################################################
#
# K-Mean Clustering? --> Is this going to work?
#




###################################################################################################
#
# Neural Networks
#


# Logarithmic spaced lambdas
# TODO: how to determine the span of these?
# lambdas = np.logspace(-6,np.log(20))
lambdas = [0,0.25,0.5,0.75,1,2,4]



# T1 = db1_data[0:5000,:]
# T2 = db1_data[5000:10000,:]
# y1 = db1_labels[0:5000]
# y1 = db1_labels[5000:10000]
[error_rate1, squared_error1] = ridge_regression(db2_data, db3_data, db4_data, db5_data, db6_data, 
                                                    db2_labels, db3_labels, db4_labels, db5_labels, db6_labels, 
                                                    db1_data[0:5000,:], db1_data[5000:10000,:], db1_labels[0:5000], 
                                                    db1_labels[5000:10000], lambdas)
print("Ridge Regression Iteration 1")
print("Error Rate: " + str(round(error_rate1*100,3)) + ", Sqaured Error: " + str(round(squared_error1,3)))
print()

[error_rate2, squared_error2] = ridge_regression(db1_data, db3_data, db4_data, db5_data, db6_data, 
                                                    db1_labels, db3_labels, db4_labels, db5_labels, db6_labels, 
                                                    db2_data[0:5000,:], db2_data[5000:10000,:], db2_labels[0:5000], 
                                                    db2_labels[5000:10000], lambdas)
print("Ridge Regression Iteration 2")
print("Error Rate: " + str(round(error_rate2*100,3)) + ", Sqaured Error: " + str(round(squared_error2,3)))
print()

[error_rate3, squared_error3] = ridge_regression(db1_data, db2_data, db4_data, db5_data, db6_data, 
                                                    db1_labels, db2_labels, db4_labels, db5_labels, db6_labels, 
                                                    db3_data[0:5000,:], db3_data[5000:10000,:], db3_labels[0:5000], 
                                                    db3_labels[5000:10000], lambdas)
print("Ridge Regression Iteration 3")
print("Error Rate: " + str(round(error_rate3*100,3)) + ", Sqaured Error: " + str(round(squared_error3,3)))
print()

[error_rate4, squared_error4] = ridge_regression(db1_data, db2_data, db3_data, db5_data, db6_data, 
                                                    db1_labels, db2_labels, db3_labels, db5_labels, db6_labels, 
                                                    db4_data[0:5000,:], db4_data[5000:10000,:], db4_labels[0:5000], 
                                                    db4_labels[5000:10000], lambdas)
print("Ridge Regression Iteration 4")
print("Error Rate: " + str(round(error_rate4*100,3)) + ", Sqaured Error: " + str(round(squared_error4,3)))
print()

[error_rate5, squared_error5] = ridge_regression(db1_data, db2_data, db3_data, db4_data, db6_data, 
                                                    db1_labels, db2_labels, db3_labels, db4_labels, db6_labels, 
                                                    db5_data[0:5000,:], db5_data[5000:10000,:], db5_labels[0:5000], 
                                                    db5_labels[5000:10000], lambdas)
print("Ridge Regression Iteration 5")
print("Error Rate: " + str(round(error_rate5*100,3)) + ", Sqaured Error: " + str(round(squared_error5,3)))
print()

[error_rate6, squared_error6] = ridge_regression(db1_data, db2_data, db3_data, db4_data, db5_data, 
                                                    db1_labels, db2_labels, db3_labels, db4_labels, db5_labels, 
                                                    db6_data[0:5000,:], db6_data[5000:10000,:], db6_labels[0:5000], 
                                                    db6_labels[5000:10000], lambdas)
print("Ridge Regression Iteration 6")
print("Error Rate: " + str(round(error_rate6*100,3)) + ", Sqaured Error: " + str(round(squared_error6,3)))
print()

# Aset = [db1_data, db2_data, db3_data, db4_data, db5_data, db6_data]
# yset = [db1_labels, db2_labels, db3_labels, db4_labels, db5_labels, db6_labels]

# # Looping over the samples
# for i in range(0,6):
#     A1 = None
#     A2 = None
#     A3 = None
#     A4 = None
#     A5 = None
            
#     for j in range(0,6):
#         if i == j:

#         elif i != j and l != j and T1 is None:
#             A1 = Xset[l]
#             d1 = yset[l]
#         elif l != i and l != j and T2 is None:
#             A2 = Xset[l]
#             d2 = yset[l]
#         elif l != i and l != j and T3 is None:
#             A3 = Xset[l]
#             d3 = yset[l]
#         elif l != i and l != j and T4 is None:
#             A4 = Xset[l]
#             d4 = yset[l]
#         elif l != i and l != j and T5 is None:
#             a5 = Xset[l]
#             d5 = yset[l]

