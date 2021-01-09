import numpy as np
from libsvm.svmutil import *
from scipy.spatial.distance import cdist
import csv

def precomputed_kernel(X, X_, gamma):
    kernel_linear=X @ X_.T
    kernel_RBF=np.exp(-gamma*cdist(X, X_, 'sqeuclidean'))
    kernel=kernel_linear+kernel_RBF
    kernel=np.hstack((np.arange(1,len(X)+1).reshape(-1,1),kernel))
    return kernel

#----load data----#
"""
    (5000, 784)
    (5000, 1)
    (2500, 784)
    (2500, 1)
"""
read_X_train=open('C:/Users/tim/Desktop/ML/HW05/X_train.csv')
read_Y_train=open('C:/Users/tim/Desktop/ML/HW05/Y_train.csv')
read_X_test=open('C:/Users/tim/Desktop/ML/HW05/X_test.csv')
read_Y_test=open('C:/Users/tim/Desktop/ML/HW05/Y_test.csv')

csv_reader_lines_X_train = csv.reader(read_X_train)
csv_reader_lines_Y_train = csv.reader(read_Y_train)
csv_reader_lines_X_test = csv.reader(read_X_test)
csv_reader_lines_Y_test = csv.reader(read_Y_test)

X_train=[]
Y_train=[]
X_test=[]
Y_test=[]

for line in csv_reader_lines_X_train:
    X_train.append(line)
for line in csv_reader_lines_Y_train:
    Y_train.append(line)
for line in csv_reader_lines_X_test:
    X_test.append(line)
for line in csv_reader_lines_Y_test:
    Y_test.append(line)
        
X_train = np.asarray(X_train,dtype='float')
Y_train = np.asarray(Y_train,dtype='int').flatten()
X_test = np.asarray(X_test,dtype='float')
Y_test = np.asarray(Y_test,dtype='int').flatten()

kernel_train=precomputed_kernel(X_train, X_train, 2**-4)
prob=svm_problem(Y_train,kernel_train,isKernel=True)
param=svm_parameter('-q -t 4')
model=svm_train(prob,param)

kernel_test=precomputed_kernel(X_test, X_train, 2**-4)
label,acc,vals=svm_predict(Y_test,kernel_test,model,'-q')
print('linear kernel + RBF kernel accuracy: {:.2f}%'.format(acc[0]))



