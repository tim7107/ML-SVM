import csv
import sys
from libsvm.svmutil import *
from libsvm.svm import *

import numpy as np



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

#----initial setting----#
kernel_types={'linear':'-t 0','polynomial':'-t 1','radial basis function':'-t 2'}
accuracy=[]
for name,num in kernel_types.items():
    model=svm_train(Y_train,X_train,'-q '+num)
    label,acc,vals=svm_predict(Y_test,X_test,model,'-q')
    accuracy.append(acc[0])

iter_ = 0
for name,num in kernel_types.items():
    print('{} kernel accuracy: {:.2f}%'.format(name,accuracy[iter_]))
    iter_+=1







