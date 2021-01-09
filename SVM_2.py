import matplotlib.pyplot as plt
import csv
from libsvm.svmutil import *
from libsvm.svm import *
import numpy as np

"""
    param :
        -v n : n-fold cross validation mode
        -q   : quiet mode (no outputs)
        -t 2 :  radial basis function
        -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
        -g gamma : set gamma in kernel function (default 1/num_features)
"""
def grid_search(log_c,log_g,X_train,y_train,X_test,y_test):
    confusion_matrix=np.zeros((len(log_c),len(log_g)))
    for i in range(len(log_c)):
        for j in range(len(log_g)):
            param='-q -t 2 -v 3 -c {} -g {}'.format(2**log_c[i],2**log_g[j])
            acc=svm_train(y_train,X_train,param)
            confusion_matrix[i,j]=acc
    return confusion_matrix

def plot_confusion_matrix(confusion_matrix,log_c,log_g):
    fig, ax = plt.subplots()
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    ax.set_xticklabels([''] + log_g)
    ax.xaxis.set_label_position('top')
    ax.set_yticklabels([''] + log_c)
    for i in range(len(log_c)):
        for j in range(len(log_g)):
            ax.text(i, j, '{:.2f}'.format(confusion_matrix[j, i]), va='center', ha='center')
    ax.set_xlabel('lg(G)')
    ax.set_ylabel('lg(C)')
    plt.savefig("C:/Users/tim/Desktop/碩二/ML/HW05/confusion_matrix.png")
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

#----init setting----#
log_c=[-4,-3,-2,-1,0,1,2,3,4]
log_g=[-4,-3,-2,-1,0,1,2,3,4]

#---- C-SVC ----#
confusion_matrix=grid_search(log_c,log_g,X_train,Y_train,X_test,Y_test)
plot_confusion_matrix(confusion_matrix,log_c,log_g)