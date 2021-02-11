import csv
import numpy as np


Threshold_value_of_difference_of_cost_values=0.0000001

# This function read the training csv file
def Import_data():
    X=np.genfromtxt("train_X_lg_v2.csv",delimiter=',',dtype=np.float64,skip_header=1)
    Y=np.genfromtxt("train_Y_lg_v2.csv",delimiter=',',dtype=np.float64)
    return X,Y

def sigmoid_function(Z):
    s=1.0/(1.0+np.exp(-Z))
    return s

def compute_cost(X, Y, W, b):
    Z=np.dot(X,W.T)+b
    sigmoid_value=sigmoid_function(Z)
    sigmoid_value[sigmoid_value==1]=0.9999 #it is used to avoid nan value vy sum function as log(0) is not defined 
    sigmoid_value[sigmoid_value==0]=0.0001  #similarliy when a value is much larege or log(1) its result is zero.on that time we get nan
    sum=np.sum(np.multiply(Y,np.log(sigmoid_value))+np.multiply((1-Y),np.log(1-sigmoid_value)))
    return(((-1.0)*sum)/len(Y))

def compute_gradient_of_cost_function(X, Y, W, b):
    Z=np.dot(X,W.T)+b
    sigmoid_value=sigmoid_function(Z)
    db=np.sum(sigmoid_value-Y)/len(Y)
    dw=np.dot((sigmoid_value-Y).T,X)/len(Y)
    return dw,db


def Optimize_weights_using_gradient_descent(X,Y,W,b,learning_rate,class_value):
    i=1;
    prev_cost_value=0
    while True:
        dw,db=compute_gradient_of_cost_function(X,Y,W,b)
        W=W-(learning_rate*dw)
        b=b-(learning_rate*db)
        cost_value=compute_cost(X,Y,W,b)
        if((i%1000000)==0):
            print("current i value",i,"cost value--",cost_value)
        if abs(cost_value-prev_cost_value)<(Threshold_value_of_difference_of_cost_values):
            print("final no of iteration",i)
            list_of_no_iterations.append((i,class_value))
            break
        prev_cost_value=cost_value
        i+=1
    return W


def train_model(X,Y,learning_rate,class_value):
    Y=Y.reshape(X.shape[0],1)
    W=np.zeros((1,X.shape[1]))
    b=0
    W=Optimize_weights_using_gradient_descent(X,Y,W,b,learning_rate,class_value)
    return W

def get_train_data_for_class(train_X,train_Y,class_label):
    class_X=np.copy(train_X)
    class_Y=np.copy(train_Y)
    class_Y=np.where(class_Y==class_label,1,0)
    return class_X,class_Y


if __name__=="__main__":
    X,Y=Import_data()
    length_of_differnt_Y_value=len(np.unique(Y))
    learning_rate=[0.006,0.0035,0.003,0.006]
    list_of_no_iterations=[]
    for i in range(length_of_differnt_Y_value):
        class_X,class_Y=get_train_data_for_class(X,Y,i)
        weights=train_model(class_X,class_Y,learning_rate[i],i)
        print("this is iteration for given class binery logistic regression---",i)

    print(list_of_no_iterations)