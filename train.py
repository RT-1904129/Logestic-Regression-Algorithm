import csv
import numpy as np
import matplotlib.pyplot as plt

#no_iterations=10000

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


def Optimize_weights_using_gradient_descent(X,Y,W,b,no_iterations,learning_rate):
    for i in range(1,no_iterations):
        dw,db=compute_gradient_of_cost_function(X,Y,W,b)
        W=W-(learning_rate*dw)
        b=b-(learning_rate*db)
        cost_value=compute_cost(X,Y,W,b)
        if (i%100000)==0:
            print("i value ",i)
#         list_cost_index.append(i)
#         list_cost_value.append(cost_value)
    return (W,b)


def train_model(X,Y,learning_rate,no_iterations):
    Y=Y.reshape(X.shape[0],1)
    W=np.zeros((1,X.shape[1]))
    b=0
    W,b=Optimize_weights_using_gradient_descent(X,Y,W,b,no_iterations,learning_rate)
    return (W,b)

# Here we use csv module to write W values in newly created csv file
def save_model(weights,weights_file_name):
    with open(weights_file_name,'a',newline='') as weight_file:# we use newline='' because i want to remove blank row during writing 
        file_writer=csv.writer(weight_file,delimiter=",") #csv.writer(weight_file,delimiter=in what way you want sepration)
        file_writer.writerows(weights) # for writing arrays row in rows of created csv file
        weight_file.close()
        
def get_train_data_for_class(train_X,train_Y,class_label):
    class_X=np.copy(train_X)
    class_Y=np.copy(train_Y)
    class_Y=np.where(class_Y==class_label,1,0)
    return class_X,class_Y
    
if __name__=="__main__":
    X,Y=Import_data()
    length_of_differnt_Y_value=len(np.unique(Y))
    learning_rate=[0.006,0.0035,0.003,0.006]
    list_of_no_iterations=[(309937, 0), (36466, 1), (41686, 2), (375072, 3)]

    for i in range(length_of_differnt_Y_value):
    #     list_cost_value=[]
    #     list_cost_index=[]
        class_X,class_Y=get_train_data_for_class(X,Y,i)
        weights,b_value=train_model(class_X,class_Y,learning_rate[i],list_of_no_iterations[i][0])
        weights=np.insert(weights,0,b_value,axis=1)
        save_model(weights,"WEIGHTS_FILE.csv")


    #these commented code used to get desired Learning rate value for all the 4 class value
    #     plt.title("Graph respect to given feature value--{}".format(i))
    #     plt.plot(list_cost_index,list_cost_value)
    #     plt.show()
