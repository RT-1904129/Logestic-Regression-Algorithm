import numpy as np
import csv
import sys

from validate import validate



def import_data_and_weights(test_X_file_path, weights_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    weights = np.genfromtxt(weights_file_path, delimiter=',', dtype=np.float64)
    return test_X, weights

def sigmoid_function(Z):
    s=1.0/(1.0+np.exp(-Z))
    return s

def predict_target_values(test_X, weights):
    b_list=weights[:,0]
    W_list=weights[:,1:]
    Predicted_list=[]
    for i in range(len(weights)):
        Z=np.dot(test_X,W_list[i].T)+b_list[i]
        sigmoid_value=sigmoid_function(Z)
        Predicted_list.append(sigmoid_value)
        
    Predicted_array=np.array(Predicted_list)
    max_value=np.max(Predicted_array,axis=0)
    predicted_value=[]
    for i in range(len(test_X)):
        if(Predicted_array[0][i]==max_value[i]):
            predicted_value.append(0)
        elif(Predicted_array[1][i]==max_value[i]):
            predicted_value.append(1)
        elif(Predicted_array[2][i]==max_value[i]):
            predicted_value.append(2)
        elif(Predicted_array[3][i]==max_value[i]):
            predicted_value.append(3)
    
    predicted_value=np.array(predicted_value)
    predicted_value=predicted_value.reshape(test_X.shape[0],1)
    return predicted_value
        


def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()


def predict(test_X_file_path):
    test_X, weights= import_data_and_weights(test_X_file_path, "WEIGHTS_FILE.csv")
    pred_Y = predict_target_values(test_X, weights)
    write_to_csv_file(pred_Y, "predicted_test_Y_lg.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    #validate(test_X_file_path, actual_test_Y_file_path="train_Y_lg_v2.csv") 