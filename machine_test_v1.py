## machine_test
# ver1 
#   learn acoustic feature (mfcc/lpc from wav) files and their labels, train a SVM model


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import seaborn as sn
import pandas as pd
import csv
import scipy

import functions_v1

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay

def loadmodel(model_data_directory, model_type, kernel, feature, n_band, num_epoch, cv=[]):
    
    featurename = feature + str(n_band)
    if model_type != 'svm':
        filename = 'model_' + model_type + "_" + featurename + "_epch" + str(num_epoch) #NN
    elif kernel == 'linear':
        filename = 'model_' + model_type + "_" + kernel + "_" + featurename #linear SVM
    elif cv == []:
        filename = 'model_' + model_type + "_" + kernel + '_' + featurename #RBF SVM
    else:
        filename = 'model_' + model_type + "_" + kernel + "best_cv" + str(cv) + '_' + featurename #RBF cross-varidation
        
    model_path = model_data_directory + filename + ".pickle"
    with open(model_path, mode='rb') as fp:
        model = pickle.load(fp)
        
    return model, filename

def mfcc2tensol(mfccs):
    
    num_chunk = np.size(mfccs, 0)
    num_mfcc_bin = np.size(mfccs, 1)
    num_mfcc_frame = np.size(mfccs, 2)
    
    mfcc_tensor = tf.reshape(mfccs, [num_chunk, num_mfcc_bin*num_mfcc_frame])
    
    return mfcc_tensor


def preprocess(feature_folder, feature, n_band):    

    filename_list      = os.listdir(feature_folder)
    filename_list_feature = [file for file in filename_list if file.startswith(feature)] # Filter audio feature files
    filename_list_feature = [file for file in filename_list_feature if file.endswith(str(n_band)+'.npy')] # Filter band

    data  = []
    labels = []

    for filename in filename_list_feature:
        file_path   = feature_folder + filename
        loaded_data = np.load(file_path, allow_pickle=True).item()
        data_temp   = np.array(loaded_data[feature+'s'])
        labels_temp = np.array(loaded_data['labels'])
        print(f"Converted '{filename}'.")
        
        data.append(data_temp)
        labels.append(labels_temp)

    data_concat   = np.concatenate(data,   axis=0) 
    labels_concat = np.concatenate(labels, axis=0) 
    
    if feature == 'mfcc':
        # Convert lists to numpy arrays
        X = mfcc2tensol(data_concat)
        y = labels_concat
    else:
        X = data_concat
        worN = 512 
        Xenv=np.zeros([X.shape[0], worN])
        for i in range(X.shape[0]):
            Xenv[i,:] = np.abs(scipy.signal.freqz(1.0, X[i,:], worN=worN)[1])
        X = Xenv
        ndind = ~np.isnan(X).any(axis=1) #non-nan index
        X = X[ndind,:]
        y = labels_concat[ndind]
        
    scaler = StandardScaler()
    X = scaler.fit_transform(X) # Scale features
    mfcc_length = np.size(X,1) 

    # Data Preprocessing
    label_uniq = np.unique(y)
    num_classes = np.size(label_uniq)
    label_encoder = LabelEncoder() #initialize
    y = label_encoder.fit_transform(y)
    
    return X, y, num_classes, label_uniq


def model_test(model, X_test, y_test, model_type, kernel, filename_fig, label_uniq, prediction, cv=[]):
    

    recall = recall_score(y_test, prediction,average=None)
    print("Recall: ", recall)
    precision = precision_score(y_test, prediction,average=None)
    print("Precision: ", precision)
    f1 = f1_score(y_test, prediction, average=None)
    print("F1-Score: ", f1)
    accuracy_norm = accuracy_score(y_test, prediction, normalize=True)
    accuracy      = accuracy_score(y_test, prediction,normalize=False)
    print("Accuracy: %.2f  ," % accuracy_norm, accuracy)
    
    cfsn_mtrx = confusion_matrix(y_test, prediction)

    df_cm = pd.DataFrame(cfsn_mtrx, index=label_uniq, columns=label_uniq)
    plt.figure(figsize = (100,70))
    sn.set(font_scale=18.0)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 80},fmt='g')# font size

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(filename_fig) 
    plt.show()

    return recall, precision, f1, accuracy_norm, accuracy, cfsn_mtrx

# def main():
    ## data info    
model_data_directory = 'model/'     #model data directory
test_data_directory  = 'testdata/'  #test  data directory
feature = 'mfcc'
n_band = 13; #Mel band / lpc order 
feature_folder = test_data_directory + feature + 's_processed/';
num_epoch = 200
model_type = 'svm'
kernel = 'rbf'
cv = [] #cross-varidation
prediction_flg = 1 #do prediction or not
model_directory = 'model/'
functions_v1.create_directory(model_directory)
figure_directory = 'figure/'
functions_v1.create_directory(figure_directory)
results_directory = 'result/'
functions_v1.create_directory(results_directory)

#load model
print('-- Loading model --')
[model, filename] = loadmodel(model_data_directory, model_type, kernel, feature, n_band, num_epoch, cv)

print('-- Begin preparation --')
[X_test, y_test, num_classes, label_uniq] = preprocess(feature_folder, feature, n_band) 
print('Preparation has been done') 

#predict label
filename_pred = results_directory + 'prediction_' + filename + '.csv'
if prediction_flg:
    print('  -- Label prediction --')
    prediction = model.predict(X_test)
    print('  -- Done prediction --')
else:
    # Initialize lists to store data from CSV
    predictions = []
    
    # Read the CSV file
    with open(filename_pred, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            predictions.append(row)
    prediction = np.squeeze(np.array(predictions, dtype=float))

print('-- Begin testing --')
filename_fig = figure_directory + 'cfsnmtrx_' + filename + '.png'
[recall, precision, f1, accuracy_norm, accuracy, cfsn_mtrx] = model_test(model, X_test, y_test, model_type, kernel, filename_fig, label_uniq, prediction, cv)

filname_result = results_directory + 'result_' + filename + '.txt'
with open(filname_result, 'w') as f:
    print(type(f))
    f.write('recall'        + '\n' + str(recall)        + '\n')
    f.write('precision'     + '\n' + str(precision)     + '\n')
    f.write('F1_Score'      + '\n' + str(f1)            + '\n')
    f.write('accuracy_norm' + '\n' + str(accuracy_norm) + '\n')
    f.write('accuracy'      + '\n' + str(accuracy)      + '\n')
    f.write('cfsn_mtrx'     + '\n' + str(cfsn_mtrx)     + '\n')
    f.write('labels'        + '\n' + str(label_uniq)    + '\n')
       
# Write the prediction to the CSV file
with open(filename_pred, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(prediction)


# Adjust the size of accuracy to match the length of other arrays
accuracy_norm_filled = np.full(len(label_uniq), accuracy_norm)  # Fill accuracy_norm with a single value
accuracy_filled      = np.full(len(label_uniq), accuracy)       # Fill accuracy with a single value
filname_result_csv = results_directory + 'result_' + filename + '.csv'

# Create a dictionary with arrays and labels
data = {'recall': recall, 'precision': precision, 'f1_score': f1, 'accuracy_norm': accuracy_norm_filled, 'accuracy': accuracy_filled}

# Create a DataFrame
df = pd.DataFrame(data, index = label_uniq)

# Transpose the DataFrame
df_transposed = df.T

# Write the transposed DataFrame to a CSV file
df_transposed.to_csv(filname_result_csv, index_label='label')

       
    # encoding='utf-8-sig'
    # with open(file_path, 'rb') as file:
    #     content = file.read()
    #     decoded_content = content.decode(encoding)
    #     lines = decoded_content.splitlines()
        
    # csv_reader = csv.reader(lines)

# if __name__ == '__main__':
#     main()