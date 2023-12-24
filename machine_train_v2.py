## machine_train
# ver1 
#   learn acoustic feature (mfcc from wav) files and their labels, train a Sequential model
# ver2
#   use SVM, cross-varidation for RBF
#   lpc option

import os
import numpy as np
import tensorflow as tf
import functions_v1
import pickle
import scipy

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def datasplit(test_size_ratio, dataset):

    N = np.size(X,0) #eintire data size
    # Determining the sizes for training and testing sets
    test_size = int(test_size_ratio * N)
    train_size = N - test_size

    # Splitting the dataset into training and testing sets
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    
    return train_dataset, test_dataset
    

def mfcc2tensol(mfccs):
    
    num_chunk = np.size(mfccs, 0)
    num_mfcc_bin = np.size(mfccs, 1)
    num_mfcc_frame = np.size(mfccs, 2)
    
    mfcc_tensor = tf.reshape(mfccs, [num_chunk, num_mfcc_bin*num_mfcc_frame])
    
    return mfcc_tensor


def preprocess(train_data_directory, feature, n_band):    

    filename_list      = os.listdir(train_data_directory)
    filename_list_feature = [file for file in filename_list if file.startswith(feature)] # Filter audio feature files
    filename_list_feature = [file for file in filename_list_feature if file.endswith(str(n_band)+'.npy')] # Filter band

    data  = []
    labels = []

    for filename in filename_list_feature:
        file_path = train_data_directory + filename
        loaded_data = np.load(file_path, allow_pickle=True).item()
        data_temp   = np.array(loaded_data[feature+'s'])
        labels_temp = np.array(loaded_data['labels'])
        
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
    # print(mfccs.shape)

    # Data Preprocessing
    num_classes = np.size(np.unique(y))
    label_encoder = LabelEncoder() #initialize
    y = label_encoder.fit_transform(y)
    print('num_classes:', num_classes)
    
    return X, y, num_classes, mfcc_length, labels


def training(input_feature_length, num_classes, X_train, y_train, kernel, cv=[]):

    # Initialize the SVM model
    model = SVC(kernel=kernel, decision_function_shape='ovr', verbose=1)  #multi-class classification
    
    if (kernel == 'rbf') & (cv != []):
        parameters = {'C': np.logspace(-1, 2, 30), 'gamma':[0.1, 1, 10, 100]}
        clf = GridSearchCV(model, parameters, cv=cv, scoring='precision_micro', verbose=1) #cross-varidation
        model = clf

    # Train the SVM model
    print(' - Fitting Model')
    model.fit(X_train, y_train)
    
    score = model.score(X_train, y_train)
    print("Test score: {0:.2f} %".format(100 * score))

    return model

# def main():
    ## data info    
data_directory = 'traindata/' #data directory
feature = 'lpc'
n_band = 12; #Mel band / lpc order 
feature_folder = data_directory + feature + 's_processed/';
num_epoch = 200
model_type = 'svm'
kernel = 'linear'
cv = [] #cross-varidation
model_directory = 'model/'
functions_v1.create_directory(model_directory)
figure_directory = 'figure/'
functions_v1.create_directory(figure_directory)

print('-- Begin preparation --')
[X, y, num_classes, mfcc_length, labels] = preprocess(feature_folder, feature, n_band)   

print('Preparation has been done')

print('-- Begin training --')
# [model, history] = training(mfcc_length, num_classes, X_train, y_train)
model = training(mfcc_length, num_classes, X, y, kernel, cv)
print('Training has been done')

featurename = feature + str(n_band)
if model_type != 'svm':
    filename = 'model_' + model_type + "_" + featurename + "_epch" + str(num_epoch) #NN
elif kernel == 'linear':
    filename = 'model_' + model_type + "_" + kernel + "_" + featurename #linear SVM
elif cv == []:
    filename = 'model_' + model_type + "_" + kernel + '_' + featurename #RBF SVM
else:
    filename = 'model_' + model_type + "_" + kernel + "best_cv" + str(cv) + '_' + featurename #RBF cross-varidation
        
filename_model = model_directory + filename + ".pickle"
with open(filename_model, mode='wb') as f:
    pickle.dump(model,f,protocol=2)

print('-- Model has been saved')

# print('-- Begin testing --')
# predicted_scores = model_test(model, filename)
    

# if __name__ == '__main__':
#     main()