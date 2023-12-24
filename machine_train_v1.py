## machine_train
# ver1 
#   learn acoustic feature (mfcc from wav) files and their labels, train a Sequential model

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.callbacks import History
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import functions_v1
import pickle

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
    
    # Convert lists to numpy arrays
    X = mfcc2tensol(data_concat)
    y = labels_concat

    # mfcc_length = mfccs.shape[0] 
    mfcc_length = np.size(X,1)    
    # print(mfccs.shape)

    # Data Preprocessing
    num_classes = np.size(np.unique(y))
    label_encoder = LabelEncoder() #initialize
    y_int = label_encoder.fit_transform(y)
    y = to_categorical(y_int, num_classes, dtype='int64')
    print('num_classes:', num_classes)
    
    return X, y, num_classes, mfcc_length, labels


def training(input_feature_length, num_classes, X_train, y_train, num_epoch):
    # Build the Neural Network Model
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_feature_length,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile and Train the Model
    print(' - Compiling Model')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(' - Fitting Model')
    history = model.fit(X_train, y_train, batch_size=32, epochs=num_epoch, validation_split=0.2)

    return model, history


def model_test(model, history, filename):
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:orange'
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(history.history['loss'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:green'
    ax2.set_ylabel('accuracy', color=color) 
    ax2.plot(history.history['accuracy'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    color = 'tab:purple'
    ax1.set_xlabel('iteration',color=color)
    ax1.tick_params(axis='x', labelcolor=color)
    
    fig.tight_layout()# otherwise the right y-label is slightly clipped
    filename_fig = 'figure/model_spec_' + filename + '.jpg'
    plt.savefig(filename_fig ,dpi=100)
    plt.show()

    return predicted_scores

# def main():
    ## data info    
data_directory = 'traindata/' #data directory
feature = 'mfcc'
n_band = 20; #Mel band
feature_folder = data_directory + feature + 's_processed/';
num_epoch = 200
model_type = 'sequential'
model_directory = 'model/'
functions_v1.create_directory(model_directory)
figure_directory = 'figure/'
functions_v1.create_directory(figure_directory)

print('-- Begin preparation --')
[X, y, num_classes, mfcc_length, labels] = preprocess(feature_folder, feature, n_band)   

print('Preparation has been done')

print('-- Begin training --')
# [model, history] = training(mfcc_length, num_classes, X_train, y_train)
[model, history] = training(mfcc_length, num_classes, X, y, num_epoch)
print('Training has been done')

filename = model_type + "_" + feature + "_epch" + str(num_epoch)
filename_model = "model/model_" + filename + ".pickle"
with open(filename_model, mode='wb') as f:
    pickle.dump(model,f,protocol=2)

print('-- Model has been saved')

print('-- Begin testing --')
predicted_scores = model_test(model, history, filename)
    

# if __name__ == '__main__':
#     main()