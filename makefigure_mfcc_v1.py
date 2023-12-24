## makefigure
# ver1 
#   

import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy
from sklearn.preprocessing import StandardScaler, Normalizer

import generate_mfcc_v1
import functions_v1



filename = '[pia][cla]1305__3'
orig_data_directory = '../IRMAS_dataset/IRMAS-TrainingData/pia/' #original data directory
figure_directory = 'figure/'
functions_v1.create_directory(figure_directory)
test_data_directory  = 'testdata/'  #test  data directory
feature = 'mfcc'
order_list = [13, 20, 40]; #Mel band / lpc order 
feature_folder = test_data_directory + feature + 's_processed/';
file_path   = feature_folder + filename

# Generating three random waveforms with the same length
length = 1000  # Length of the waveforms
x = np.linspace(0, 10, length)  # X-axis values

wavfile_path = orig_data_directory + filename + '.wav'
audio, fs = librosa.load(wavfile_path, sr = None)  #load audio file 

mfccs = []
for order in order_list:
    plt.figure(figsize=(7, 4))
    mfcc = generate_mfcc_v1.main(audio, order, fs)  #audio feature extraction method
    mfccs.append(mfcc)
    librosa.display.specshow(mfcc, sr=fs, x_axis='time')
    plt.colorbar()
    
    plt.xlabel('time[s]')
    plt.ylabel('frames')
    plt.title('mfcc' + str(order))


    filename_fig = figure_directory + 'figure_' + feature + str(order) + filename + '.png'
    
    plt.savefig(filename_fig) 
    plt.show()

