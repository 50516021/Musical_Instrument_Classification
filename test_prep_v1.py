## test_preparation
# ver1 
#   prepare mfcc file from wav files of IRMAS dataset (test)
#   slice data


import os
import numpy as np
import re
import librosa
import math

import generate_mfcc_v1
import functions_v1


# def main():
    ## data info    
orig_data_directory = '../IRMAS_dataset/IRMAS-TestingData-Part' #original data directory
orig_data_directory_testlist = []
for i in np.arange(3):
    dirname_parent = orig_data_directory + str(i+1) + '/' 
    dirname_data   = [
        f for f in os.listdir(dirname_parent) if os.path.isdir(os.path.join(dirname_parent, f))
    ]
    orig_data_directory_testlist.append(dirname_parent + dirname_data[0] + '/') #test directory list

inst_list = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
test_data_directory = 'testdata/'
functions_v1.create_directory(test_data_directory)
mfcc_folder = test_data_directory + 'mfccs_processed/' #mfcc path
functions_v1.create_directory(mfcc_folder)
n_band = 13; #Mel band


#list to store labels
labels = []
musicname_list = []
    
for orig_testdata_path in orig_data_directory_testlist: #part1,2,3
    filename_list = os.listdir(orig_testdata_path) 
    filename_list_wav = sorted([file for file in filename_list if file.endswith('.wav')]) # Filter .wav files
    filename_list_txt = sorted([file for file in filename_list if file.endswith('.txt')]) # Filter .txt files

    #since each music file has difefrent length, generate mfcc files for each music
    # Extract file names before the numbers
    musicname_list_all = [re.match(r'(.+?)(-\d+)?\.wav', name).group(1) for name in filename_list_wav]
    musicname_list     = np.unique(musicname_list_all)

    for i, musicname in enumerate(musicname_list):
        # List to store audio features
        mfccs = []
        labels = []
        wavfile_name_list = [name for name, extracted in zip(filename_list_wav, musicname_list_all) if extracted == musicname]
        txtfile_name_list = [name for name, extracted in zip(filename_list_txt, musicname_list_all) if extracted == musicname]
        
        for wavfile, txtfile in zip(wavfile_name_list, txtfile_name_list):
            wavfile_path = orig_testdata_path + wavfile
            txtfile_path = orig_testdata_path + txtfile
            
            print(f"Start conversion of '{wavfile_path}'.")
            audio, fs = librosa.load(wavfile_path, sr = None)  #load audio file 
            wavdur = np.shape(audio)[0]/fs #music duration (5-20 sec)
            numslice = math.floor(wavdur-2) #number of slice to adjust length for train data
            
            with open(txtfile_path, 'r') as file:
                # Read lines and strip whitespace characters '\n'
                label_list = [line.strip() for line in file.readlines()]
                for j, instID in enumerate(label_list):
                    for k in range(numslice):
                        audio_sliced = audio[k*fs:(k+3)*fs-1]
                        mfcc = generate_mfcc_v1.main(audio_sliced, n_band, fs)  #audio feature extraction method
                        labels.append(instID)
                        mfccs.append(mfcc)
                    N = j+1
            print(f"File '{wavfile_path}' converted successfully for {N} variations ({wavdur} sec {numslice} slices).")
            
        # Save MFCC and Score data to a NumPy binary file for each music 
        mfcc_filename = mfcc_folder + 'mfccs_' + musicname + '_n' + str(n_band) + '.npy'
        # Create a dictionary to store the arrays
        data_dict = {
            'mfccs': mfccs,
            'labels': labels
        }
        np.save(mfcc_filename, data_dict)
        
        
# if __name__ == '__main__':
#     main()