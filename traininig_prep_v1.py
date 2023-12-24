## training_preparation
# ver1 
#   prepare mfcc file from wav files of IRMAS dataset


import os
import numpy as np
import generate_mfcc_v1
import functions_v1

def main():
    ## data info    
    orig_data_directory = '../IRMAS_dataset/IRMAS-TrainingData/' #original data directory
    # inst_list = os.listdir(orig_data_directory)
    inst_list = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    train_data_directory = 'traindata/'
    functions_v1.create_directory(train_data_directory)
    mfcc_folder = train_data_directory + 'mfccs_processed/' #mfcc path
    functions_v1.create_directory(mfcc_folder)
    n_band = 13; #Mel band
    
    
    #list to store labels
    labels = []
        
    for instID in inst_list:
        inst_path      = orig_data_directory + instID + '/'
        filename_list = os.listdir(inst_path) 
        filename_list_wav = [file for file in filename_list if file.endswith('.wav')] # Filter .wav files
        
        # List to store audio features
        mfccs = []
        labels = []
        
        # Extract audio features
        for file_name in filename_list_wav:
            file_path = inst_path + file_name
            print(f"Start conversion of '{file_path}'.")
            mfcc = generate_mfcc_v1.main(file_path, n_band)  #audio feature extraction method
            mfccs.append(mfcc)
            labels.append(instID)
            print(f"File '{file_path}' converted successfully.")
            
        # Save MFCC and Score data to a NumPy binary file
        mfcc_filename = mfcc_folder + 'mfccs_' + instID + '_n' + str(n_band) + '.npy'
        # Create a dictionary to store the arrays
        data_dict = {
            'mfccs': mfccs,
            'labels': labels
        }
        np.save(mfcc_filename, data_dict)
   

if __name__ == '__main__':
    main()