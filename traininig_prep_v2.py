## training_preparation
# ver1 
#   prepare lpc file from wav files of IRMAS dataset


import os
import numpy as np
import generate_lpc_v1
import functions_v1

def main():
    ## data info    
    orig_data_directory = '../IRMAS_dataset/IRMAS-TrainingData/' #original data directory
    # inst_list = os.listdir(orig_data_directory)
    inst_list = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    train_data_directory = 'traindata/'
    functions_v1.create_directory(train_data_directory)
    lpc_folder = train_data_directory + 'lpcs_processed/' #lpc path
    functions_v1.create_directory(lpc_folder)
    lpc_order = 40; #lpc order
    
    
    #list to store labels
    labels = []
        
    for instID in inst_list:
        inst_path      = orig_data_directory + instID + '/'
        filename_list = os.listdir(inst_path) 
        filename_list_wav = [file for file in filename_list if file.endswith('.wav')] # Filter .wav files
        
        # List to store audio features
        lpcs = []
        labels = []
        
        # Extract audio features
        for file_name in filename_list_wav:
            file_path = inst_path + file_name
            print(f"Start conversion of '{file_path}'.")
            lpc = generate_lpc_v1.main(file_path, lpc_order)  #audio feature extraction method
            lpcs.append(lpc)
            labels.append(instID)
            print(f"File '{file_path}' converted successfully.")
            
        # Save MFCC and Score data to a NumPy binary file
        lpc_filename = lpc_folder + 'lpcs_' + instID + '_n' + str(lpc_order) + '.npy'
        # Create a dictionary to store the arrays
        data_dict = {
            'lpcs': lpcs,
            'labels': labels
        }
        np.save(lpc_filename, data_dict)
   

if __name__ == '__main__':
    main()