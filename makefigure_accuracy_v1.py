## makefigure accuracy
# ver1 
#   

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import functions_v1

figure_directory = 'figure/'
functions_v1.create_directory(figure_directory)
test_data_directory  = 'testdata/'  #test  data directory
results_directory = 'result/'
functions_v1.create_directory(results_directory)
order_list = [20, 40]; #Mel band / lpc order 
nband_list = [13, 20, 40]
# features = ['mfcc','lpc']
features = ['mfcc', 'lpc']
model_type  = 'svm'
kernel_list = ['linear', 'rbf']
score_names = ['recall', 'precision', 'F1 score', 'accuracy_norm', 'accuracy']

filenames = []
dataname_list = []

for ac_feature in features:
    for kernel in kernel_list:
        if ac_feature == 'mfcc':
            for nband in nband_list:
                featurename = ac_feature + str(nband)
                filenames.append('result_model_' + model_type + "_" + kernel + "_" + featurename + '.csv')
                dataname_list.append(featurename + model_type + '_' + kernel)
        if ac_feature == 'lpc':
            for order in order_list:
                featurename = ac_feature + str(order)
                filenames.append('result_model_' + model_type + "_" + kernel + "_" + featurename + '.csv')
                dataname_list.append(featurename + model_type + '_' + kernel)
   
file_path_list = []
for file_name in filenames:
    file_path_list.append(results_directory + file_name)

for i, scorename in enumerate(score_names):

    plt.figure(figsize=(80, 60))

    # Iterate through each CSV file
    for file_path in file_path_list:
        # Read the CSV file
        df = pd.read_csv(file_path)
    
        # Filter for 'recall' values
        dfnum = df.to_numpy()
    
        plt.plot(dfnum[i,1:], linewidth = 5)

        
    labels=[]
    for label in  np.array(df.iloc[0].index[1:], dtype=str):
        labels.append(label)
    plt.title(scorename + ' plot')
    plt.xlabel('Label')
    plt.xticks(range(11), labels)
    plt.ylabel(scorename)
    plt.legend(dataname_list, fontsize=100)
    plt.grid(True)
    filename_fig = figure_directory + 'figure_' + scorename + '.png'
    
    plt.savefig(filename_fig) 
    plt.show()