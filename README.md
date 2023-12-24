# Musical Instrumnet Classification 
Musical Instrumnet Classification machine learning for IRMAS (a dataset for instrument recognition in musical audio signals)

# DEMO

# Features

# Requirement
* TensorFlow
* numpy
* librosa
* time
* os
* csv
* difflib

# Installation
Use the files to keep their original configuration.
Place the data set folder in a particular place, then set an appropriate folder path in the scripts.
## script breakdown
### main scripts
*traininig_prep_v1.py
*traininig_prep_v2.py
*machine_train_v2.py
*test_prep_v1.py
*test_prep_v2.py
*machine_test_v1.py

### ananlysis scripts
*makefigure_accuracy_v1.py
*makefigure_mfcc_v1.py

### functions
*functions_v1.py
*generate_mfcc_v1.py
*generate_lpc_v1.py

# Usage
The process should be done in this order:
*training preparation (test_prep_v1/v2.py)
*machine training (machine_train_v2.py)
*test preparation (train_prep_v1/v2.py)
*machine test (machine_test_v1.py)

After all execution, you can also make figures by using
*makefigure_accuracy_v1.py
*makefigure_mfcc_v1.py

# Note
## preparation options
Preparation v1 is for MFCC and v2 is for LPC.
You can set options as n_band for MFCC and order for LPC.

## train/test options
You can set options as follows:
*feature: 'mfcc' or 'lpc'
*n_band for MFCC and order for LPC
*kernel: 'linear' or 'rbf'
*cv: k of k cross-validation (only for RBF)
Once you finish testing, also possible to set 'prediction_flg' as 0 which skips the prediction step and makes figures.

# Author
## *Akira Takeuchi*
### Rochester Institute of Technology
### at2163@rit.edu

# License
Musical Instrument Classification is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
