## generate_mfcc
# ver1 08/22/2023
#   generate mfcc data from sound files’ path or array
#   mfcc band option
#   filepath or array option

import librosa
import time


def main(arg0, n_mfcc, sample_rate = []):
    start = time.time()
    if sample_rate:
        audio = arg0
    else:
        filename = arg0
        # Load an audio file
        audio, sample_rate = librosa.load(filename, sr = None)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc = n_mfcc)
    input_feature_length = mfccs.shape[0]  #the number of MFCC coefficients
    
    process_time = time.time() - start
    print(process_time) 
    return mfccs

if __name__ == '__main__':
    filename = 'path/to/audio_file.mp3'
    mfccs = main(filename)