## generate_mfcc
# ver1 08/22/2023
#   generate lpc data from sound files’ path or array
#   mfcc band option
#   filepath or array option
#

import librosa
import time


def main(arg0, order, sample_rate = []):
    start = time.time()
    if sample_rate:
        audio = arg0
    else:
        filename = arg0
        # Load an audio file
        audio, sample_rate = librosa.load(filename, sr = None)

    # Extract MFCC features
    lpc = librosa.lpc(audio, order=order)
    
    process_time = time.time() - start
    print(process_time) 
    return lpc

if __name__ == '__main__':
    filename = 'path/to/audio_file.mp3'
    mfccs = main(filename)