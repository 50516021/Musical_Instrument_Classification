## audio_separation
# ver1 07/27/2023
#   load mp4 file and generate mp3 file by following timecodes in a csv file
# ver2 08/18/2023
#   load timegap list
# ver3 09/03/2023
#   timecode division

import ffmpeg
import os
import csv

def create_directory(folder_name):
    # Check if the folder already exists
    if not os.path.exists(folder_name):
        # If it doesn't exist, create the folder
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")

def timecode_to_seconds(timecode):
    #timecode conversion from teh format hh.mm.ss to second
    # print(timecode)
    h, m, s = map(int, timecode.split(':'))
    return h * 3600 + m * 60 + s

def find_data_by_id(id_to_find, csv_file_path):
    with open(csv_file_path, 'r', newline='', encoding='shift-jis') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        next(csv_reader)  # Skip header row
        
        for row in csv_reader:
            dataname, ID, data = row
            if ID == id_to_find:
                return data
        
        return None  # Return None if ID not found

def extract_audio_segment(input_file, output_file, start_time, end_time):
    ffmpeg.input(input_file, ss=start_time, to=end_time).output(output_file).run()




   
