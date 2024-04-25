
import pandas as pd
import numpy as np
import shutil, os
import csv
# from pydub import AudioSegment

# train=pd.read_csv('C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\all_data_updated.csv')
# # C:\Users\mateo\Desktop\ABBY GOES HERE\bees\all_data_updated.csv

# train.head()
# train['queen acceptance']=train['queen acceptance'].replace({0:'queen present or original queen',
#                                                      1:'queen not present',
#                                                      2 :'queen present and rejected', 
#                                                      3 :'queen present and newly accepted'})
# train['queen status']=train['queen status'].replace({0:'queen present or original queen',
#                                                      1:'queen not present',
#                                                      2 :'queen present and rejected', 
#                                                      3 :'queen present and newly accepted'})


# Id=[]

# for dirname, _, filenames in os.walk('C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\sound_files\\sound_files'):
#     for filename in filenames:
#         Id.append(os.path.join(dirname, filename))


def modify_file_names(input_file, output_file):
    with open(input_file, 'r') as csv_in, open(output_file, 'w', newline='') as csv_out:
        reader = csv.DictReader(csv_in)
        fieldnames = ['file name', 'queen status']  # Only keep these columns
        
        # Add a new field for modified file name
        fieldnames.append('modified_file_name')
        
        writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            for i in range(1, 6):  # Repeat each row 5 times
                modified_row = {'file name': row['file name'], 'queen status': row['queen status']}
                modified_row['modified_file_name'] = f"{row['file name'][:-4]}__segment{i}.wav"  # Modify file name
                writer.writerow(modified_row)

# Example usage:
output_file = 'C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\train.csv'
input_file = 'C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\all_data_updated.csv'
modify_file_names(input_file, output_file)

train = pd.read_csv("C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\all_data_updated.csv")

train.head()
train['queen acceptance']=train['queen acceptance'].replace({0:'queen present or original queen',
                                                     1:'queen not present',
                                                     2 :'queen present and rejected', 
                                                     3 :'queen present and newly accepted'})
train['queen status']=train['queen status'].replace({0:'queen present or original queen',
                                                     1:'queen not present',
                                                     2 :'queen present and rejected', 
                                                     3 :'queen present and newly accepted'})
        