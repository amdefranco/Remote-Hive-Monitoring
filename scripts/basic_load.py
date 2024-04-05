
import pandas as pd
import numpy as np
import shutil, os
from pydub import AudioSegment


train=pd.read_csv('C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\all_data_updated.csv')
# C:\Users\mateo\Desktop\ABBY GOES HERE\bees\all_data_updated.csv

train.head()
train['queen acceptance']=train['queen acceptance'].replace({0:'queen present or original queen',
                                                     1:'queen not present',
                                                     2 :'queen present and rejected', 
                                                     3 :'queen present and newly accepted'})
train['queen status']=train['queen status'].replace({0:'queen present or original queen',
                                                     1:'queen not present',
                                                     2 :'queen present and rejected', 
                                                     3 :'queen present and newly accepted'})


Id=[]

for dirname, _, filenames in os.walk('C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\sound_files\\sound_files'):
    for filename in filenames:
        # print(filename)
        Id.append(os.path.join(dirname, filename))



train['file name']=train['file name'].str[:-4]+'__segment'+train['target'].astype('str')+'.wav'

train=train[['file name','queen status']]

train.to_csv('./train.csv',index=False)