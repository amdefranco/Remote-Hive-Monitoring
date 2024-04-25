
import pandas as pd
import numpy as np
import shutil, os
from pydub import AudioSegment

train=pd.read_csv('C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\all_data_updated.csv')

import ipdb; ipdb.set_trace()

# Id=[]
# for dirname, _, filenames in os.walk('C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\sound_files\\sound_files'):
#     for filename in filenames:
#         Id.append(os.path.join(dirname, filename))
# import ipdb; ipdb.set_trace()
# file=pd.DataFrame()
# file=file.assign(filename=Id)
# file.columns=['file name']
# file['file name']=file['file name']
# file['file name']=file['file name'].str.replace('C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\sound_files\\sound_files\\',"")

# import ipdb; ipdb.set_trace()

# train=pd.merge(train, file, on='file name', how='inner')
base = 'C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\sound_files\\sound_files'


# train['file name']=base+train['file name']
train.columns=['file name','label']


test=train[int(len(train)*.8):]

test=test.sample(n=len(test))
test.reset_index(inplace=True,drop=True)

train=train[:int(len(train)*.8)]


train=train.sample(n=len(train))
train.reset_index(inplace=True,drop=True)

labels = train.sort_values('label')

class_names = list(labels.label.unique())
class_names[:5]


newpath = r'./train' 
if not os.path.exists(newpath):
    os.makedirs(newpath)


for c in class_names:
    dest =  r'./train/'+str(c)
    os.makedirs(dest)
    for i in list(labels[labels['label']==c]['file name']): # Image Id
        get_image = os.path.join(base, i) # Path to Images 
        move_image_to_cat = shutil.copy(get_image, dest)
print("Step 3 complete...")

newpath = r'./test' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

for c in class_names:
    dest =  r'./test/'+str(c)
    os.makedirs(dest)
    for i in list(test[test['label']==c]['file name']): # Image Id
        get_image = os.path.join(base, i) # Path to Images 
        move_image_to_cat = shutil.copy(get_image, dest)
print("Step 4 complete...")



def create_sol():
    sol=pd.DataFrame()
    sol=sol.assign(filename=list(set(file['file name'].unique().tolist())-set(train['file name'].unique().tolist())))

    newpath = r'./sol' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    print(len(list(sol['filename'].unique())))
    x = 0
    for i in list(sol['filename'].unique()): # Image Id
            if(x%500==0):
                print(x)
            get_image = os.path.join(base, i) # Path to Images 
            move_image_to_cat = shutil.copy(get_image, newpath)
            x+=1
    print("Step 2 complete...")
