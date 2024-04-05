
import os

prev = "555555555"
for dirname, _, filenames in os.walk('C:\\Users\\mateo\\Desktop\\ABBY GOES HERE\\bees\\sound_files\\sound_files'):
    for filename in filenames:
        if(int(filename[-5])) != (int(prev[-5])+1):
            print(filename,prev)
            prev = filename
        # print(filename)
        # Id.append(os.path.join(dirname, filename))