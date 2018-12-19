import numpy as np
import librosa.display
import os
import time
import copy
from sklearn.externals import joblib
def preprocess_mfcc(mfcc):
    mfcc_cp = copy.deepcopy(mfcc)
    for i in range(mfcc.shape[1]):
        mfcc_cp[:, i] = mfcc[:, i] - np.mean(mfcc[:, i])
        mfcc_cp[:, i] = mfcc_cp[:, i] / np.max(np.abs(mfcc_cp[:, i]))
    return mfcc_cp

start = time.perf_counter()

dirname = "C:/Users/heman/Downloads/Minor/NEW/Minor_Final/Dataset_Final/Training_Dataset"
files = [f for f in os.listdir(dirname) if not f.startswith('.')]
if "desktop.ini" in files:
    files.remove("desktop.ini")
print(type(files))

mfcc_arr = []


import json
with open('eng_dict.json') as data_file:
    eng_dict = json.load(data_file)

from sklearn.externals import joblib
filename = 'hin_dict'
hin_dict = joblib.load(filename)

y = []
for i in files:
    y.append(eng_dict[i.split("_")[0]])
print(y)

for i in range(len(files)):
    # print(i,end=' ')

    y1, sr1 = librosa.load(dirname + "/" + files[i])
    mfcc1 = librosa.feature.mfcc(y1, sr1)
    mfcc_arr.append(mfcc1)
for i in range(len(mfcc_arr)):
    mfcc_arr[i] = preprocess_mfcc(mfcc_arr[i])


# nw = int(len(files)/36)
# ll=0
# for i in range(nw):
#     ul = 36*(i+1)
#     for j in range(ll,ul):
#         y[j] = int(i)
#     ll = ul


joblib.dump(mfcc_arr, 'Training_mfcc_arr.pkl')
joblib.dump(y, 'Training_y.pkl')

print("\n\nTime used for Training MFCC Calculation: {}s".format(time.perf_counter() - start))