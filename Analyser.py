from pydub import AudioSegment
import librosa
import os
import numpy
import pandas as pd
from sklearn.neural_network import MLPClassifier
"""
healthy_train = list()
unhealthy_train = list()
healthy_val = list()
unhealthy_val = list()
directory = "heart_sound/train/healthy"
i=0
for filename in os.listdir(directory):
    x, fs = librosa.load(directory+'/'+filename)
    mfccs = librosa.feature.mfcc(x, sr=fs)
    average_mfccs = [numpy.average(i) for i in mfccs]
    healthy_train.append(average_mfccs)
    print(i)
    i+=1
directory = "heart_sound/train/unhealthy"
for filename in os.listdir(directory):
    x, fs = librosa.load(directory+'/'+filename)
    mfccs = librosa.feature.mfcc(x, sr=fs)
    average_mfccs = [numpy.average(i) for i in mfccs]
    unhealthy_train.append(average_mfccs)
directory = "heart_sound/val/healthy"
for filename in os.listdir(directory):
    x, fs = librosa.load(directory+'/'+filename)
    mfccs = librosa.feature.mfcc(x, sr=fs)
    average_mfccs = [numpy.average(i) for i in mfccs]
    healthy_val.append(average_mfccs)
directory = "heart_sound/val/unhealthy"
for filename in os.listdir(directory):
    x, fs = librosa.load(directory+'/'+filename)
    mfccs = librosa.feature.mfcc(x, sr=fs)
    average_mfccs = [numpy.average(i) for i in mfccs]
    unhealthy_val.append(average_mfccs)
healthy_val_ds = pd.DataFrame(healthy_val)
healthy_val_ds.to_csv("hv.csv")
unhealthy_val_ds = pd.DataFrame(unhealthy_val)
unhealthy_val_ds.to_csv("uhv.csv")
healthy_train_ds = pd.DataFrame(healthy_train)
healthy_train_ds.to_csv("ht.csv")
unhealthy_train_ds = pd.DataFrame(unhealthy_train)
unhealthy_train_ds.to_csv("uht.csv")"""
healthy_train = pd.read_csv("ht.csv").to_numpy()[:, 1:]
unhealthy_train = pd.read_csv("uht.csv").to_numpy()[:, 1:]
healthy_val = pd.read_csv("hv.csv").to_numpy()[:, 1:]
unhealthy_val = pd.read_csv("uhv.csv").to_numpy()[:, 1:]
print(len(unhealthy_train))
clf = MLPClassifier(activation='logistic', random_state=1, max_iter=300)

clf.fit(numpy.concatenate((healthy_train, unhealthy_train)), [0 for i in range(len(healthy_train))] + [1 for i in range(len(unhealthy_train))])
healthy_predict = clf.predict(healthy_val)
unhealthy_predict = clf.predict(unhealthy_val)

f1 = open('healthy_predict', 'w')

print(1 - sum(healthy_predict)/len(healthy_predict))

f2 = open('unhealthy_predict', 'w')

print(sum(unhealthy_predict)/len(unhealthy_predict))
