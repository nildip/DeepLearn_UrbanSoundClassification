import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
import warnings

from os import listdir
from os.path import isdir, join

# function to load data
def data_loader(dir_path):
    warnings.filterwarnings('ignore')
    alldirs = [d for d in listdir(dir_path)if isdir(join(dir_path, d))]
    out_X = []
    out_Y = []
    for i in alldirs:
        allfiles = [f for f in listdir(join(dir_path, i))if not f.endswith(('.csv','.json','.DS_Store'))]
        for fp in tqdm(allfiles):
            try: 
                X,sr = librosa.load(join(dir_path, i, fp), sr = 11025)
                out_X.append(X)
                out_Y.append(str(i))
            except:
                next
    return out_X, out_Y

# function to reshape all lists to common length by adding trailing 0s
def padder(X):
    max_len = max([len(i) for i in X])
    for j in tqdm(X):
        j.extend([0] * (max_len - len(j)))    
    return X,max_len

# function to extract features
def feat_extract_concat(X):
    out = []
    for i in tqdm(X):
        i = np.array(i)
        stft = np.mean(np.abs(librosa.stft(i)).T, axis = 0)
        mfccs = np.mean(librosa.feature.mfcc(i, sr = 11025).T, axis = 0)
        chroma = np.mean(librosa.feature.chroma_stft(i, sr = 11025).T, axis = 0)
        mel = np.mean(librosa.feature.melspectrogram(i, sr = 11025).T, axis = 0)
        contrast = np.mean(librosa.feature.spectral_contrast(i, sr = 11025, n_bands = 4).T, axis = 0)
        tonnetz = np.mean(librosa.feature.tonnetz(librosa.effects.harmonic(i), sr = 11025).T, axis = 0)
        out_tmp = list(stft)+list(mfccs)+list(chroma)+list(mel)+list(contrast)+list(tonnetz)
        out.append(out_tmp)
    return out

def feat_extract_sep(X):
    out_stft = []
    out_mfccs = []
    out_chroma = []
    out_mel = []
    out_contrast = []
    out_tonnetz = []
    out_cqt = []
    out_cens = []
    out_rmse = []
    out_spcentroid = []
    out_spband = []
    out_sproll = []
    out_poly = []
    out_zcr = []
    for i in tqdm(X):
        i = np.array(i)
        out_stft.append(list(np.mean(np.abs(librosa.stft(i)).T, axis = 0)))
        out_mfccs.append(list(np.mean(librosa.feature.mfcc(i, sr = 11025).T, axis = 0)))
        out_chroma.append(list(np.mean(librosa.feature.chroma_stft(i, sr = 11025).T, axis = 0)))
        out_mel.append(list(np.mean(librosa.feature.melspectrogram(i, sr = 11025).T, axis = 0)))
        out_contrast.append(list(np.mean(librosa.feature.spectral_contrast(i, sr = 11025, n_bands = 4).T, axis = 0)))
        out_tonnetz.append(list(np.mean(librosa.feature.tonnetz(librosa.effects.harmonic(i), sr = 11025).T, axis = 0)))
    return out_stft,out_mfccs,out_chroma,out_mel,out_contrast,out_tonnetz