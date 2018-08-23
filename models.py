from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, concatenate, Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras import backend as K


# Multitowered 1D-CNN for concatenated feature set
def concat_1d_CNN(n_class, n_col):    
    
    input_shape = Input(shape=(n_col,1))

    tower_1 = Conv1D(nb_filter = 5, kernel_size = 15, strides = 1, activation = 'relu')(input_shape)
    tower_1 = Dropout(0.25)(tower_1)
    tower_1 = (MaxPooling1D(pool_size=2))(tower_1)

    tower_2 = Conv1D(nb_filter = 5, kernel_size = 15, strides = 1, activation = 'relu')(input_shape)
    tower_2 = Dropout(0.25)(tower_2)
    tower_2 = (MaxPooling1D(pool_size=2))(tower_2)

    merged = concatenate([tower_1, tower_2], axis=1)
    merged = Flatten()(merged)

    out = Dense(200, activation='relu')(merged)
    out = Dropout(0.25)(out)
    out = Dense(50, activation='relu')(out)
    out = Dropout(0.1)(out)
    if n_class > 2:
        out = Dense(n_class, activation='softmax')(out)
    else:
        out = Dense(n_class, activation='sigmoid')(out)
    model = Model(input_shape, out)
    return model

# Mulitowered 1D-CNN for seperated feature set
def sep_1d_CNN(n_class, input_shape_dict):
    
    input_stft = Input(shape=(input_shape_dict['len_stft'],1))
    input_mfccs = Input(shape=(input_shape_dict['len_mfccs'],1))
    input_chroma = Input(shape=(input_shape_dict['len_chroma'],1))
    input_mel = Input(shape=(input_shape_dict['len_mel'],1))
    input_contrast = Input(shape=(input_shape_dict['len_contrast'],1))
    input_tonnetz = Input(shape=(input_shape_dict['len_tonnetz'],1))
    
    tower_stft = Conv1D(nb_filter = 5, kernel_size = 5, strides = 1, activation = 'relu', use_bias = False)(input_stft)
    tower_stft = Dropout(0.25)(tower_stft)
    tower_stft = (MaxPooling1D(pool_size=2))(tower_stft)
    
    tower_mfccs = Conv1D(nb_filter = 5, kernel_size = 5, strides = 1, activation = 'relu', use_bias = False)(input_mfccs)
    tower_mfccs = Dropout(0.25)(tower_mfccs)
    tower_mfccs = (MaxPooling1D(pool_size=2))(tower_mfccs)
    
    tower_chroma = Conv1D(nb_filter = 5, kernel_size = 5, strides = 1, activation = 'relu', use_bias = False)(input_chroma)
    tower_chroma = Dropout(0.25)(tower_chroma)
    tower_chroma = (MaxPooling1D(pool_size=2))(tower_chroma)
    
    tower_mel = Conv1D(nb_filter = 5, kernel_size = 5, strides = 1, activation = 'relu', use_bias = False)(input_mel)
    tower_mel = Dropout(0.25)(tower_mel)
    tower_mel = (MaxPooling1D(pool_size=2))(tower_mel)
    
    tower_contrast = Conv1D(nb_filter = 5, kernel_size = 5, strides = 1, activation = 'relu', use_bias = False)(input_contrast)
    tower_contrast = Dropout(0.25)(tower_contrast)
    tower_contrast = (MaxPooling1D(pool_size=2))(tower_contrast)
    
    tower_tonnetz = Conv1D(nb_filter = 5, kernel_size = 5, strides = 1, activation = 'relu', use_bias = False)(input_tonnetz)
    tower_tonnetz = Dropout(0.25)(tower_tonnetz)
    tower_tonnetz = (MaxPooling1D(pool_size=2))(tower_tonnetz)
    
    merged = concatenate([tower_stft, tower_mfccs, tower_chroma, tower_mel, tower_contrast, tower_tonnetz], axis=1)
    merged = Flatten()(merged)
    
    out = Dense(200, activation='relu', use_bias = False)(merged)
    out = Dropout(0.25)(out)
    out = Dense(50, activation='relu', use_bias = False)(out)
    out = Dropout(0.1)(out)
    if n_class > 2:
        out = Dense(n_class, activation='softmax')(out)
    else:
        out = Dense(n_class, activation='sigmoid')(out)
    
    model = Model([input_stft,input_mfccs,input_chroma,input_mel,input_contrast,input_tonnetz], out)
    return model