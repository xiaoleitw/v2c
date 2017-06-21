#!/usr/bin/env python
#coding=utf-8  
'''CREATED:2014-05-22 16:43:44 by Brian McFee <brm2132@columbia.edu>

Pitch-shift a recording to be in A440 tuning.

Usage: ./adjust_tuning.py [-h] input_file output_file
'''
#from __future__ import print_function

import argparse
import sys
import librosa
import numpy as np 
import glob

from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Conv2D, MaxPooling2D 
from keras.models import Model
from keras.utils import plot_model
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

#from keras.datasets import imdb

import pickle



PN = 10 
input_file = '/home/xl/Documents/voice2text/testmp3/0_开放车门锁.mp3'

output_file = "/home/xl/Documents/voice2text/out.wav"
pitchshiftset = np.arange(PN) - int(0.5*PN) 
timestretchset = [0.75, 1, 1.5, 2, 4]
mp3fileset = glob.glob("/home/xl/Documents/voice2text/testmp3/*.mp3" )
wavfileset = glob.glob("/home/xl/Documents/voice2text/librosa/examples/*.wav")
#print wavfileset 

outfolder = "/home/xl/Documents/voice2text/librosa/examples/"

import os
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

#plt.switch_backend('Qt4Agg')
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
np.random.seed(1337)  # for reproducibility                                                          
MAXLEN = 800 
min_pitch = 38  # in 
batch_size = 128                                                                                                        
max_pitch = 78  # in MIDI  
MAXROW = 600
MAXY = 40 

# to their embedding vector
def figures(history, figure_name="plots100"):
    import matplotlib.pyplot as plt

    hist = history.history
    epoch = history.epoch
    acc = hist['acc']
    loss = hist['loss']
    val_loss = hist['val_loss']
    val_acc = hist['val_acc']

    plt.figure(1)

    plt.subplot(221)
    plt.plot(epoch, acc)
    plt.title("Training accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(222)
    plt.plot(epoch, loss)
    plt.title("Training loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(223)
    plt.plot(epoch, val_acc)
    plt.title("Validation Acc vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")

    plt.subplot(224)
    plt.plot(epoch, val_loss)
    plt.title("Validation loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")

    plt.tight_layout()
    plt.savefig(figure_name)

def myFeatureExtraction(PATH_TEST_FILE):

    y, sr = librosa.load(PATH_TEST_FILE, sr=8000)
    print PATH_TEST_FILE, sr, len(y)
    #label = PATH_TEST_FILE
    # str = "Line1-abcdef \nLine2-abc \nLine4-abcd";
    #print str.split( )
    # /home/xl/Documents/voice2text/testmp3/11_请车窗锁开.mp3 8000 24660

    s1 = PATH_TEST_FILE.split('/')
    #print s1[-1]
    s2 =  s1[-1].split('_')
    #print s2
    #print s2[0] 
    #print s2[1]
    #print s2[2]

    label = int(s2[0]) 
    print label 

    S = librosa.core.spectrum.stft(y, n_fft=1024, hop_length=80, win_length=1024)
    x_spec = np.abs(S)
    log_S = librosa.logamplitude(x_spec, ref_power=np.max)
    log_S = log_S.astype(np.float32)
    #[lm,ln] = log_S.shape
    #print 
    result = np.zeros((MAXROW, MAXLEN)) 
    if log_S.shape[1] > MAXLEN: 
        log_S = log_S[:, :MAXLEN] 

    result[:log_S.shape[0],:log_S.shape[1]] = log_S
    #log_S = np.lib.pad(log_S, (,MAXLEN), 'constant', constant_values=(0))
    
    return result , label 
        


def making_multi_frame(features, num_frames = 11): 

    max_bin = 256
    min_bin = 0    
    max_num = np.shape(features)[1]
    x = np.zeros(shape = (max_num, num_frames*(max_bin-min_bin)))

    h_frames = int((num_frames-1)/2)
    total_num = 0

    for j in range(max_num):
        if num_frames > 1 :
            if j < h_frames:
                x[total_num] = np.reshape(features[min_bin:max_bin,0:num_frames], (num_frames*(max_bin-min_bin)))
            elif j >= max_num - h_frames:
                x[total_num] = np.reshape(features[min_bin:max_bin,np.shape(features)[1]- num_frames:], (num_frames*(max_bin-min_bin)))
            else:    
                x[total_num] = np.reshape(features[min_bin:max_bin,j-h_frames:j+h_frames+1], (num_frames*(max_bin-min_bin)))

        else:
            x[total_num] = features[min_bin:max_bin,j-h_frames:j+h_frames+1].T
        total_num = total_num + 1
    return x
    

def MelodyExtraction_SCDNN(x_test,note_res=1):
    '''
    '''    
    dim_output = 40*note_res + 1 
       
    # DNN training
    model = Sequential()
    model.add(Dense(output_dim= 512, input_dim=x_test.shape[1]))
    model.add(Activation('relu')) #relu softplus
    model.add(Dropout(0.2))
    model.add(Dense(output_dim= 512, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim= 256, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim = dim_output))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer = RMSprop())
    y_predict = model.predict(x_test, batch_size=128, verbose=2)  
    print 'complete _res_'+str(note_res) +'_SCDNN'

    return y_predict
    


def adjust_tuning(input_file, output_file):
    '''Load audio, estimate tuning, apply pitch correction, and save.'''
    #print('Loading ', input_file)
    for inputwav in mp3fileset: 
        y, sr = librosa.load(inputwav)
        s1 = inputwav.split('/')
        fname = s1[-1] 
        #print s1[-1]
        #$s2 =  s1[-1].split('_')
        #print inputwav 
        #D = y 
        #print('Separating harmonic component ... ')
        #y_harm = librosa.effects.harmonic(y)

        #print pitchshiftlen
        #print('{:+0.2f} cents'.format(100 * tuning))
        #print('Applying pitch-correction of {:+0.2f} cents'.format(-100 * tuning)) 
        #H, P = librosa.decompose.hpss(D, margin=3.0)
        #R = D - (H+P)
        for i in pitchshiftset: 
            #print('Estimating tuning ... ')
            # Just track the pitches associated with high magnitude
            #tuning = librosa.estimate_tuning(y=y_harm, sr=sr
            for j in timestretchset: 
                y_tuned = librosa.effects.pitch_shift(y, sr, n_steps=i)
                y_stretch = librosa.effects.time_stretch(y_tuned, j) 
                #y_tuned = librosa.effects.pitch_shift(y, sr, -tuning)
                output_file = outfolder + fname[:-4] + '_tune' +str(i) +'_speed' + str(j) +'.wav'
                #print output_file 
                #print('Saving tuned audio to: ', output_file)
                librosa.output.write_wav(output_file, y_stretch, sr)
            


def process_arguments(args):
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='Tuning adjustment example')

    parser.add_argument('input_file',
                        action='store',
                        help='path to the input file (wav, mp3, etc)')

    parser.add_argument('output_file',
                        action='store',
                        help='path to store the output wav')

    return vars(parser.parse_args(args))




def model2(X, y): 
    #MAXY = max(y)
    labellist = [] 
    #print MAXY 

    if 0:#max(int(y)) > 2: 
        for i in range(len(y)) :   
            labels = np.zeros((1,MAXY) )   
            #print labels.shape 

            ind = int(y[i]) 
            labels[0][ind] = 1 
            #print labels 
            labellist.append(labels)

    if 1:
        for i in range(len(y)) : 
            ind = int(y[i]) 
            labellist.append(ind) 

    MAXY = len(y)

    x_train, x_test, y_train, y_test = train_test_split(X, labellist, test_size=0.2, random_state=42)

    #print x_test
    #print y_test 
    #x_train = x_train.astype('float32')
    x_train = np.array(x_train) 
    x_test = np.array(x_test)
    #y_train = np.array(y_train)
    #y_test = np.array(y_test)

    print('Train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print('Training model.')
    print x_train.shape
    x_train = np.expand_dims(x_train, axis = 3) 
    x_test = np.expand_dims(x_test, axis = 3) 
    y_test = np.squeeze(y_test, axis=(1,))
    y_train = np.squeeze(y_train, axis = (1,))
    #y_test = np.expand_dims(y_test, axis = 3) 
    #y_train = np.expand_dims(y_train, axis = 3) 
    print x_train.shape
    print y_train.shape 
    print y_test.shape 


    # train a 1D convnet with global maxpooling
    #sequence_input = Input(shape=(None, MAXROW, MAXLEN), dtype='float32')
    #embedded_sequences = embedding_layer(sequence_input)
    #keras.layers.convolutional.Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

    #x = Conv2D(128, (3,3), activation='relu')(sequence_input)
    #x = MaxPooling2D(x)
    #x = Conv2D(128, (3,3), activation='relu')(x)
    #x = MaxPooling2D(x)
    #x = Flatten()(x)
    #x = Dense(128, activation='relu')(x)
    #x = LSTM(32, input_shape=(MAXROW, MAXLEN))(sequence_input) 
    #x = LSTM(16)(x)
    del X
    del y 
    model = Sequential()
    #model.add(Embedding(input_shape = (MAXROW, MAXLEN,1) ,1000))
    if 0 : 
        model.add(Conv2D(16,  kernel_size = 3 , input_shape = (MAXROW, MAXLEN,1), border_mode='same', activation='relu', data_format='channels_last'))
        model.add(MaxPooling2D())
        #model.add(Conv2D(16, (3,3), activation ='relu')) 
        #model.add(MaxPooling2D())
        #model.add(Conv2D(16, (3,3), activation ='relu')) 
        #model.add(MaxPooling2D())
        #model.add(Dropout(0.2))
        #model.add(Conv2D(512, 3, 3, activation='relu', name='conv5_3'))
        #model.add(MaxPooling2D((2, 2), strides=(2, 2)))
        #model.add(Flatten())
        #model.add(Dense(256, activation = 'softmax')) 
        model.add(Dropout(0.2))
        model.add(Dense(128, activation = 'softmax')) 
        model.add(Dropout(0.2))
        #model.add(GRU(100,return_sequences=True))
        #model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
        model.add(Dense(MAXY, activation='softmax')) 
        #model.compile('rmsprop', 'categorical_crossentropy')
        model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['acc'])

    if 1: 

        model = Sequential()
        model.add(Conv2D(32, kernel_size = 3, input_shape=(MAXROW, MAXLEN,1), data_format='channels_last')) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, kernel_size = 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, kernel_size = 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])


    #seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
    #               activation='sigmoid',
    #               padding='same', data_format='channels_last'))
    #seq.add(Dense(MAXY, activation='softmax')) 

    #preds = Dense(MAXY, activation='softmax')(x)

    #model = Model(sequence_input, preds)
    #
    #model = seq 
    hist = model.fit(x_train, y_train,
              batch_size=64,
              epochs=10,
              validation_data=(x_test, y_test))
    MODELSAVED = "glovemodel.h5" 
    model.save(MODELSAVED)
    print(hist.history)
    figname = "Iteration_" + MODELSAVED + '.png'
    figures(hist, figname)
    plot_model(model, to_file='model.png')
    return True 

def model3(X,y): 
    print('Build model...')
    model = Sequential()
    model.add(Dense(output_dim= MAXROW, input_dim=(MAXROW, MAXLEN,)))
    model.add(Activation('relu')) #relu softplus
    model.add(Dropout(0.2))
    model.add(Dense(output_dim= 512, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim= 256, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim = dim_output))
    model.add(Activation('sigmoid'))
    #model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    #model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=15,
              validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


def main(param=0.2, PATH_LOAD_FILE=input_file, PATH_SAVE_FILE='./SAVE_RESULTS/pop1.txt'):


    #i = 1 
    if 1 : 
        X = [] 
        y = [] 
        wav2fileset = glob.glob("/home/xl/Documents/voice2text/librosa/examples/[19]_*.wav")

        for inputwav in wav2fileset: 
            x_test_log, label  = myFeatureExtraction(inputwav) 
            print x_test_log.shape, label 
            X.append(x_test_log)
            y.append(label) 

        with open('WAV2objs.pickle', 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump([X, y], f)

    # Getting back the objects:
    if 0: 
        with open('objs.pickle') as f:  # Python 3: open(..., 'rb')
            X,y = pickle.load(f)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    #print X_train
    #print y_train 
    #print X_train.shape
    #print y_train.shape 
    dim_output = 1 
    newy = [] 
    newx = [] 

    for i in range(len(y)) : 
        if int(y[i]) == 1: 
            newy.append(1)
            newx.append(X[i]) 
        if  int(y[i])  == 9: 
            newy.append(0)
            newx.append(X[i]) 


    #test1 = model2(X,y) 
    test1 = model2(newx, newy) 

    #y_predict_1st = MelodyExtraction_SCDNN(x_test_MF, 1)
    #print y_predict_1st.shape 
    #print y_predict_1st



    return True 

if __name__ == '__main__':
    # Get the parameters
    #params = process_arguments(sys.argv[1:])

    # Run the beat tracker
    #adjust_tuning(params['input_file'], params['output_file'])
    #adjust_tuning(input_file, output_file) 
    main(param=0.2, PATH_LOAD_FILE=input_file, PATH_SAVE_FILE='./SAVE_RESULTS/pop1.txt')

else:
     param = sys.argv[1]           
     PATH_LOAD_FILE = sys.argv[2]   
     PATH_SAVE_FILE = sys.argv[3] 
     #main()
