from keras.models import Sequential, Model
import keras
from keras.layers import Merge, LSTM, Dense,GRU, SimpleRNN, core, Dropout, InputLayer,Input
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, AveragePooling1D, MaxPooling1D, LocallyConnected1D
from keras.layers import Flatten, Lambda, TimeDistributed, Activation, Permute, RepeatVector
from keras.optimizers import Adam, SGD, Adagrad,Adamax, Nadam
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import Concatenate, Average, Multiply, Add
from keras import backend as K
import numpy as np
import pickle
import re
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from pprint import pprint
from matplotlib import pyplot as plt
import pickle as pk
import utils
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        allow_growth=True
    )
)
set_session(tf.Session(config=config))


def create_seq2seq(n_class,attr_size=4,pad_size=48,out_size=8,attn=False,loss='mse',latent_dim=50):
    input_seq = Input(shape=(pad_size, n_class),name='input_seq')
    encoded,state_h,state_c = LSTM(
        latent_dim,dropout=0.3,return_sequences=False,
        kernel_regularizer=l2(0.00005),return_state=True,
        name='encoder'
    )(input_seq)
    states=[state_h,state_c]

    input_attr = Input(shape=(out_size,attr_size),name='input_attr')
    attr_encoded = TimeDistributed(Dense(50,name='attr_encoder'))(input_attr)

    #decoder_inputs = Input(shape=(1, n_class),name='decoder_inputs')
    decoder_inputs = Lambda(lambda x:x[:,-1:,:])(input_seq)
    attr_inputs = Lambda(lambda x:x[:,:1])(attr_encoded)
    decoder_lstm = LSTM(latent_dim,return_sequences=True,dropout=0.3,return_state=True,name='decoder')
    decoder_dense = Dense(n_class, activation='sigmoid',name='decoder_output')

    all_outputs = []
    merge = Merge(mode='concat',name='concat')
    inputs = merge([decoder_inputs,attr_inputs])
    i = 0
    while(1):
        # Run the decoder on one timestep
        outputs, state_h, state_c = decoder_lstm(inputs,
                                                 initial_state=states)
        outputs = decoder_dense(outputs)
        # Store the current prediction (we will concatenate all predictions later)
        all_outputs.append(outputs)
        # Reinject the outputs as inputs for the next loop iteration
        # as well as update the states
        i += 1
        if i >= out_size :
            i=0
            break
        else:
            attr_inputs = Lambda(lambda x:x[:,i:i+1])(attr_encoded)
            inputs = merge([outputs,attr_inputs])
            states = [state_h, state_c]

    # Concatenate all predictions
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

    model = Model([input_seq,input_attr],decoder_outputs)
    model.compile(optimizer=Adam(),loss=loss,metrics=['mae','accuracy'])
    
    return model

def create_conv_encoder_seq2seq(n_class,attr_size=4,pad_size=48,out_size=8,attn=False,loss='mse',latent_dim=50,kernel_sizes=[5]):
    input_seq = Input(shape=(pad_size, n_class),name='input_seq')
    convs = []
    for size in kernel_sizes:
        inner_conv = Conv1D(
                latent_dim,
                size,
                padding='same',
                #activation='relu',
                strides=2,
                batch_input_shape=(None, pad_size, n_class),init='uniform',#kernel_regularizer=l2(0.00005)
        )(input_seq)
        inner_conv = Conv1D(
                latent_dim,
                size,
                padding='same',
                #activation='relu',
                strides=2,
                init='uniform',#kernel_regularizer=l2(0.00005)
        )(inner_conv)
        inner_conv = MaxPooling1D()(inner_conv)
        inner_conv = Conv1D(
                latent_dim,
                int(pad_size/4),
                padding='same',
                #activation='relu',
                strides=1,
                init='uniform',#kernel_regularizer=l2(0.00005)
        )(inner_conv)
        inner_conv = Lambda(lambda x:x[:,0,:])(inner_conv)
        #inner_conv = GlobalMaxPooling1D()(inner_conv)
        convs.append(inner_conv)
        
    if len(kernel_sizes) > 1:
        encoded = Merge(mode='sum')(convs)
    else:
        encoded = convs[0]
    states = [encoded,encoded]
    input_attr = Input(shape=(out_size,attr_size),name='input_attr')
    attr_encoded = TimeDistributed(Dense(50,name='attr_encoder'))(input_attr)
    
    #decoder_inputs = Input(shape=(1, n_class),name='decoder_inputs')
    decoder_inputs = Lambda(lambda x:x[:,-1:,:])(input_seq)
    #decoder_inputs = Lambda(lambda x:x[:,np.newaxis])(encoded)
    attr_inputs = Lambda(lambda x:x[:,:1])(attr_encoded)
    decoder_lstm = LSTM(latent_dim,return_sequences=True,dropout=0.3,return_state=True,name='decoder')
    decoder_dense = Dense(n_class, activation='sigmoid',name='decoder_output')
    decoder_input_dense = Dense(latent_dim)

    all_outputs = []
    merge = Merge(mode='concat',name='concat')
    inputs = merge([decoder_inputs,attr_inputs])
    i = 0
    while(1):
        # Run the decoder on one timestep
        if i == 0:
            outputs, state_h, state_c = decoder_lstm(decoder_input_dense(inputs),initial_state=states)
        else:
            outputs, state_h, state_c = decoder_lstm(decoder_input_dense(inputs),initial_state=states)
        outputs = decoder_dense(outputs)
        # Store the current prediction (we will concatenate all predictions later)
        all_outputs.append(outputs)
        # Reinject the outputs as inputs for the next loop iteration
        # as well as update the states
        i += 1
        if i >= out_size :
            i=0
            break
        else:
            attr_inputs = Lambda(lambda x:x[:,i:i+1])(attr_encoded)
            inputs = merge([outputs,attr_inputs])
            states = [state_h, state_c]

    # Concatenate all predictions
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

    model = Model([input_seq,input_attr],decoder_outputs)
    model.compile(optimizer=Adam(),loss=loss,metrics=['mae','accuracy'])
    
    return model



def harmonic(target):
    return len(target)/sum([1/elem for elem in target])

def _mse(x,y):
    return K.mean(K.square(y - x), axis=-1)

def _mse_onIncrement(x,y):
    return K.mean(K.square((x[:,-1] - x[:,0]) - (y[:,-1] - y[:,0]))) / 10

def _mse_onTotal(x,y):
    return K.mean(K.square(K.sum(y,axis=-1) - K.sum(x,axis=-1)))

def _mae(x,y):
    return K.mean(K.abs(y - x), axis=-1)

def _mae_onIncrement(x,y):
    return K.mean(K.abs((x[:,-1] - x[:,0]) - (y[:,-1] - y[:,0]))) / 10

def _mae_onTotal(x,y):
    return K.mean(K.abs(K.sum(y,axis=-1) - K.sum(x,axis=-1)))

def mae_inc_total(x,y):
    return harmonic([_mae(x,y),_mae_onIncrement(x,y),_mae_onTotal(x,y)])

def mae_inc(x,y):
    return _mae(x,y) + _mae_onIncrement(x,y)

def mse_inc_total(x,y):
    return _mse(x,y) + _mse_onIncrement(x,y) + _mse_onTotal(x,y)

def mse_inc(x,y):
    return _mse(x,y) + _mse_onIncrement(x,y)