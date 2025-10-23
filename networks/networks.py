# architectures/lstm_conv_arch.py

from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input,Dense,Reshape,Conv1D,LSTM,GRU
from keras._tf_keras.keras.regularizers import l2
from ncps.wirings import AutoNCP,FullyConnected
from ncps.keras import CfC, LTC
from tcn import TCN



#mlp network
def mlp_network(input_shape=(16,)):
    inp = Input(shape=input_shape)
    x = Dense(32, activation='relu')(inp)
    x = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(1, activation='linear')(x)
    return Model(inp, out)

#cnn network
def cnn_network(input_shape=(16,)):
    inp = Input(shape=input_shape)
    x = Reshape((1,input_shape[0]))(inp)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(64, kernel_size=2, activation='relu', padding='same')(x)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(32, kernel_size=2, activation='relu', padding='same')(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(1, activation='linear')(x)
    return Model(inp, out)

#lstm network
def lstm_network(input_shape=(16,)):
    inp = Input(shape=input_shape)
    x = Reshape((1,input_shape[0]))(inp)  # Reshape for LSTM
    x = Conv1D(16, kernel_size=5, dilation_rate=1, activation='relu', padding='same')(x)
    x = LSTM(20,activation='relu',dropout=0.0, return_sequences=False)(x)
    x = Dense(20, activation='relu',kernel_regularizer=l2(1.6091060903581087e-05))(x)
    out = Dense(1, activation='linear')(x)
    return Model(inp, out)

#gru network
def gru_network(input_shape=(16,)):
    inp = Input(shape=input_shape)
    x = Reshape((1,input_shape[0]))(inp)  # Reshape for LSTM
    x = Conv1D(32, kernel_size=5, dilation_rate=1, activation='relu', padding='same')(x)
    x = GRU(64,activation='relu', dropout=0.0, return_sequences=False)(x)
    x = Dense(20, activation='relu',kernel_regularizer=l2(0.0044))(x)
    out = Dense(1, activation='linear')(x)
    return Model(inp, out)

#tcn network


# CFC network
def cfc_network(input_shape=(16,)):
    inp = Input(shape=input_shape)
    x = Reshape((1,input_shape[0]))(inp)
    x = Conv1D(32,kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(16,kernel_size=2, activation='relu', padding='same')(x)
    out = CfC(units=AutoNCP(8,1), activation='linear')(x)
    return Model(inp, out)

#ltc network
def ltc_network(input_shape=(16,)):
    inp = Input(shape=input_shape)
    x = Reshape((1,input_shape[0]))(inp)
    x = Conv1D(32,kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(16,kernel_size=2, activation='relu', padding='same')(x)
    out = LTC(units=AutoNCP(8,1))(x)
    return Model(inp, out)


#TCN network
def tcn_network(input_shape=(16,)):
    inp = Input(shape=input_shape)
    x = Reshape((1,input_shape[0]))(inp)
    x = TCN(64, kernel_size=5, dilations=[1, 2, 4], return_sequences=True, activation='relu', padding='same')(x)
    x = TCN(32, kernel_size=3, dilations=[1, 2], return_sequences=False, activation='relu', padding='same')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(20, activation='relu')(x)
    out = Dense(1, activation='linear')(x)
    return Model(inp, out)
    


    
# CFC-FC network
def cfc_fc_network(input_shape=(16,)):
    inp = Input(shape=input_shape)
    x = Reshape((1,input_shape[0]))(inp)
    x = Conv1D(32,kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(16,kernel_size=2, activation='relu', padding='same')(x)
    out = CfC(units=FullyConnected(16,1), activation='linear')(x)
    return Model(inp, out)

#ltc-FC network
def ltc_fc_network(input_shape=(16,)):
    inp = Input(shape=input_shape)
    x = Reshape((1,input_shape[0]))(inp)
    x = Conv1D(32,kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(16,kernel_size=2, activation='relu', padding='same')(x)
    out = LTC(units=FullyConnected(16,1))(x)
    return Model(inp, out)