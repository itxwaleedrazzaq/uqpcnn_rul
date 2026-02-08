import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation,Dropout, Reshape, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
import official.nlp.modeling.layers as nlp_layers
import evidential_deep_learning as edl
from ncps.tf import LTC,CfC
from ncps.wirings import AutoNCP

class NormalInverseGamma(Layer):
    def __init__(self, base_layer, **kwargs):
        super(NormalInverseGamma, self).__init__(**kwargs)
        self.base_layer = base_layer

    def build(self, input_shape):
        super().build(input_shape)
        if not self.base_layer.built:
            self.base_layer.build(input_shape)

    def evidence(self, x):
        return tf.nn.softplus(x)

    def call(self, inputs):
        output = self.base_layer(inputs)
        num_features = output.shape[-1]
        if num_features != 4:
            raise ValueError(f"Expected output of shape (..., 4), got (..., {num_features})")
        
        mu, logv, logalpha, logbeta = tf.split(output, 4, axis=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return tf.concat([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return self.base_layer.compute_output_shape(input_shape)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({
            'base_layer': tf.keras.layers.serialize(self.base_layer),
        })
        return base_config

    @classmethod
    def from_config(cls, config):
        base_layer = tf.keras.layers.deserialize(config.pop('base_layer'))
        return cls(base_layer=base_layer, **config)



#SNGP_network
def SNGP_network(input_shape=(16,),gamma=0.5):
        inputs = Input(input_shape)
        x = nlp_layers.SpectralNormalization(Dense(32),norm_multiplier=0.9)(inputs)
        x = Activation('relu')(x)
        x = nlp_layers.SpectralNormalization(Dense(32),norm_multiplier=0.9)(x)
        x = Activation('relu')(x)
        x = nlp_layers.SpectralNormalization(Dense(64),norm_multiplier=0.9)(x)
        x = Activation('relu')(x)
        x = nlp_layers.SpectralNormalization(Dense(64),norm_multiplier=0.9)(x)
        x = Activation('relu')(x)
        x = nlp_layers.SpectralNormalization(Dense(32),norm_multiplier=0.9)(x)
        x = Activation('relu')(x)
        x = nlp_layers.SpectralNormalization(Dense(32),norm_multiplier=0.9)(x)
        x = Activation('relu')(x)
        mu,cov = nlp_layers.RandomFeatureGaussianProcess(units=1,
            gp_cov_momentum=-1,
            gp_kernel_scale=gamma,
            gp_cov_ridge_penalty=1.0,
            scale_random_features=False,
            normalize_input=True,
            num_inducing=1024,)(x)
        return Model(inputs=inputs, outputs=[mu,cov])

#without SN GP network
def GP_network(input_shape=(16,),gamma=0.5):
        inputs = Input(input_shape)
        x  = Dense(32)(inputs)
        x = Activation('relu')(x)
        x  = Dense(64)(x)
        x = Activation('relu')(x)
        x  = Dense(64)(x)
        x = Activation('relu')(x)
        x  = Dense(32)(x)
        x = Activation('relu')(x)
        x  = Dense(32)(x)
        x = Activation('relu')(x)
        mu,cov = nlp_layers.RandomFeatureGaussianProcess(units=1,
            gp_cov_momentum=-1,
            gp_kernel_scale=gamma,
            gp_cov_ridge_penalty=1.0,
            scale_random_features=False,
            normalize_input=True,
            num_inducing=1024,)(x)
        return Model(inputs=inputs, outputs=[mu,cov])

#SNER network
def SNER_network(input_shape=(16,)):
    inputs = Input(input_shape)
    x = nlp_layers.SpectralNormalization(Dense(32),norm_multiplier=0.9)(inputs)
    x = Activation('relu')(x)
    x = nlp_layers.SpectralNormalization(Dense(32),norm_multiplier=0.9)(x)
    x = Activation('relu')(x)
    x = nlp_layers.SpectralNormalization(Dense(64),norm_multiplier=0.9)(x)
    x = Activation('relu')(x)
    x = nlp_layers.SpectralNormalization(Dense(64),norm_multiplier=0.9)(x)
    x = Activation('relu')(x)
    x = nlp_layers.SpectralNormalization(Dense(32),norm_multiplier=0.9)(x)
    x = Activation('relu')(x)
    x = nlp_layers.SpectralNormalization(Dense(32),norm_multiplier=0.9)(x)
    x = Activation('relu')(x)
    out = edl.layers.DenseNormalGamma(1)(x)
    return Model(inputs=inputs, outputs=out)


#ER network
def ER_network(input_shape=(16,)):
    inputs = Input(input_shape)
    x  = Dense(32)(inputs)
    x = Activation('relu')(x)
    x  = Dense(64)(x)
    x = Activation('relu')(x)
    x  = Dense(64)(x)
    x = Activation('relu')(x)
    x  = Dense(32)(x)
    x = Activation('relu')(x)
    x  = Dense(32)(x)
    x = Activation('relu')(x)
    out = edl.layers.DenseNormalGamma(1)(x)
    return Model(inputs=inputs, outputs=out)

#MLP based MC network
def mc_network(input_shape=(16,), dropout_rate=0.1):
    inp = Input(shape=input_shape)
    x = Dense(32, activation='relu')(inp)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, activation='linear')(x)
    return Model(inp, out)

#ELTC network
def eltc_network(input_shape=(16,)):
    inp = Input(shape=input_shape)
    x = Reshape((1,input_shape[0]))(inp)
    x = Conv1D(32,kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(16,kernel_size=2, activation='relu', padding='same')(x)
    out = NormalInverseGamma(LTC(units=AutoNCP(8,4)))(x)
    return Model(inp, out)

#CFC network
def ecfc_network(input_shape=(16,)):
    inp = Input(shape=input_shape)
    x = Reshape((1,input_shape[0]))(inp)
    x = Conv1D(32,kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(16,kernel_size=2, activation='relu', padding='same')(x)
    out = NormalInverseGamma(CfC(units=AutoNCP(8,4)))(x)
    return Model(inp, out)


#MC networks

def cfc_network(input_shape=(16,),dropout_rate=0.1):
    inp = Input(shape=input_shape)
    x = Reshape((1,input_shape[0]))(inp)
    x = Conv1D(32,kernel_size=3, activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(16,kernel_size=2, activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    out = CfC(units=AutoNCP(8,1), activation='linear')(x)
    return Model(inp, out)

#ltc network
def ltc_network(input_shape=(16,),dropout_rate=0.1):
    inp = Input(shape=input_shape)
    x = Reshape((1,input_shape[0]))(inp)
    x = Conv1D(32,kernel_size=3, activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(16,kernel_size=2, activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    out = LTC(units=AutoNCP(8,1))(x)
    return Model(inp, out)
