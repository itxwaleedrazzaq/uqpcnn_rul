# main.py

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
import tensorflow
import pandas as pd
from utils.preprocess import process_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from utils.callbacks import LossLogger, ResetStatesCallback
from keras._tf_keras.keras.losses import MeanSquaredError,MeanAbsoluteError
from networks.UQ_networks import SNGP_network, GP_network
from models.UQ_SNGP import UQ_SNGP
from networks.losses import NLL


model_name = 'SNGP_without_physics_guidance_gamma_1_weights'

feature_dir = 'tf_features'
weights_dir = 'model_weights'
stat_dir = 'statistics'

bearings = ['Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7']

mse_loss = MeanSquaredError()
mae_loss = MeanAbsoluteError()


# Load and preprocess data
dfs = [pd.read_csv(f'{feature_dir}/{bearing}_features.csv') for bearing in bearings]

# Process features
horizontal_data = [np.array(df['Horizontal'].apply(eval).tolist()) for df in dfs]
X_h = np.vstack([process_features(data) for data in horizontal_data])
vertical_data = [np.array(df['Vertical'].apply(eval).tolist()) for df in dfs]
X_v = np.vstack([process_features(data) for data in vertical_data])
vibration_features = np.concatenate((X_h, X_v), axis=-1)

# Get other features
t_data = np.concatenate([df['Time'].values.reshape(-1, 1) for df in dfs], axis=0)
T_data = np.concatenate([(df['Temperature'].values + 273.15).reshape(-1, 1) for df in dfs], axis=0)
y = np.concatenate([df['Degradation'].values.reshape(-1, 1) for df in dfs], axis=0)
RPM = np.concatenate([df['RPM'].values.reshape(-1, 1) for df in dfs], axis=0)
Load = np.concatenate([df['Load'].values.reshape(-1, 1) for df in dfs], axis=0)

# Combine features and normalize
X = np.concatenate([vibration_features, t_data, T_data], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)



loss_logger = LossLogger()

model = SNGP_network(gamma=1.0)

model.summary()
# model.load_weights(f"{weights_dir}/{model_name}.keras")

model.compile(tf.keras.optimizers.Adam(learning_rate=0.001),loss=NLL)


history = model.fit(X,y,epochs=50,validation_split=0.2,callbacks=[loss_logger])
model.save(f"{weights_dir}/{model_name}.keras")

df = pd.DataFrame(history.history)
df.to_csv(f'{stat_dir}/{model_name}_training_history.csv', index=False)

