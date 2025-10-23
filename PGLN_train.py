# main.py

import os
import numpy as np
import pandas as pd
from utils.preprocess import process_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras._tf_keras.keras.losses import MeanSquaredError,MeanAbsoluteError
from networks.networks import ltc_network, cfc_network, ltc_fc_network, cfc_fc_network
from models.PGNN import PGNN
from utils.callbacks import LossLogger


model_name = 'PG_LTC_with_fully_connected_weights'

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

# Split data
X_train, X_val, y_train, y_val, t_train, t_val, T_train, T_val, Load_train, Load_val, RPM_train, RPM_val = train_test_split(
    X, y, t_data, T_data, Load, RPM, test_size=0.2, random_state=42,shuffle=True
)


def create_dataset(X, y, t_data, T_data, Load, RPM, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((
        (X, t_data, T_data),  # Inputs
        (y, (Load, RPM))      # Targets and physics data
    ))
    return dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# Create datasets
train_dataset = create_dataset(
    X_train, y_train, t_train, T_train, Load_train, RPM_train
)
val_dataset = create_dataset(
    X_val, y_val, t_val, T_val, Load_val, RPM_val
)


loss_logger = LossLogger()

model = PGNN(
    model_fn=ltc_fc_network,
    optimizer='adam',
    learning_rate=0.001,
    loss_fn= mse_loss,
    metrics_fn=mae_loss,
    dynamic_weights=True,
    model_name=model_name
)

model.summary()
# model.load_weights(f"{weights_dir}/{model_name}.keras")

model.compile()

model.train(train_dataset, val_dataset, epochs=100,callbacks=[loss_logger])
model.save_weights(f"{weights_dir}/{model_name}.keras")

df = pd.DataFrame(loss_logger.history)
df.to_csv(f'{stat_dir}/{model_name}_training_history.csv', index=False)

