# models/degradation_model.py

import os
import tensorflow as tf
from keras._tf_keras.keras.losses import get as get_loss
from keras._tf_keras.keras.optimizers import get as get_optimizer
from utils.callbacks import LossLogger
from utils.physics import dDdt
import numpy as np

stat_dir = 'statistics'
weights_dir = 'model_weights'




class UQ_DE(tf.keras.Model):
    def __init__(self, model_fn, input_shape=(16,), optimizer='adam',
                 learning_rate=0.001, loss_fn='mse', metrics_fn='mae', dynamic_weights=True, model_name='UQ_DE',**model_kwargs):
        super().__init__()

        self.model = model_fn(input_shape=input_shape, **model_kwargs)
        self.model_name = model_name
        self.loss_logger = LossLogger()
        self.dynamic_weights = dynamic_weights
        self.loss_fn = get_loss(loss_fn)
        self.metrics_fn = get_loss(metrics_fn)
        self.optimizer = get_optimizer({
            'class_name': optimizer,
            'config': {'learning_rate': learning_rate}
        })


    def compile(self):
        super().compile(optimizer=self.optimizer)


    def train_step(self, data):
        # Unpack the data
        inputs, (y_true, physics_data) = data
        X_batch, t_batch, T_batch = inputs
        Load_batch, RPM_batch = physics_data

        X_batch = tf.cast(X_batch, tf.float32)
        t_batch = tf.cast(t_batch, tf.float32)
        T_batch = tf.cast(T_batch, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        Load_batch = tf.cast(Load_batch, tf.float32)
        RPM_batch = tf.cast(RPM_batch, tf.float32)
        
        # Create physics inputs
        batch_size = tf.shape(X_batch)[0]
        t_batch = tf.cast(t_batch, tf.float32)
        T_batch = tf.cast(T_batch, tf.float32)

        X_physics_batch = tf.concat([
            tf.zeros((batch_size, 14), dtype=tf.float32),
            t_batch,
            T_batch
        ], axis=1)

        
        with tf.GradientTape() as tape:
            # Data loss
            y_pred = self.model(X_batch, training=True)
            data_loss = self.loss_fn(y_true, y_pred)
            
            # Physics loss
            with tf.GradientTape() as phys_tape:
                phys_tape.watch(X_physics_batch)
                D_pred_physics = self.model(X_physics_batch, training=True)
            
            # Calculate time derivative
            dD_dt = phys_tape.gradient(D_pred_physics, X_physics_batch)[:, 14:15]

            # Physics-based prediction
            physics_pred = dDdt(t_batch, Load_batch, RPM_batch, T_batch)
            physics_loss = self.loss_fn(dD_dt, physics_pred)

            if self.dynamic_weights:
                phyx_weight = tf.math.reduce_std(physics_pred, keepdims=False)/tf.math.reduce_std(X_batch, keepdims=False)
                data_weight = 1 - phyx_weight
                weights_raw = tf.stack([phyx_weight, data_weight])
                weights = tf.nn.softmax(weights_raw)
                phyx_weight = weights[0]
                data_weight = weights[1]
            else:
                phyx_weight = 1
                data_weight = 1

            total_loss = (data_weight * data_loss) + (phyx_weight * physics_loss)
        
        # Compute gradients and update weights
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return {
            "loss": tf.reduce_mean(total_loss),
            "data_loss": tf.reduce_mean(data_loss),
            "physics_loss": tf.reduce_mean(physics_loss),
        }

    def test_step(self, data):
        (X_batch, _, _), (y_true, _) = data
        X_batch = tf.cast(X_batch, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = self.model(X_batch, training=False)
        return {
            "loss": self.loss_fn(y_true, y_pred),
            "score": self.metrics_fn(y_true, y_pred)
        }

    def train(self, train_dataset, val_dataset=None, epochs=100, callbacks=None):
        self.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

    def save_weights(self, path):
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        self.model.save(path)


    def summary(self):
        dummy_input = tf.zeros((1, self.model.input_shape[-1]), dtype=tf.float32)
        self.model(dummy_input)
        return self.model.summary()

    def load_weights(self, path):
        if os.path.exists(path):
            self.model.load_weights(path)
            print(f"Loaded weights from: {path}")
        else:
            raise FileNotFoundError(f"No weights found at: {path}")

    
    def _predict(self, X, num_models=10, noise_scale=0.01):

        X = tf.cast(X, tf.float32)
        base_weights = self.model.get_weights()  # Save original weights
        preds = []
        for _ in range(num_models):
            perturbed_weights = [w + np.random.normal(0, noise_scale, size=w.shape) for w in base_weights]
            self.model.set_weights(perturbed_weights)
            pred = self.model(X, training=False).numpy()
            preds.append(pred)
        self.model.set_weights(base_weights)
        all_samples = np.stack(preds, axis=0)
        mean_pred = np.mean(all_samples, axis=0)
        epistemic_unc = np.var(all_samples, axis=0)
        return mean_pred.squeeze(), epistemic_unc.squeeze()

