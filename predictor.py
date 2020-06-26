from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf


class TsPredictor(ABC):

    @abstractmethod
    def train(self, x_train: np.array, y_train: np.array,
              x_val: Optional[np.array] = None, y_val: Optional[np.array] = None):
        pass

    @abstractmethod
    def predict(self, history: np.array):
        pass


class UnivariateTsPredictor(TsPredictor):

    def __init__(self, input_shape: Tuple[int, int],
                 epochs: int = 10, train_steps: int = 500, validation_steps: int = 50,
                 batch_size: int = 256, buffer_size: int = 10000):
        self.input_shape = input_shape
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(8, input_shape=input_shape),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mae')
        self.epochs = epochs
        self.train_steps = train_steps
        self.validation_steps = validation_steps
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def train(self, x_train: np.array, y_train: np.array,
              x_val: Optional[np.array] = None, y_val: Optional[np.array] = None):
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()

        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_data = val_data.batch(self.batch_size).repeat()

        self.model.fit(train_data, epochs=self.epochs, steps_per_epoch=self.train_steps,
                       validation_data=val_data, validation_steps=self.validation_steps)

    def predict(self, history: np.array):
        if history.shape == self.input_shape:
            shape = (1,) + self.input_shape
            history = history.reshape(shape)
        return self.model.predict(history)[0][0]


class MultivariateTsPredictor(TsPredictor):

    def __init__(self, input_shape: Tuple[int, int],
                 epochs: int = 10, train_steps: int = 500, validation_steps: int = 50,
                 batch_size: int = 256, buffer_size: int = 10000):
        self.input_shape = input_shape
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(32, input_shape=input_shape),
            tf.keras.layers.Dense(1),
        ])
        self.model.compile(optimizer='adam', loss='mae')
        self.epochs = epochs
        self.train_steps = train_steps
        self.validation_steps = validation_steps
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def train(self, x_train: np.array, y_train: np.array,
              x_val: Optional[np.array] = None, y_val: Optional[np.array] = None):
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()

        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_data = val_data.batch(self.batch_size).repeat()

        self.model.fit(train_data, epochs=self.epochs, steps_per_epoch=self.train_steps,
                       validation_data=val_data, validation_steps=self.validation_steps)

    def predict(self, history: np.array):
        if history.shape == self.input_shape:
            shape = (1,) + self.input_shape
            history = history.reshape(shape)
        return self.model.predict(history)[0][0]


class MultivariateMultistepTsPredictor(TsPredictor):

    def __init__(self, input_shape: Tuple[int, int],
                 lstm1_units: int = 32, lstm2_units: int = 16,
                 epochs: int = 10, train_steps: int = 500, validation_steps: int = 50,
                 batch_size: int = 256, buffer_size: int = 10000):
        self.input_shape = input_shape
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(lstm1_units, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.LSTM(lstm2_units, activation='relu'),
            tf.keras.layers.Dense(72),
        ])

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
        self.epochs = epochs
        self.train_steps = train_steps
        self.validation_steps = validation_steps
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def train(self, x_train: np.array, y_train: np.array,
              x_val: Optional[np.array] = None, y_val: Optional[np.array] = None):
        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_data = train_data.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()

        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_data = val_data.batch(self.batch_size).repeat()

        self.model.fit(train_data, epochs=self.epochs, steps_per_epoch=self.train_steps,
                       validation_data=val_data, validation_steps=self.validation_steps)

    def predict(self, history: np.array):
        assert history.shape == self.input_shape
        shape = (1,) + self.input_shape
        history = history.reshape(shape)
        return self.model.predict(history).reshape(-1)

    def calc_loss(self, x_val, y_val):
        preds = self.model.predict(x_val)
        return np.abs(preds - y_val).sum()
