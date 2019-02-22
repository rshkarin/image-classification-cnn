import os
import cv2
import time
import logging

import numpy as np
import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Input, Dense, Flatten, Dropout, Activation
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from model import utils

K.set_image_data_format('channels_last')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNN(object):
    """Convolutional neural network with adjustable architecture.
    """
    def __init__(self, input_shape=(128, 128), net_name='network', n_dense_neurons=1000, dropout_ratio=0.5, n_epochs=10,
                 pool_size=(2, 2), model_output_dir=None, activation_name='sigmoid', depth=4,
                 n_base_filters=16, n_classes=2, batch_normalization=False, min_receptive_field_size=None,
                 loss='binary_crossentropy', metrics=['accuracy'], learning_rate=1e-6,
                 early_stopping_patience=10, learning_rate_patience=5, validation_split=0.3):
        self.n_dense_neurons = n_dense_neurons
        self.dropout_ratio = dropout_ratio
        self.input_shape = tuple(input_shape)
        self.net_name = '_'.join([net_name, '{}d'.format(len(self.input_shape))])
        self.n_epochs = n_epochs
        self.activation_name = activation_name
        self.pool_size = pool_size
        self.depth = depth
        self.n_base_filters = n_base_filters
        self.n_classes = n_classes
        self.batch_normalization = batch_normalization
        self.loss = loss
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate_patience = learning_rate_patience
        self.validation_split = validation_split
        self.model_output_dir = '.' if model_output_dir is None else model_output_dir
        self.verbose = 1

        self.min_receptive_field_size = min_receptive_field_size
        if min_receptive_field_size is not None:
            if depth is not None:
                logger.warning('The depth will be recalculated automatically '
                'based on min_receptive_field_size; the corresponding variable '
                'will be overridden.')
            self.depth = self._recalculate_depth(min(self.input_shape),
                                                 min_receptive_field_size)

    @classmethod
    def model_from_file(cls, model_path, input_shape, n_dense_neurons=1000,
                        dropout_ratio=0.5, pool_size=(2, 2), activation_name='sigmoid',
                        depth=4, n_base_filters=16, n_classes=2,
                        batch_normalization=False, min_receptive_field_size=None):
        """Creates a network model from a model weights by `model_path` and arguments.
        """
        net_name = os.path.splitext(os.path.basename(model_path))[0]
        net = cls(input_shape, net_name, n_dense_neurons=n_dense_neurons,
                  dropout_ratio=dropout_ratio, pool_size=pool_size,
                  activation_name=activation_name, depth=depth,
                  n_base_filters=n_base_filters, n_classes=n_classes,
                  batch_normalization=batch_normalization,
                  min_receptive_field_size=min_receptive_field_size)
        net.model = net._get_model()
        net.model.load_weights(model_path)
        return net

    @classmethod
    def model_from_kwargs(cls, **kwargs):
        """Creates a network model from a set of arguments.
        """
        net = cls()
        net.__dict__.update(kwargs)
        net.input_shape = tuple(net.input_shape)
        net.model = net._get_model()
        net.model.load_weights(net.model_path)
        return net

    def _recalculate_depth(self, shape, min_size):
        """Recalculates a network depth by provided a min receptive field size.
        """
        val = min(shape)
        depth = 1
        while True:
            val /= 2
            if val > min_size:
                depth +=1
            else:
                break
        return depth

    def _get_callbacks(self, learning_rate_drop=0.5,
                       learning_rate_epochs=None, verbosity=1):
        """Returns a set of callbacks.
        """
        if not os.path.exists(self.model_output_dir):
            os.makedirs(self.model_output_dir)

        ts = time.strftime('%Y%m%d-%H%M%S')

        self.model_params_path = \
                    os.path.join(self.model_output_dir,
                                 '-'.join([self.net_name, ts, 'params']) + '.ini')
        self.model_path = \
                    os.path.join(self.model_output_dir,
                                 '-'.join([self.net_name, ts, 'model']) + '.hdf5')
        self.csv_log_path = \
                    os.path.join(self.model_output_dir,
                                 '-'.join([self.net_name, ts, 'training']) + '.log')

        callbacks = list()
        callbacks.append(ModelCheckpoint(self.model_path, monitor='val_loss',
                                         verbose=verbosity, save_best_only=True))
        callbacks.append(CSVLogger(self.csv_log_path, append=True))
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop,
                                           patience=self.learning_rate_patience,
                                           verbose=verbosity))
        callbacks.append(EarlyStopping(verbose=verbosity,
                                       patience=self.early_stopping_patience))
        return callbacks

    def _build_model(self):
        """Creates a network model by dynamically combining layers.
        """
        inputs = Input(self.input_shape+(1,))
        current_layer = self._create_convolution_block(input_layer=inputs,
                                                       kernel=(3, 3),
                                                       n_filters=self.n_base_filters*2,
                                                       batch_normalization=self.batch_normalization)
        current_layer = MaxPooling2D(pool_size=self.pool_size)(current_layer)

        for layer_depth in range(self.depth-1):
            layer1 = self._create_convolution_block(input_layer=current_layer,
                                                    kernel=(3, 3),
                                                    n_filters=(self.n_base_filters + layer_depth*2)*2,
                                                    batch_normalization=self.batch_normalization)
            layer2 = self._create_convolution_block(input_layer=layer1,
                                                    kernel=(3, 3),
                                                    n_filters=(self.n_base_filters + (layer_depth+1)*2)*2,
                                                    batch_normalization=self.batch_normalization)
            if layer_depth < (self.depth - 2):
                current_layer = MaxPooling2D(pool_size=self.pool_size)(layer2)
            else:
                current_layer = layer2

        layer = self._create_dense_block(current_layer,
                                         self.n_dense_neurons,
                                         dropout_ratio=self.dropout_ratio)
        predictions = Dense(self.n_classes, activation=self.activation_name)(layer)
        model = Model(inputs=inputs, outputs=predictions)
        return model

    def _get_model(self):
        """Compiles a model from provided architecture.
        """
        model = self._build_model()
        model.compile(optimizer=Adam(lr=self.learning_rate),
                      loss=self.loss,
                      metrics=self.metrics)
        return model

    def _create_convolution_block(self, input_layer, n_filters, batch_normalization=False,
                                  kernel=(3, 3), activation=None, padding='same',
                                  strides=(1, 1)):
        """Creates a convolutional block with two convolutional layers.
        """
        layer = Conv2D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
        if batch_normalization:
            layer = BatchNormalization(axis=-1)(layer)

        if activation is None:
            return Activation('relu')(layer)
        else:
            return activation()(layer)

    def _create_dense_block(self, input_layer, n_neurons, dropout_ratio=0.5):
        """Creates a dense block.
        """
        layer = Flatten()(input_layer)
        layer = Dense(n_neurons, activation='relu')(layer)
        return Dropout(dropout_ratio)(layer)

    def get_metrics(self):
        """Returns a set of metrics.
        """
        return self.model.metrics_names

    def get_parameters(self):
        """Returns a model parameters.
        """
        params = dict((key, value) for key, value in self.__dict__.items() \
                      if isinstance(value, (str, int, float, list, tuple)))
        return params

    def train(self, data, labels, batch_size=1):
        """Perform traning directly from data.
        """
        self.model = self._get_model()
        self.model.fit(data, labels, batch_size=batch_size, epochs=self.n_epochs,
                       verbose=self.verbose, validation_split=self.validation_split,
                       shuffle=True, callbacks=self._get_callbacks())

    def train_with_generator(self, train_generator, valid_generator,
                             steps_per_epoch_multiplier=1, verbose=1):
        """Perform training with generator.
        """
        self.model = self._get_model()
        self.model.fit_generator(train_generator, validation_data=valid_generator,
                                 steps_per_epoch=len(train_generator) * steps_per_epoch_multiplier,
                                 epochs=self.n_epochs, verbose=verbose,
                                 callbacks=self._get_callbacks())

    def predict(self, data, batch_size=1, verbose=1):
        """Perform prediction directly from data.
        """
        return self.model.predict(data, batch_size=batch_size, verbose=verbose)

    def predict_with_generator(self, predict_generator, verbose=1):
        """Perform prediction with generator.
        """
        return self.model.predict_generator(predict_generator, verbose=verbose)

    def evaluate_with_generator(self, evaluate_generator, verbose=1):
        """Perform evaluation with generator.
        """
        return self.model.evaluate_generator(evaluate_generator, verbose=verbose)


class DataGenerator(keras.utils.Sequence):
    """Generates batches of data of the specified size.
    """
    def __init__(self, image_paths, normalize=True, batch_size=8,
                 n_classes=2, shuffle=True, augmentation_params=None,
                 prediction=False, crop_size=(128, 128), struct_element_size=5,
                 min_area=256, interp=cv2.INTER_LINEAR,
                 processing_func=utils.process_image,
                 path_mapping={'bad': 0, 'good': 1}):
        self.image_paths = np.array(image_paths)
        self.normalize = normalize
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augmentation_params = augmentation_params
        self.path_mapping = path_mapping
        self.processing_func = processing_func
        self.process_kwargs = {
            'crop_size': crop_size,
            'struct_element_size': struct_element_size,
            'min_area': min_area,
            'interp': interp
        }

        # Get aunmenter if data augmentation is required
        self.augmenter = self._get_augmenter(self.augmentation_params) \
                            if self.augmentation_params is not None else None

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(self.image_paths.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def _get_augmenter(self, params):
        """Returns configured data augmentator.
        """
        seq = iaa.Sequential(random_order=True)

        # Add horizontal flipping
        if 'flip_prob' in params:
            seq.append(iaa.Fliplr(params['flip_prob']))

        # Add contrast normalization
        if 'cn' in params:
            seq.append(iaa.Sometimes(params['cn']['prob'],
                                     iaa.ContrastNormalization(params['cn']['params'])))

        # Add affine transformation
        if 'af' in params:
            seq.append(iaa.Sometimes(params['af']['prob'],
                                     iaa.Affine(**params['af']['params'])))
        return seq

    def _normalize_data(self, data):
        """Normalizes data to a range of [0, 1].
        """
        data /= 255.
        return data

    def __len__(self):
        return int(np.floor(self.image_paths.shape[0] / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Cyclically runs through the data if data augmentation is required
        if self.augmenter is not None:
            indices = np.array([np.mod(ii, self.image_paths.shape[0]-1) \
                                        for ii in indices])

        X, y = self._data_generation(indices)

        return X, y

    def _get_label_from_path(self, path):
        """Returns a label from a filepath.
        """
        for key in self.path_mapping.keys():
            if key in path:
                return self.path_mapping[key]
        return None

    def _data_generation(self, indices):
        """Generates a batch of data using provided indices.
        """
        if self.processing_func is not None:
            X = [self.processing_func(cv2.imread(self.image_paths[idx],
                                                 cv2.IMREAD_GRAYSCALE),
                                      **self.process_kwargs) \
                         for idx in indices]
        else:
            X = [cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE) \
                         for idx in indices]

        # Perform data augmentation
        if self.augmenter is not None:
            seq = self.augmenter.to_deterministic()
            X = [seq.augment_image(img) for img in X]

        # Prepare data before returning to a model
        X = np.asarray(X, dtype=np.float32)[..., np.newaxis]
        y = np.array([self._get_label_from_path(self.image_paths[idx]) \
                          for idx in indices], dtype=np.uint8)

        # Normalize data if required
        if self.normalize:
            X = self._normalize_data(X)

        return X, y
