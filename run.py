#!/usr/bin/env python

import os
import argparse
import sys
import time
import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from model import utils, network


def run_training(args):
    """Perform a model traning by provided parameters.
    """
    if not len(glob.glob(args.images_dir)):
        raise ValueError('The directory \'{}\' is empty.'.format(args.images_dir))

    augmentation_params = {
        'flip_prob': args.aug_hf_probability,
        'cn': {
            'prob': args.aug_cn_probability,
            'params': args.aug_contrast_normalization
        },
        'af': {
            'prob': args.aug_af_probability,
            'params': {
                'scale': {
                    'x': args.aug_affine_scale,
                    'y': args.aug_affine_scale
                },
                'translate_percent': {
                    'x': args.aug_affine_translation,
                    'y': args.aug_affine_translation
                },
                'rotate': args.aug_affine_rotation
            }
        }
    }

    aug_params = augmentation_params if args.do_augmentation else None
    steps_per_epoch_mult = args.aug_steps_epoch_multiplier \
                                if args.do_augmentation else 1

    if not utils.USE_AUG:
        aug_params = None
        steps_per_epoch_mult = 1

    data_gen_kwargs = {
        'normalize': True,
        'batch_size': args.batch_size,
        'n_classes': args.num_classes,
        'shuffle': True,
        'augmentation_params': aug_params,
        'prediction': False,
        'crop_size': (args.input_size, args.input_size),
        'struct_element_size': args.size_structure_element,
        'min_area': args.obj_min_area,
        'interp': utils.get_interp_by_name(args.interpolation_type),
        'processing_func': utils.process_image,
        'path_mapping': {args.invalid_image_keyword: 0,
                         args.valid_image_keyword: 1}
    }

    train_paths, valid_paths = utils.get_data_paths(args.images_dir)
    train_gen, valid_gen = network.DataGenerator(train_paths, **data_gen_kwargs), \
                           network.DataGenerator(valid_paths, **data_gen_kwargs)

    net = network.CNN(input_shape=(args.input_size, args.input_size),
                      net_name=args.network_name,
                      n_dense_neurons=args.num_dense_neurons,
                      dropout_ratio=args.dropout_ratio,
                      n_epochs=args.num_epochs,
                      pool_size=(args.pool_size, args.pool_size),
                      model_output_dir=args.model_output_dir,
                      activation_name=args.activation_name,
                      depth=args.depth,
                      n_base_filters=args.num_base_filters,
                      n_classes=args.num_classes,
                      batch_normalization=args.batch_normalization,
                      min_receptive_field_size=args.min_receptive_field,
                      loss=args.loss_name,
                      metrics=[args.accuracy_name],
                      learning_rate=args.learning_rate,
                      optimizer=args.optimizer_name,
                      early_stopping_patience=args.early_stopping_patience,
                      learning_rate_patience=args.learning_rate_patience,
                      validation_split=args.validation_split)

    net.train_with_generator(train_gen,
                             valid_gen,
                             steps_per_epoch_multiplier=steps_per_epoch_mult,
                             verbose=args.verbosity)

    arch_params, proc_params = net.get_parameters(), train_gen.process_kwargs
    utils.write_config_file(args.model_params_output_path, arch_params, proc_params)


def run_prediction(args):
    """Perform prediction of a set of images in `image-dir`.
    """
    image_paths = glob.glob(args.images_dir)
    if not len(image_paths):
        raise ValueError('The directory \'{}\' is empty.'.format(args.images_dir))

    arch_params, proc_params = utils.read_config_file(args.model_params_path)
    data_gen_kwargs = {
        'normalize': True,
        'batch_size': args.batch_size,
        'n_classes': arch_params['n_classes'],
        'shuffle': True,
        'augmentation_params': None,
        'prediction': True,
        'crop_size': proc_params['crop_size'],
        'struct_element_size': proc_params['struct_element_size'],
        'min_area': proc_params['min_area'],
        'interp': proc_params['interp'],
        'processing_func': utils.process_image
    }

    net = network.CNN.model_from_kwargs(**arch_params)
    predict_gen = network.DataGenerator(image_paths, **data_gen_kwargs)
    outputs = net.predict_with_generator(predict_gen, verbose=args.verbosity)

    ts = time.strftime('%Y%m%d-%H%M%S')
    pred_output_path = os.path.join(args.output_dir,
                                    '-'.join(['predictions', ts]) + '.csv')
    utils.save_predictions(pred_output_path, outputs, image_paths)


class ModelCLI(object):
    """Multi-level CLI for ease usage.
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Model interaction CLI',
            usage = '''network <command> [<args>]

The commands are:

    train        Train a model on images in a specified directory
    predict      Predict class of images in a specified directory
                    ''')
        parser.add_argument('command', help='Print a list of available commands')

        # Parse only argument at a command position
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            logger.error('Unrecognized command')
            parser.print_help()
            exit(1)

        # Invoke a corresponding method
        getattr(self, args.command)()

    def train(self):
        parser = argparse.ArgumentParser(
            description='Train a model on images in a specified directory')

        parser.add_argument('--images-dir',
                            type=str,
                            help='Path to a directory with training images',
                            required=True)
        parser.add_argument('--network-name',
                            type=str,
                            help='Name of a network model',
                            required=True)
        parser.add_argument('--model-output-dir',
                            type=str,
                            help='Output directory of a network model',
                            required=True)
        parser.add_argument('--model-params-output-path',
                            type=str,
                            help='Output path of a network model',
                            required=True)
        parser.add_argument('--input-size',
                            type=int,
                            help='Size of input images (e.g. 128 means a size of 128x128,)',
                            default=128)
        parser.add_argument('--num-dense-neurons',
                            type=int,
                            help='Number of dense neurons on a fully-connected layer',
                            default=16)
        parser.add_argument('--dropout-ratio',
                            type=float,
                            help='Amount of dropout after a fully-connected layer',
                            default=0.5)
        parser.add_argument('--num-base-filters',
                            type=int,
                            help='Number of filters at the first convolutional layer',
                            default=8)
        parser.add_argument('--num-epochs',
                            type=int,
                            help='Number of training epochs',
                            default=50)
        parser.add_argument('--batch-size',
                            type=int,
                            help='Number of samples used per iteration',
                            default=8)
        parser.add_argument('--pool-size',
                            type=int,
                            help='Pooling size',
                            default=2)
        parser.add_argument('--depth',
                            type=int,
                            help='Depth of a network model',
                            default=3)
        parser.add_argument('--num-classes',
                            type=int,
                            help='Number of classes to predict',
                            default=2)
        parser.add_argument('--activation-name',
                            type=str,
                            help='Name of activation function',
                            default='sigmoid')
        parser.add_argument('--batch-normalization',
                            action='store_true',
                            help='Use batch normalization')
        parser.add_argument('--min-receptive-field',
                            help='Minimal size of receptive field',
                            default=None)
        parser.add_argument('--loss-name',
                            type=str,
                            help='Name of loss function',
                            default='binary_crossentropy')
        parser.add_argument('--accuracy-name',
                            type=str,
                            help='Name of accuracy function',
                            default='accuracy')
        parser.add_argument('--learning-rate',
                            type=float,
                            help='Learning rate of a network model',
                            default=0.0001)
        parser.add_argument('--optimizer-name',
                            type=str,
                            choices=['adam', 'sgd', 'rmsprop', 'nadam'],
                            help='Optimizer used for traning of a network model',
                            default='adam')
        parser.add_argument('--early-stopping-patience',
                            type=int,
                            help='Number of epochs with no improvement '
                            'after which training will be stopped',
                            default=15)
        parser.add_argument('--learning-rate-patience',
                            type=int,
                            help='Number of epochs with no improvement '
                            'after which learning rate will be reduced',
                            default=10)
        parser.add_argument('--validation-split',
                            type=float,
                            help='Amount of training data used for validation',
                            default=0.3)
        parser.add_argument('--size-structure-element',
                            type=int,
                            help='Size of a structural element for morphological '
                            'operations during image pre-processing',
                            default=5)
        parser.add_argument('--obj-min-area',
                            type=int,
                            help='Minimum area size of structures to '
                            'keep at an image',
                            default=256)
        parser.add_argument('--interpolation-type',
                            type=str,
                            choices=['Nearest', 'Linear', 'Cubic', 'Lanczos'],
                            help='Interpolation strategy of image resizing',
                            default='Linear')
        parser.add_argument('--do-augmentation',
                            action='store_true',
                            help='Do data augmentation')
        parser.add_argument('--aug-steps-epoch-multiplier',
                            type=int,
                            help='Number of times pass through the complete '
                                 'dataset during data augmentation',
                            default=10)
        parser.add_argument('--aug-hf-probability',
                            type=float,
                            help='Probability of horizontal flip during augmentation',
                            default=0.5)
        parser.add_argument('--aug-cn-probability',
                            type=float,
                            help='Probability of contrast normalization during augmentation',
                            default=0.5)
        parser.add_argument('--aug-contrast-normalization',
                            nargs='+',
                            type=float,
                            help='Contrast normalization range for data augmentation',
                            default=[0.75, 1.25])
        parser.add_argument('--aug-af-probability',
                            type=float,
                            help='Probability of affine transformation during augmentation',
                            default=0.5)
        parser.add_argument('--aug-affine-scale',
                            nargs='+',
                            type=float,
                            help='Image scaling range for data augmentation',
                            default=[0.75, 1.25])
        parser.add_argument('--aug-affine-translation',
                            nargs='+',
                            type=float,
                            help='Image translation range for data augmentation',
                            default=[-0.25, 0.25])
        parser.add_argument('--aug-affine-rotation',
                            nargs='+',
                            type=float,
                            help='Image rotation range in degrees for data augmentation',
                            default=[-15.0, 15.0])
        parser.add_argument('--valid-image-keyword',
                            type=str,
                            help='Keyword in a name of a valid image',
                            default='good')
        parser.add_argument('--invalid-image-keyword',
                            type=str,
                            help='Keyword in a name of a invalid image',
                            default='bad')
        parser.add_argument('--verbosity',
                            type=int,
                            help='Verbosity of a training process',
                            default=0)

        # Exclude a command and subcommand
        args = parser.parse_args(sys.argv[2:])
        run_training(args)

    def predict(self):
        parser = argparse.ArgumentParser(
            description='Predict class of images in a specified directory')

        parser.add_argument('--images-dir',
                            type=str,
                            help='Path to a directory with images for prediction',
                            required=True)
        parser.add_argument('--model-params-path',
                            type=str,
                            help='Path to a network model parameter file',
                            required=True)
        parser.add_argument('--output-dir',
                            type=str,
                            help='Output directory for prediction results',
                            required=True)
        parser.add_argument('--batch-size',
                            type=int,
                            help='Number of samples used per iteration',
                            default=1)
        parser.add_argument('--verbosity',
                            type=int,
                            help='Verbosity of a prediction process',
                            default=0)

        # Exclude a command and subcommand
        args = parser.parse_args(sys.argv[2:])
        run_prediction(args)

if __name__ == '__main__':
    ModelCLI()
