import os
import sys
import glob
import math
import ast
import errno
import configparser
import logging

import cv2
import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_interp_by_name(interp_name):
    """Get interpolation flag by name.
    """
    interp_name = interp_name.lower()
    if interp_name == 'nearest':
        return cv2.INTER_NEAREST
    elif interp_name == 'linear':
        return cv2.INTER_LINEAR
    elif interp_name == 'cubic':
        return cv2.INTER_CUBIC
    elif interp_name == 'lanczos':
        return cv2.INTER_LANCZOS4
    else:
        raise ValueError('There is no \'{}\' interpolation type.'.format(interp_name))


def process_image(img, crop_size=(128, 128), struct_element_size=5,
                  min_area=256, interp=cv2.INTER_LINEAR, find_bbox=False):
    """Pre-process images.

    Process image by localizing the object of interest, cropping it and resizing.
    """
    img_center = [v // 2 for v in img.shape]

    # Create structure element
    struct_element = cv2.cv2.getStructuringElement(cv2.MORPH_RECT, (struct_element_size,
                                                                    struct_element_size))
    # Perform grayscale morphology processing and segmentation
    img_filtered = cv2.dilate(img, struct_element, iterations=1)
    _, img_bin = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_OTSU)
    img_bin = cv2.morphologyEx(img_bin.astype(np.uint8),
                               cv2.MORPH_OPEN,
                               struct_element)

    # Label segmented structures and remove all having small area
    _, _, stats, centroids = cv2.connectedComponentsWithStats(img_bin)
    obj_idxs = [i for i, st in enumerate(stats) \
                    if st[cv2.CC_STAT_AREA] > min_area][1:]

    # Estimate Euclidean distance between image center and a structure centroid
    cdist = [np.linalg.norm(com - img_center) \
                    for com in centroids[obj_idxs]]

    # Obtain an index of structure which is closest to the image center
    obj_idx = obj_idxs[cdist.index(min(cdist))]

    # Determine the largest side of an object bounding box and centroid
    max_size = max([stats[obj_idx][cv2.CC_STAT_WIDTH],
                    stats[obj_idx][cv2.CC_STAT_HEIGHT]])
    nst_com = [int(v) for v in centroids[obj_idx]]

    # Calculate left and right side if the size is odd
    ls, rs = math.ceil(max_size / 2.), math.floor(max_size / 2.)

    # Crop and resize the target structure
    img_cropped = img[nst_com[1]-ls:nst_com[1]+rs+1, \
                      nst_com[0]-ls:nst_com[0]+rs+1]
    img_resized = cv2.resize(img_cropped, dsize=crop_size,
                             interpolation=interp)

    # Calculate the boudning box if required
    if find_bbox:
        obj_bbox = [stats[obj_idx][cv2.CC_STAT_LEFT],
                    stats[obj_idx][cv2.CC_STAT_TOP],
                    stats[obj_idx][cv2.CC_STAT_WIDTH],
                    stats[obj_idx][cv2.CC_STAT_HEIGHT]]
        return img_resized, obj_bbox
    else:
        return img_resized


def get_data_paths(path_pattern, shuffle=True, validation_split=0.3):
    """Obtain filepaths defined by the directory `path_pattern`.

    Obtain paths of images and splits them into traning and validation sets.
    """
    paths = glob.glob(path_pattern)
    paths = np.asarray(paths)
    if shuffle:
        np.random.shuffle(paths)
    split_idx = int(len(paths) * validation_split)
    return paths[split_idx:], paths[:split_idx]


def write_config_file(output_path, arch_params, proc_params):
    """Write configuration file for model architecture and pre-processing.
    """
    try:
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
    except IOError as ex:
        logger.error('Error: {}'.format(os.strerror(ex.errno)))
        sys.exit(1)

    config = configparser.ConfigParser()
    config['Network architecture'] = {k: v for k, v in arch_params.items()}
    config['Processing parameters'] = {k: v for k, v in proc_params.items()}

    try:
        with open(output_path, 'w') as conf_file:
            config.write(conf_file)
    except IOError as ex:
        logger.error('Error: {}'.format(os.strerror(ex.errno)))
        sys.exit(1)

    logger.info('Model configuration saved: {}'.format(output_path))


def read_config_file(config_path):
    """Read configuration file by `config_path` and returns them as `dict`.
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    arch_params = dict(config.items('Network architecture'))
    str_keys = set(['model_output_dir', 'csv_log_path', 'loss',
                    'model_params_path', 'net_name', 'optimizer',
                    'activation_name', 'model_path'])
    digit_keys = set(arch_params.keys()) - str_keys

    arch_strs = dict((k, arch_params[k]) for k in str_keys)
    arch_digits = dict((k, ast.literal_eval(arch_params[k])) for k in digit_keys)

    arch_params = arch_strs.copy()
    arch_params.update(arch_digits)

    proc_params = dict(config.items('Processing parameters'))
    proc_params = dict((k, ast.literal_eval(proc_params[k])) for k in proc_params.keys())
    return arch_params, proc_params


def get_image_with_url(url, timeout=5):
    """Obtain an image object of `numpy.ndarray` type by url.
    """
    def _get_error_json(ex, msg):
        return {
            'code': ex.response.status_code,
            'type': ex.response.reason,
            'message': msg
        }

    try:
        req = requests.get(url, timeout=timeout)
        req.raise_for_status()
    except requests.exceptions.HTTPError as ex:
        logger.error('HTTP Error')
        return _get_error_json(ex, 'HTTP Error')
    except requests.exceptions.ConnectionError as ex:
        logger.error('Connection Error')
        return _get_error_json(ex, 'Connection Error')
    except requests.exceptions.Timeout as ex:
        logger.error('Timeout Error')
        return _get_error_json(ex, 'Timeout Error')
    except requests.exceptions.RequestException as ex:
        logger.error('Unknown Error')
        return _get_error_json(ex, 'Unknown Error')

    return cv2.imdecode(np.frombuffer(req.content, dtype=np.uint8),
                        cv2.IMREAD_GRAYSCALE)


def predict_with_image_url(model, url, processing_kwargs=None, prob_threshold=0.5,
                           class_mapping={0: 'bent', 1: 'straight'}, norm_value=255.):
    """Perform prediction image class with provided `model` and image `url`.
    """
    res = get_image_with_url(url)

    if isinstance(res, dict):
        return res

    kw = {**processing_kwargs, 'find_bbox': True}
    img_proc, obj_bbox = process_image(res, **kw)

    obj_bbox = [v.item() for v in obj_bbox]
    img_proc = img_proc[np.newaxis, ..., np.newaxis]
    img_proc = img_proc.astype(np.float32)
    img_proc /= norm_value
    prob = np.squeeze(model.predict(img_proc, verbose=0)).tolist()

    return {
        'probabilities': prob,
        'class_name': ['bent', 'straight'],
        'bounding_box': {
            'x': obj_bbox[0],
            'y': obj_bbox[1],
            'width': obj_bbox[2],
            'height': obj_bbox[3]
        }
    }


def save_predictions(output_path, predictions, image_paths):
    """Saves the prediction results in a CSV file.
    """
    try:
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
    except IOError as ex:
        logger.error('Error: {}'.format(os.strerror(ex.errno)))
        sys.exit(1)

    df = pd.DataFrame({'paths': image_paths,
                       'probability class 0': predictions[:,0],
                       'probability class 1': predictions[:,1]})

    df.to_csv(output_path, index=False)

    logger.info('Predictions saved: {}'.format(output_path))
