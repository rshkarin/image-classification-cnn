import argparse
import sys
import logging

import tensorflow as tf
from flask import Flask, request, jsonify

from model import utils, network

graph = tf.get_default_graph()
logger = logging.getLogger(__name__)
app = Flask(__name__)

@app.route('/predict_gd')
def predict_gd():
    """Perform prediction by Google Drive ID.
    """
    if request.method == 'GET':
        google_base = 'https://drive.google.com/uc?export=view&id='
        image_id = request.args.get('image_id', '')
        image_url = google_base + image_id

        global graph
        with graph.as_default():
            resp = utils.predict_with_image_url(model, image_url,
                                                processing_kwargs=proc_params)
            return jsonify(resp)

@app.route('/predict_url')
def predict_url():
    """Perform prediction by URL.
    """
    if request.method == 'GET':
        image_url = request.args.get('image_url', '')

        global graph
        with graph.as_default():
            resp = utils.predict_with_image_url(model, image_url,
                                                processing_kwargs=proc_params)
            return jsonify(resp)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run the image prediction server with a pre-trained model')

    parser.add_argument('--port',
                        type=int,
                        help='Port to run server',
                        default=80)
    parser.add_argument('--host',
                        type=str,
                        help='Host to run server',
                        default='0.0.0.0')
    parser.add_argument('--model-params-path',
                        type=str,
                        help='Path to a network model parameter file',
                        required=True)
    parser.add_argument('--batch-size',
                        type=int,
                        help='Number of samples used per iteration',
                        default=1)
    parser.add_argument('--verbosity',
                        type=int,
                        help='Verbosity of a prediction process',
                        default=0)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()

    arch_params, proc_params = utils.read_config_file(args.model_params_path)
    model = network.CNN.model_from_kwargs(**arch_params)

    app.run(host=args.host, port=args.port, debug=True)
