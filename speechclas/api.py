"""
API for speech to text

Date: September 2018
Author: Lara Lloret Iglesias
Email: lloret@ifca.unican.es
Github: laramaktub


Descriptions:
The API will use the model files inside ../models/api. If not found it will use the model files of the last trained model.
If several checkpoints are found inside ../models/api/ckpts we will use the last checkpoint.

Warnings:
There is an issue of using Flask with Keras: https://github.com/jrosebr1/simple-keras-rest-api/issues/1
The fix done (using tf.get_default_graph()) will probably not be valid for standalone wsgi container e.g. gunicorn,
gevent, uwsgi.
"""

import json
import os
from datetime import datetime
import pkg_resources
import builtins
import re
from collections import OrderedDict

import urllib.request
import requests
from tensorflow.keras import backend as K
from webargs import fields
from aiohttp.web import HTTPBadRequest

from speechclas import paths, utils, config, label_wav
from speechclas.data_utils import mount_nextcloud
from speechclas.train_runfile import train_fn


# Mount NextCloud folders (if NextCloud is available)
try:
    mount_nextcloud('ncplants:/data/dataset_files', paths.get_splits_dir())
    mount_nextcloud('ncplants:/data/images', paths.get_audio_dir())
    #mount_nextcloud('ncplants:/models', paths.get_models_dir())
except Exception as e:
    print(e)

# Empty model variables for inference (will be loaded the first time we perform inference)
loaded = False
graph, model, conf, class_names, class_info = None, None, None, None, None

# Additional parameters
allowed_extensions = set(['wav']) # allow only certain file extensions
top_K = 5  # number of top classes predictions to return


def load_inference_model():
    """
    Load a model for prediction.

    If several timestamps are available in `./models` it will load `.models/api` or the last timestamp if `api` is not
    available.
    If several checkpoints are available in `./models/[timestamp]/ckpts` it will load
    `.models/[timestamp]/ckpts/final_model.h5` or the last checkpoint if `final_model.h5` is not available.
    """
    global loaded, conf, MODEL_NAME, LABELS_FILE

    # Set the timestamp
    timestamps = next(os.walk(paths.get_models_dir()))[1]
    if not timestamps:
        raise Exception(
            "You have no models in your `./models` folder to be used for inference. "
            "This module does not come with a pretrained model so you have to train a model to use it for prediction.")
    else:
        if 'api' in timestamps:
            TIMESTAMP = 'api'
        else:
            TIMESTAMP = sorted(timestamps)[-1]
        paths.timestamp = TIMESTAMP
        print('Using TIMESTAMP={}'.format(TIMESTAMP))

        # Set the checkpoint model to use to make the prediction
        ckpts = os.listdir(paths.get_checkpoints_dir())
        if not ckpts:
            raise Exception(
                "You have no checkpoints in your `./models/{}/ckpts` folder to be used for inference. ".format(
                    TIMESTAMP) +
                "Therefore the API can only be used for training.")
        else:
            if 'model.pb' in ckpts:
                MODEL_NAME = 'model.pb'
            else:
                MODEL_NAME = sorted([name for name in ckpts if name.endswith('*.pb')])[-1]
            print('Using MODEL_NAME={}'.format(MODEL_NAME))

            if 'conv_labels.txt' in ckpts:
                LABELS_FILE = 'conv_labels.txt'
            else:
                LABELS_FILE = sorted([name for name in ckpts if name.endswith('*.txt')])[-1]
            print('Using LABELS_FILE={}'.format(LABELS_FILE))

            # Clear the previous loaded model
            K.clear_session()

            # Load the class names and info
            ckpts_dir = paths.get_checkpoints_dir()
            MODEL_NAME = os.path.join(ckpts_dir, MODEL_NAME )
            LABELS_FILE = os.path.join(ckpts_dir, LABELS_FILE )

            # Load training configuration
            conf_path = os.path.join(paths.get_conf_dir(), 'conf.json')
            with open(conf_path) as f:
                conf = json.load(f)

    # Set the model as loaded
    loaded = True


def update_with_query_conf(user_args):
    """
    Update the default YAML configuration with the user's input args from the API query
    """
    # Update the default conf with the user input
    CONF = config.CONF
    for group, val in sorted(CONF.items()):
        for g_key, g_val in sorted(val.items()):
            if g_key in user_args:
                g_val['value'] = json.loads(user_args[g_key])

    # Check and save the configuration
    config.check_conf(conf=CONF)
    config.conf_dict = config.get_conf_dict(conf=CONF)


def warm():
    if not loaded:
        load_inference_model()


def catch_error(f):
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)
    return wrap


def catch_url_error(url_list):

    url_list=url_list['urls']
    # Error catch: Empty query
    if not url_list:
        raise ValueError('Empty query')

    for i in url_list:
        # Error catch: Inexistent url
        try:
            url_type = requests.head(i).headers.get('content-type')
        except:
            raise ValueError("Failed url connection: "
                             "Check you wrote the url address correctly.")

        # Error catch: Wrong formatted urls
        if url_type != 'audio/x-wav':
            raise ValueError("Url wav format error: "
                             "Some urls were not in wav format.")


def catch_localfile_error(file_list):

    # Error catch: Empty query
    if not file_list[0].filename:
        raise ValueError('Empty query')


@catch_error
def predict(**args):

    if (not any([args['urls'], args['files']]) or
            all([args['urls'], args['files']])):
        raise Exception("You must provide either 'url' or 'data' in the payload")

    if args['files']:
        args['files'] = [args['files']]  # patch until list is available
        return predict_data(args)
    elif args['urls']:
        args['urls'] = [args['urls']]  # patch until list is available
        return predict_url(args)


def predict_url(args):
    """
    Function to predict an url
    """
    # # Check user configuration
    # update_with_query_conf(args)
    # conf = config.conf_dict

    catch_url_error(args['urls'])

    # Load model if needed
    if not loaded:
        load_inference_model()

    # Download the url
    urllib.request.urlretrieve(args['urls'][0], '/tmp/file.wav')
    pred_lab, pred_prob = label_wav.predict('/tmp/file.wav',
                                            LABELS_FILE,
                                            MODEL_NAME,
                                            "wav_data:0",
                                            "labels_softmax:0",
                                            3)

    return format_prediction(pred_lab, pred_prob)


def predict_data(args):
    """
    Function to predict an audio file
    """
    # # Check user configuration
    # update_with_query_conf(args)
    # conf = config.conf_dict

    if not loaded:
        load_inference_model()

    # Create a list with the path to the audios
    filenames = [f.filename for f in args['files']]

    pred_lab, pred_prob = label_wav.predict(filenames[0],
                                            LABELS_FILE,
                                            MODEL_NAME,
                                            "wav_data:0",
                                            "labels_softmax:0",
                                            3)

    return format_prediction(pred_lab, pred_prob)


def format_prediction(labels, probabilities):
    d = {
        "status": "ok",
         "predictions": [],
    }
    class_names = conf["model_settings"]["wanted_words"]
    for label_id, prob in zip(labels, probabilities):
        name = label_id

        pred = {
            "label": name,
            "probability": float(prob)
            }
        
        d["predictions"].append(pred)
    return d


def image_link(pred_lab):
    """
    Return link to Google images
    """
    base_url = 'https://www.google.es/search?'
    params = {'tbm':'isch','q':pred_lab}
    link = base_url + requests.compat.urlencode(params)
    return link


def wikipedia_link(pred_lab):
    """
    Return link to wikipedia webpage
    """
    base_url = 'https://en.wikipedia.org/wiki/'
    link = base_url + pred_lab.replace(' ', '_')
    return link


def train(**args):
    """
    Train an image classifier
    """
    # print('#####################')
    # raise Exception('error')
    update_with_query_conf(user_args=args)
    CONF = config.conf_dict
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    config.print_conf_table(CONF)
    K.clear_session() # remove the model loaded for prediction
    train_fn(TIMESTAMP=timestamp, CONF=CONF)
    
    # Sync with NextCloud folders (if NextCloud is available)
    try:
        mount_nextcloud(paths.get_models_dir(), 'ncplants:/models')
    except Exception as e:
        print(e)    


def populate_parser(parser, default_conf):
    """
    Returns a arg-parse like parser.
    """
    for group, val in default_conf.items():
        for g_key, g_val in val.items():
            gg_keys = g_val.keys()

            # Load optional keys
            help = g_val['help'] if ('help' in gg_keys) else ''
            type = getattr(builtins, g_val['type']) if ('type' in gg_keys) else None
            choices = g_val['choices'] if ('choices' in gg_keys) else None

            # Additional info in help string
            help += '\n' + "<font color='#C5576B'> Group name: **{}**".format(str(group))
            if choices:
                help += '\n' + "Choices: {}".format(str(choices))
            if type:
                help += '\n' + "Type: {}".format(g_val['type'])
            help += "</font>"

            # Create arg dict
            opt_args = {'missing': json.dumps(g_val['value']),
                        'description': help,
                        'required': False,
                        }
            if choices:
                opt_args['enum'] = [json.dumps(i) for i in choices]

            parser[g_key] = fields.Str(**opt_args)

    return parser


def get_train_args():

    parser = OrderedDict()
    default_conf = config.CONF
    default_conf = OrderedDict([('general', default_conf['general']),
                                ('model_settings', default_conf['model_settings']),
                                ('audio_processor', default_conf['audio_processor']),
                                ('training_parameters', default_conf['training_parameters'])])

    return populate_parser(parser, default_conf)


def get_predict_args():

    parser = OrderedDict()

    # Add data and url fields
    parser['files'] = fields.Field(required=False,
                                   missing=None,
                                   type="file",
                                   data_key="data",
                                   location="form",
                                   description="Select the image you want to classify.")

    # Use field.String instead of field.Url because I also want to allow uploading of base 64 encoded data strings
    parser['urls'] = fields.String(required=False,
                                   missing=None,
                                   description="Select an URL of the image you want to classify.")

    # missing action="append" --> append more than one url

    return parser


@catch_error
def get_metadata(distribution_name='speechclas'):
    """
    Function to read metadata
    """

    pkg = pkg_resources.get_distribution(distribution_name)
    meta = {
        'Name': None,
        'Version': None,
        'Summary': None,
        'Home-page': None,
        'Author': None,
        'Author-email': None,
        'License': None,
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        for par in meta:
            if line.startswith(par):
                _, value = line.split(": ", 1)
                meta[par] = value

    # Update information with Docker info (provided as 'CONTAINER_*' env variables)
    r = re.compile("^CONTAINER_(.*?)$")
    container_vars = list(filter(r.match, list(os.environ)))
    for var in container_vars:
        meta[var.capitalize()] = os.getenv(var)

    return meta
