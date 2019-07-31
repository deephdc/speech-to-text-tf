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
import tempfile
import warnings
from datetime import datetime
import pkg_resources
import builtins
import re
import urllib.request

import numpy as np
import requests
from werkzeug.exceptions import BadRequest
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from speechclas import paths, utils, config, label_wav
from speechclas.data_utils import load_class_names, load_class_info, mount_nextcloud
from speechclas.test_utils import predict
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
        raise BadRequest(
            """You have no models in your `./models` folder to be used for inference.
            Therefore the API can only be used for training.""")
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
            raise BadRequest(
                """You have no checkpoints in your `./models/{}/ckpts` folder to be used for inference.
                Therefore the API can only be used for training.""".format(TIMESTAMP))
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
            MODEL_NAME=  os.path.join(ckpts_dir, MODEL_NAME )
            LABELS_FILE=  os.path.join(ckpts_dir, LABELS_FILE )
            

            # Load training configuration
            conf_path = os.path.join(paths.get_conf_dir(), 'conf.json')
            with open(conf_path) as f:
                conf = json.load(f)


    # Set the model as loaded
    loaded = True


def catch_error(f):
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise e
    return wrap


def catch_url_error(url_list):

    url_list=url_list['urls']
    # Error catch: Empty query
    if not url_list:
        raise BadRequest('Empty query')

    for i in url_list:
        # Error catch: Inexistent url
        try:
            url_type = requests.head(i).headers.get('content-type')
        except:
            raise BadRequest("""Failed url connection:
            Check you wrote the url address correctly.""")

        # Error catch: Wrong formatted urls
        if url_type != 'audio/x-wav':
            raise BadRequest("""Url wav format error:
            Some urls were not in wav format.""")


def catch_localfile_error(file_list):

    # Error catch: Empty query
    if not file_list[0].filename:
        raise BadRequest('Empty query')

    # Error catch: Image format error
    for f in file_list:
        extension = f.split('.')[-1]
        if extension not in allowed_extensions:
            raise BadRequest("""Local image format error:
            At least one file is not in a standard image format (jpg|jpeg|png).""")


@catch_error
def predict_url(urls, merge=True):
    """
    Function to predict an url
    """
    catch_url_error(urls)

    if not loaded:
        load_inference_model()
    urllib.request.urlretrieve(urls['urls'][0], '/tmp/file.wav')
    pred_lab, pred_prob =label_wav.predict('/tmp/file.wav', LABELS_FILE, MODEL_NAME, "wav_data:0","labels_softmax:0", 3)
    return format_prediction(pred_lab, pred_prob)



@catch_error
def predict_file(filenames, merge=True):
    """
    Function to predict a local image
    """
    catch_localfile_error(filenames)

    if not loaded:
        load_inference_model()
    with graph.as_default():
        pred_lab, pred_prob = predict(model=model,
                                      X=filenames,
                                      conf=conf,
                                      top_K=top_K,
                                      filemode='local',
                                      merge=merge)

    if merge:
        pred_lab, pred_prob = np.squeeze(pred_lab), np.squeeze(pred_prob)

    return format_prediction(pred_lab, pred_prob)


@catch_error
def predict_data(audios, merge=True):
    """
    Function to predict an audio file
    """
    if not loaded:
        load_inference_model()
    if not isinstance(audios, list):
        audios = [audios]
    filenames = []
    for audio in audios:

        thename=audio['files'].filename
        thefile="/tmp/"+thename
        audio['files'].save(thefile)

    pred_lab, pred_prob =label_wav.predict(thefile, LABELS_FILE, MODEL_NAME, "wav_data:0","labels_softmax:0", 3)
    return format_prediction(pred_lab, pred_prob)



def format_prediction(labels, probabilities):
    d = {
        "status": "ok",
         "predictions": [],
    }
    class_names=conf["model_settings"]["wanted_words"]
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


def metadata():
    d = {
        "author": None,
        "description": None,
        "url": None,
        "license": None,
        "version": None,
    }
    return d


@catch_error
def train(user_conf):
    """
    Parameters
    ----------
    user_conf : dict
        Json dict (created with json.dumps) with the user's configuration parameters that will replace the defaults.
        Must be loaded with json.loads()
        For example:
            user_conf={'num_classes': 'null', 'lr_step_decay': '0.1', 'lr_step_schedule': '[0.7, 0.9]', 'use_early_stopping': 'false'}
    """
    CONF = config.CONF

    # Update the conf with the user input
    for group, val in sorted(CONF.items()):
        for g_key, g_val in sorted(val.items()):
            g_val['value'] = json.loads(user_conf[g_key])

    # Check the configuration
    try:
        config.check_conf(conf=CONF)
    except Exception as e:
        raise BadRequest(e)

    CONF = config.conf_dict(conf=CONF)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')

    config.print_conf_table(CONF)
    K.clear_session() # remove the model loaded for prediction
    train_fn(TIMESTAMP=timestamp, CONF=CONF)
    
    # Sync with NextCloud folders (if NextCloud is available)
    try:
        mount_nextcloud(paths.get_models_dir(), 'ncplants:/models')
    except Exception as e:
        print(e)    


@catch_error
def get_train_args():
    """
    Returns a dict of dicts with the following structure to feed the deepaas API parser:
    { 'arg1' : {'default': '1',     #value must be a string (use json.dumps to convert Python objects)
                'help': '',         #can be an empty string
                'required': False   #bool
                },
      'arg2' : {...
                },
    ...
    }
    """
    train_args = {}
    default_conf = config.CONF
    for group, val in default_conf.items():
        for g_key, g_val in val.items():
            gg_keys = g_val.keys()

            # Load optional keys
            help = g_val['help'] if ('help' in gg_keys) else ''
            type = getattr(builtins, g_val['type']) if ('type' in gg_keys) else None
            choices = g_val['choices'] if ('choices' in gg_keys) else None

            # Additional info in help string
            help += '\n' + "Group name: **{}**".format(str(group))
            if choices: help += '\n' + "Choices: {}".format(str(choices))
            if type: help += '\n' + "Type: {}".format(g_val['type'])

            opt_args = {'default': json.dumps(g_val['value']),
                        'help': help,
                        'required': False}
            # if type: opt_args['type'] = type # this breaks the submission because the json-dumping
            #                                     => I'll type-check args inside the train_fn

            train_args[g_key] = opt_args
    return train_args


@catch_error
def get_metadata():
    """
    Function to read metadata
    """

    module = __name__.split('.', 1)

    pkg = pkg_resources.get_distribution(module[0])
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
