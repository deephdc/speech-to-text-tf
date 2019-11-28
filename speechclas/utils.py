"""
Miscellaneous utils

Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia
"""

import os

from speechclas import paths


def create_dir_tree():
    """
    Create directory tree structure
    """
    dirs = paths.get_dirs()
    for d in dirs.values():
        if not os.path.isdir(d):
            print('creating {}'.format(d))
            os.makedirs(d)


def remove_empty_dirs():
    basedir = paths.get_base_dir()
    dirs = os.listdir(basedir)
    for d in dirs:
        d_path = os.path.join(basedir, d)
        if not os.listdir(d_path):
            os.rmdir(d_path)
