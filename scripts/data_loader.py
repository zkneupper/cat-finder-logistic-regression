#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Module docstring
"""


# Python standard library imports
import pathlib


# External library imports
import h5py
import numpy as np
import requests


def _load_h5_data_from_file(file_path):
    with h5py.File(file_path, "r") as dataset:
        data_dict = {}
        for key in dataset.keys():
            data_dict.update({key: np.array(dataset[key][:])})
    return data_dict


def load_data(file_path_train, file_path_test):
    """
    Docstring

    Example:
    >>>
    >>> import data_loader
    >>>
    >>> file_path_train = ...
    >>> file_path_test = ...
    >>>
    >>> (
    >>>     train_set_x,
    >>>     train_set_y,
    >>>     train_classes,
    >>>     test_set_x,
    >>>     test_set_y,
    >>>     test_classes,
    >>> ) = data_loader.load_data(file_path_train, file_path_test)
    >>>
    """

    data_dict_train = _load_h5_data_from_file(file_path_train)
    data_dict_test = _load_h5_data_from_file(file_path_test)

    train_set_x = data_dict_train["train_set_x"]
    train_set_y = data_dict_train["train_set_y"]

    train_classes = data_dict_train["list_classes"].astype(str)

    test_set_x = data_dict_test["test_set_x"]
    test_set_y = data_dict_test["test_set_y"]

    test_classes = data_dict_test["list_classes"].astype(str)

    return (
        train_set_x,
        train_set_y,
        train_classes,
        test_set_x,
        test_set_y,
        test_classes,
    )


def download_cat_image_data(file_path_data=None, verbose=1):
    """Download the cat image data from
    https://github.com/ridhimagarg/Cat-vs-Non-cat-Deep-learning-implementation/
    """
    url_base = "https://github.com/ridhimagarg/Cat-vs-Non-cat-Deep-learning-implementation/blob/master/datasets/"
    url_tail_train = "train_catvnoncat.h5?raw=true"
    url_tail_test = "test_catvnoncat.h5?raw=true"

    for url_tail in [url_tail_train, url_tail_test]:

        url = url_base + url_tail
        file_name = url.split("/")[-1]
        file_name = file_name.split("?")[0]

        file_path_target = pathlib.Path(file_path_data) / file_name
        if verbose:
            print(f"Downloading {file_name}...")

        r = requests.get(url)

        with open(file_path_target, "wb") as f:
            f.write(r.content)

    if verbose:
        print("Download completed")
