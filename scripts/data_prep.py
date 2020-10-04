#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Functions for data prep, reverse data prep, and displaying data
"""


import time


# External library imports
import numpy as np
import matplotlib.pyplot as plt


# Data prep


def reshape_y_data(array):
    return array.reshape((1, array.shape[0]))


def flatten_x_data(array):
    return array.reshape(array.shape[0], -1).T


def standardize_image_pixels(array, max_pixel_value=255):
    return array / max_pixel_value


# Reverse data prep


def unflatten_record(array):
    assert array.shape == (12288,)
    return array.reshape(64, 64, 3)


def unstandardize(array, max_pixel_value=255):
    return ((array * max_pixel_value)).astype("int")


def show_unflatten_record(array, max_pixel_value=255):
    array_out = unflatten_record(array)
    array_out = unstandardize(array_out, max_pixel_value=max_pixel_value)
    plt.imshow(array_out)


# Show images


def show_record(record):
    ax = plt.imshow(record)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()
