import pytest
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mock import patch

from calculon.training.tfrecord_provider_3d import TFRecordsReader3D

def test_init():
    a = TFRecordsReader3D(tf.data.TFRecordDataset('train.tfrecords'), 1, 1, 2, 1, [64, 64, 64], False)
    assert isinstance(a, TFRecordsReader3D)

def test_load_image_train():
    a = TFRecordsReader3D(tf.data.TFRecordDataset('train.tfrecords'), 1, 1, 2, 1, [64, 64, 64], False)
    features = {
        'data/img': tf.io.FixedLenFeature([], tf.string),
        'data/seg': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_dataset = a.dataset.map(lambda x: a._parse_function(x, features))
    dataset = parsed_dataset.map(a.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tf.debugging.assert_proper_iterable(dataset)

def test_read():
    a = TFRecordsReader3D(tf.data.TFRecordDataset('train.tfrecords'), 1, 1, 2, 1, [64, 64, 64], False)
    dataset = a.read()
    tf.debugging.assert_proper_iterable(dataset)

@patch("matplotlib.pyplot.show")
def test_display(mock_show):
    a = TFRecordsReader3D(tf.data.TFRecordDataset('train.tfrecords'), 1, 1, 2, 1, [64, 64, 64], False)
    img1 = np.zeros(shape=(512,512,1))
    img2 = np.zeros(shape=(512,512,1))
    img3 = np.zeros(shape=(512,512,1))
    display_list = [img1, img2, img3]
    assert a._display(display_list) == None