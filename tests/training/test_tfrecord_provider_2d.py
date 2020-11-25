import pytest
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mock import patch

from calculon.training.tfrecord_provider_2d import TFRecordsReader2D

def test_init():
    a = TFRecordsReader2D(tf.data.TFRecordDataset('train.tfrecords'), 1, 0, 1, 512, 512, 128, 128, 1)
    assert isinstance(a, TFRecordsReader2D)

def test_load_image_train():
    a = TFRecordsReader2D(tf.data.TFRecordDataset('train.tfrecords'), 1, 0, 1, 512, 512, 128, 128, 1)
    features = {
            'data/slice': tf.io.FixedLenFeature([], tf.string),
            'data/seg': tf.io.FixedLenFeature([], tf.string),
        }
    parsed_dataset = a.dataset.map(lambda x: a._parse_function(x, features))
    dataset = parsed_dataset.map(a.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    tf.debugging.assert_proper_iterable(dataset)

@patch("matplotlib.pyplot.show")
def test_display(mock_show):
    a = TFRecordsReader2D(tf.data.TFRecordDataset('train.tfrecords'), 1, 0, 1, 512, 512, 128, 128, 1)
    img1 = np.zeros(shape=(512,512,1))
    img2 = np.zeros(shape=(512,512,1))
    img3 = np.zeros(shape=(512,512,1))
    display_list = [img1, img2, img3]
    assert a.display(display_list) == None

def test_parse_image_function():
    a = TFRecordsReader2D(tf.data.TFRecordDataset('train.tfrecords'), 1, 0, 1, 512, 512, 128, 128, 1)
    features = {
            'data/slice': tf.io.FixedLenFeature([], tf.string),
            'data/seg': tf.io.FixedLenFeature([], tf.string),
        }
    dataset = a.dataset.map(lambda x: a._parse_function(x, features))
    tf.debugging.assert_proper_iterable(dataset)

def test_read():
    a = TFRecordsReader2D(tf.data.TFRecordDataset('train.tfrecords'), 1, 0, 1, 512, 512, 128, 128, 1)
    dataset = a.read()
    tf.debugging.assert_proper_iterable(dataset)

def test_read_valueerror():
    a = TFRecordsReader2D(tf.data.TFRecordDataset('train.tfrecords'), 1, -1, 1, 512, 512, 128, 128, 1)
    with pytest.raises(ValueError):
        assert a.read()