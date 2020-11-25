import pytest
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mock import patch

from calculon.utils.read_tfrecords import TFRecordsReader

def test_init():
    a = TFRecordsReader('./', 10, 512, 512, 128, 128, 2, 10)
    assert isinstance(a, TFRecordsReader)

def test_parse_image_function():
    a = TFRecordsReader('./', 10, 512, 512, 128, 128, 2, 10)
    dataset = tf.data.TFRecordDataset('train.tfrecords')
    features = {
            'data/slice': tf.io.FixedLenFeature([], tf.string),
            'data/seg': tf.io.FixedLenFeature([], tf.string),
        }
    dataset = dataset.map(lambda x: a._parse_image_function(x, features))
    tf.debugging.assert_proper_iterable(dataset)

def test_read_tfrecords():
    a = TFRecordsReader('./', 10, 512, 512, 128, 128, 2, 10)
    dataset = a.read_tfrecords()
    tf.debugging.assert_proper_iterable(dataset)

@patch("matplotlib.pyplot.show")
def test_display(mock_show):
    a = TFRecordsReader('./', 10, 512, 512, 128, 128, 2, 10)
    img1 = np.zeros(shape=(512,512,1))
    img2 = np.zeros(shape=(512,512,1))
    img3 = np.zeros(shape=(512,512,1))
    display_list = [img1, img2, img3]
    assert a.display(display_list) == None

def test_get_batch():
    a = TFRecordsReader('./', 10, 512, 512, 128, 128, 2, 10)
    batch = a.get_batch()
    tf.debugging.assert_proper_iterable(batch)
