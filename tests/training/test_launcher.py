import pytest
import tensorflow as tf
import shutil
import tempfile
import mock

from calculon.training.launcher import Launcher
from calculon.training.launcher import DisplayCallback

def test_init():
    l = Launcher(".", "./", ["any"], ["any"], ["lungs"], 1, "nifti", "nifti", "nii", "./", "./", False, "axial", 0, "./", "./", 1, 1, [64, 64, 64], 2, 1, False, "./", "adam", 0.001, 0.95, 10000)
    assert isinstance(l, Launcher)

def test_init_display():
    file_writer = tf.summary.create_file_writer('./img')
    d = DisplayCallback(tf.keras.Model, tf.data.TFRecordDataset, file_writer)
    assert isinstance(d, DisplayCallback)