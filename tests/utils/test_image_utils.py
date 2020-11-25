import pytest
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mock import patch

import calculon.utils.image_utils as i

def test_create_mask():
    pred_mask = tf.constant(
        [[[0.9, 0.1],
        [0.2, 0.8]],

        [[0.3, 0.7],
        [0.1, 0.9]]])
    a = i.create_mask(pred_mask)
    b = tf.constant([[0],[1]], dtype=tf.int64)
    tf.debugging.assert_equal(a, b)

@patch("matplotlib.pyplot.show")
def test_display(mock_show):
    img1 = np.zeros(shape=(128, 128, 1))
    img2 = np.zeros(shape=(128, 128, 1))
    img3 = np.zeros(shape=(128, 128, 1))
    display_list=[img1, img2, img3]
    i.display(display_list)
    plt.close()
    assert plt.gcf().number == 2 # get current figure