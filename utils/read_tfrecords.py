import os
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import clear_output

from calculon.utils.tfrecords_utils import *

class TFRecordsReader:
    '''Reads TFRecords for training and outputs batches of images and segmentations.
    '''
    def __init__(self, tfrecords_path, batch_size, image_height, image_width, image_height_resize, 
                image_width_resize, num_classes, buffer_size):
        '''Initialize TFRecordsReader
        
        Args:
            sess (tf.sess):             TF session
            tfrecords_path (str):       Path to the tfrecords directory
            batch_size (int):           Batch size
            image_height (int):         Image height of input in pixel
            image_width (int):          Image width of input in pixel
            image_height_resize (int):  New image height in pixel
            image_width_resize (int):   New image width in pixel
            num_classes (int):          Number of classes
            buffer_size (int):          Number of images to be loaded at once
        '''
        self.tfrecords_path = tfrecords_path
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.image_height_resize = image_height_resize
        self.image_width_resize = image_width_resize
        self.num_classes = num_classes
        self.buffer_size = buffer_size

    def _parse_image_function(self, example_proto, features):
        # Parse the input tf.Example proto using the dictionary above.
        feature = tf.io.parse_single_example(example_proto, features)
        img, seg = decode_tfrecords(feature, self.image_height, self.image_width)
        img = resize_image_tf(img, self.image_height_resize, self.image_width_resize)
        seg = resize_image_tf(seg, self.image_height_resize, self.image_width_resize)
        seg = onehot_labels_tf(seg, self.num_classes)
        return img, seg

    @tf.function
    def read_tfrecords(self):
        '''Reads the tfrecords files and creates an iterable dataset.
        
        Returns:
            tf.data.TFRecordDataset: The iterable tfrecord dataset.
        '''
        # Get the paths to the tfrecords files to be read
        filenames = get_tfrecord_filenames(self.tfrecords_path)
        raw_image_dataset = tf.data.TFRecordDataset(filenames)
        print("Reading: {}".format(filenames))

        # Create a dictionary describing the features
        features = {
            'data/slice': tf.io.FixedLenFeature([], tf.string),
            'data/seg': tf.io.FixedLenFeature([], tf.string),
        }

        dataset = raw_image_dataset.map(lambda x: self._parse_image_function(x, features))

        return dataset

    def display(self, display_list):
        plt.figure(figsize=(15, 15))

        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
        plt.show()

    def get_batch(self):
        '''Iterates through a TFRecordDataset and outputs batches of images and
        segmentations.
        
        Returns:
            dict: Dictionary of images and segmentations.
        '''
        # Get the dataset
        dataset = self.read_tfrecords()

        # Shuffle the data based on the buffer size
        dataset = dataset.shuffle(self.buffer_size)

        # Repeat the dataset number of times
        dataset = dataset.repeat(None)

        batch = dataset.cache().shuffle(self.buffer_size).batch(self.batch_size).repeat()
        batch = batch.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return batch