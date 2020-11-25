import sys
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt

class TFRecordsReader3D():
    '''Reads tfrecords and batches it if required.
    
    Returns:
        tf.dataset: Batched tf dataset.
    '''
    def __init__(self, dataset, buffer_size, batch_size, num_classes, channels, image_shape_resize, validating):
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.channels = channels
        self.image_shape_resize = image_shape_resize
        self.output_shape = [self.image_shape_resize[0], self.image_shape_resize[1], self.image_shape_resize[2], self.channels]
        self.validating = validating

    @tf.function
    def load_image_train(self, feature):
        '''Decodes the raw tf data to floats for images and segmentations, 
        it is therefore only usable for training, not testing purposes.
        
        Args:
            feature (tf.dtype): Example feature.
        
        Returns:
            float, float: image, segmentation
        '''
        np.set_printoptions(threshold=sys.maxsize)
        
        # Decode the raw bytes to floats
        image = tf.io.decode_raw(feature['data/img'], tf.float32)
        seg = tf.io.decode_raw(feature['data/seg'], tf.float32)
        
        # Reshape floats to original image sizes
        image = tf.reshape(image, shape=self.output_shape)
        seg = tf.reshape(seg, shape=[self.image_shape_resize[0], self.image_shape_resize[1], self.image_shape_resize[2], self.channels])
        
        # Segmentation mask to 0s and 1s
        seg = tf.cast(seg, tf.int32)
        seg = tf.squeeze(seg, axis=-1)
        seg = tf.one_hot(seg, depth=self.num_classes, on_value=1, off_value=0)
        seg = tf.cast(seg, tf.float32)

        return image, seg

    def _display(self, display_list):
        '''Displays a list of images.
        
        Args:
            display_list (array): Array of image tensors to display.
        '''
        plt.figure(figsize=(10, 10))

        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i], data_format='channels_last'), cmap='gray')
            plt.axis('off')
        plt.show()
    
    def _parse_function(self, example_proto, features):
        '''Parse the input tf.Example proto using the dictionary above.
        
        Args:
            example_proto (tf.dtype):   Input proto.
            features (tf.dtype):        Features dictionary.
        
        Returns:
            tf.dtype: Parsed single example.
        '''
        feature = tf.io.parse_single_example(example_proto, features)
        return feature

    def _bytes_feature(self, value):
        '''Returns a bytes_list from a string / byte
        
        Args:
            value (bytes):  The value to be written as a bytes feature.
        
        Returns:
            bytes_list:     The tensorflow bytes list.
        '''
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def read(self):
        features = {
            'data/img': tf.io.FixedLenFeature([], tf.string),
            'data/seg': tf.io.FixedLenFeature([], tf.string),
        }
        
        parsed_dataset = self.dataset.map(lambda x: self._parse_function(x, features))
        dataset = parsed_dataset.map(self.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #test = parsed_dataset.map(self.load_image_test)

        # While validating: data is not shuffled and not skipped
        if self.validating:
            dataset_batched = dataset.batch(self.batch_size).repeat()
        # While training: data is shuffled
        else:
            dataset_batched = dataset.shuffle(self.buffer_size).batch(self.batch_size).repeat()
        
        # Display sample images
        #for image, mask in dataset_batched.take(1):
            # Get the middle slice
            #sample_image, sample_mask = image[0,64,:,:,0,None], mask[0,64,:,:,1,None]
            #self._display([sample_image, sample_mask])
        
        print('dataset_batched', dataset_batched)
        return dataset_batched
