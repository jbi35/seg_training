import os
import tensorflow as tf

def get_tfrecord_filenames(path_to_tfrecord_dir):
    '''Lists all tfrecords file paths in a directory.
    
    Args:
        path_to_tfrecord_dir (str):    Path to the tfrecord directory
    
    Returns:
        array:                         Array of file paths to tfrecords
    '''
    file_paths =  [os.path.join(path_to_tfrecord_dir, x) for x in 
                os.listdir(path_to_tfrecord_dir) if x.endswith(".tfrecords")]
    return file_paths

def decode_tfrecords(feature, image_height, image_width):
    '''Parse the input tf.Example proto using the feature dictionary.
    
    Args:
        feature (dict):         The features of the TFRecordDataset.
        image_height (int):     Input image height.
        image_width (int):      Input image width.
    
    Returns:
        image, seg: Image and segmentaiton pairs.
    '''
    # Decode the raw bytes to floats
    image = tf.io.decode_raw(feature['data/slice'], tf.float32)
    seg = tf.io.decode_raw(feature['data/seg'], tf.float32)

    # Reshape floats to original image sizes
    image = tf.reshape(image, shape=[image_height, image_width, 1])
    seg = tf.reshape(seg, shape=[image_height, image_width, 1])
    return image, seg

def resize_image_tf(image, image_height_resize, image_width_resize):
    '''Resizes a tf image to a new shape.
    
    Args:
        image (tf.Tensor):          Image
        image_height_resize (int):  New image height.
        image_width_resize (int):   New image width.
    
    Returns:
        tf.Tensor:                  Resized image
    '''
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize(image, size=[image_height_resize, image_width_resize])
    image = tf.squeeze(image, axis=0)
    return image

def onehot_labels_tf(seg, num_classes):
    '''Changes the ground truth labels to one hot encoded labels.
    
    Args:
        seg (tf.Tensor):    Segmentation ground truth.
        num_classes (int):  Number of classes.
    
    Returns:
        tf.Tensor:          New segmentation ground truth.
    '''
    seg = tf.cast(seg, tf.int32)
    seg = tf.squeeze(seg, axis=[2])
    seg = tf.one_hot(seg, depth=num_classes)
    return seg