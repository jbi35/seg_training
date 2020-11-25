import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt

class TFRecordsReader2D():
    '''Reads tfrecords and batches it if required.
    
    Returns:
        tf.dataset: Batched tf dataset.
    '''
    def __init__(self, dataset, buffer_size, skip_slices, batch_size, image_height, image_width, image_height_resize, 
                image_width_resize, num_classes):
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.skip_slices = skip_slices
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.image_height_resize = image_height_resize
        self.image_width_resize = image_width_resize
        self.num_classes = num_classes
        
    @tf.function
    def load_image_train(self, feature):
        '''Decodes the raw tf data to floats for images and segmentations, 
        it is therefore only usable for training, not testing purposes.
        
        Args:
            feature (tf.dtype): Example feature.
        
        Returns:
            float, float: image, segmentation
        '''
        # Decode the raw bytes to floats
        image = tf.io.decode_raw(feature['data/slice'], tf.float32)
        seg = tf.io.decode_raw(feature['data/seg'], tf.float32)

        # Reshape floats to original image sizes
        image = tf.reshape(image, shape=[self.image_height, self.image_width, 1])
        seg = tf.reshape(seg, shape=[self.image_height, self.image_width, 1])

        # Flip by 90 degrees
        image = tf.image.rot90(image, k=3, name=None)
        seg = tf.image.rot90(seg, k=3, name=None)

        # Resize images to fit input of Unet
        image = tf.image.resize(image, (self.image_height_resize, self.image_width_resize))
        seg = tf.image.resize(seg, (self.image_height_resize, self.image_width_resize))

        # One hot encoding
        #seg = tf.cast(seg, tf.int32)
        #seg = tf.squeeze(seg, axis=[2])
        #seg = tf.one_hot(seg, depth=self.num_classes)
        return image, seg

    def display(self, display_list):
        '''Displays a list of images.
        
        Args:
            display_list (array): Array of image tensors to display.
        '''
        plt.figure(figsize=(15, 15))

        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i], data_format='channels_last'))
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

    def read(self):
        features = {
            'data/slice': tf.io.FixedLenFeature([], tf.string),
            'data/seg': tf.io.FixedLenFeature([], tf.string),
        }
        
        parsed_dataset = self.dataset.map(lambda x: self._parse_function(x, features))
        dataset = parsed_dataset.map(self.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #test = parsed_dataset.map(self.load_image_test)

        # While validating (buffer_size is zero): data is not shuffled and not skipped
        if self.buffer_size==0:
            dataset_batched = dataset.batch(self.batch_size).repeat()
        # While training: data is shuffled
        elif self.skip_slices==0:
            dataset_batched = dataset.shuffle(self.buffer_size).batch(self.batch_size).repeat()
        # While training: slices may be skipped
        elif self.skip_slices>0:
            dataset_batched = dataset.skip(self.skip_slices).shuffle(self.buffer_size).batch(self.batch_size).repeat()
        else:
            raise ValueError("Skip_slices has invalid value.")

        # Prefetch elements from the input dataset ahead of the time they are requested
        dataset_batched_prefetched = dataset_batched.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
        #test_dataset = test.batch(self.batch_size)
        
        # Display sample images
        #for image, mask in train.take(1):
        #    sample_image, sample_mask = image, mask
        #    self.display([sample_image, sample_mask])

        return dataset_batched_prefetched
