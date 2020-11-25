import os, sys
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from skimage.transform import resize

import calculon.utils.log as log
from calculon.utils.validoptions import ValidOption
from calculon.utils.image_utils import create_mask, standardize_volume, read_nifti_file, read_dicom_series
from calculon.models.model_utils import dice_coef, dice_coef_loss

class Inference:
    ''' Class for the inference of segmenting lung CTs


    '''
    default_options = {
        'path_to_data':
            ValidOption(
                type=str,
                default='./',
                help='Path to data to be tested.'),
        'path_to_savedmodel':
            ValidOption(
                type=str,
                default='./',
                help='Path to pretrained tensorflow model.'),
        'image_shape_resize':
            ValidOption(
                type=list,
                subtype=int,
                default=[128, 128, 128],
                help='Resize image to this height when training'),
        'num_classes':
            ValidOption(
                type=int,
                default=2,
                help='Number of classes to predict into')
    }

    @classmethod
    def from_options(cls, options):
        ''' Create tester from options dictionary

        Args:
            options (dict): Dictionary with all necessary options

        Returns:
            Tester:   Tester object

        '''

        output_dir = options['output_dir']
        path_to_data = options['inference']['path_to_data']
        path_to_savedmodel = options['inference']['path_to_savedmodel']
        image_shape_resize = options['inference']['image_shape_resize']
        num_classes = options['inference']['num_classes']
        return cls(output_dir, path_to_data, path_to_savedmodel, 
                image_shape_resize, num_classes)

    def __init__(self, output_dir, path_to_data, path_to_savedmodel, 
                image_shape_resize, num_classes):
        '''Initialize the segmentation.
        
        Args:
            output_dir (str):               Path to output directory.
            path_to_data (str):             Path to lung CT image to be segmented.
            path_to_savedmodel (str):       Path to saved model directory.
            image_shape_resize (array):     Shape of image to be resized when segmenting.
            num_classes (int):              Number of classes to segment into. Binary segmentation: num_classes=2.
        '''

        self.output_dir = output_dir
        self.path_to_data = path_to_data
        self.path_to_savedmodel = path_to_savedmodel
        self.image_shape_resize = image_shape_resize
        self.num_classes = num_classes
    
    def _preprocess(self, image):
        '''Preprocess an image to be segmented.
        
        Args:
            image (array): Image array to be preprocessed.
        
        Returns:
            array: Preprocessed image
        '''
        # Resize to new shape
        image = resize(image, self.image_shape_resize, mode='constant', order=1, preserve_range=True, anti_aliasing=False) # bi-linear
        
        # Clip values to -1024 HU and 600 HU
        image = np.clip(image, a_min=-1024, a_max=600)

        # Standardize data with mean = 0 and std. = 1
        image = standardize_volume(image)

        # Expand dims
        image = tf.expand_dims(image, axis=0, name=None)
        image = tf.expand_dims(image, axis=-1, name=None)
        return image

    def _load_model(self):
        '''Recreates the same tf.model
        
        Returns:
            tf.model: A compiled tf.model
        '''
        model = tf.keras.models.load_model(self.path_to_savedmodel, custom_objects = { 'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef },compile=False)
        model.compile( loss = dice_coef_loss, optimizer = tf.keras.optimizers.Adam())
        return model
    
    def predict(self, image, model):
        '''Takes an image and a model and predicts the segmentation mask.
        
        Args:
            image (array):      Image to be predicted on.
            model (tf.model):   Tensorflow model to be predicted on.
        
        Returns:
            array: Mask array.
        '''
        # Get the original shape
        orig_shape = image.shape
            
        # Preprocess
        image = self._preprocess(image)

        # Predict
        pred_mask = model.predict(image)

        # Argmax over the classes and remove last dim
        mask = create_mask(pred_mask)
        mask = tf.squeeze(mask, axis=-1)

        # Transform to numpy array
        mask = tf.make_tensor_proto(mask)
        mask = tf.make_ndarray(mask)

        # Resize to original shape
        mask = resize(mask, output_shape=orig_shape, mode='constant', order=0, preserve_range=True, anti_aliasing=False) # order = 0: nearest neighbor
        return mask
    
    def _create_output_filename(self):
        '''Creates a output mask filename based on the CT basename, e.g. pat01_mask.nii.gz
        '''
        basename = os.path.basename(self.path_to_data)
        basename = os.path.splitext(basename)[0]
        outputfilename = os.path.join(self.output_dir, basename + '_mask.nii.gz')
        return outputfilename

    def write_image(self, image):
        '''Writes an image with SITK.
        
        Args:
            image (array):      Image array.
        '''
        outputImageFileName = self._create_output_filename()
        sitk_image = sitk.GetImageFromArray(image)
        sitk.WriteImage(sitk_image, outputImageFileName)
        print('Segmentation written to:', outputImageFileName)

    def execute(self):
        '''Execute the inference process.
        '''
        model = self._load_model()
        image = read_nifti_file(self.path_to_data)
        mask = self.predict(image, model)
        self.write_image(mask)