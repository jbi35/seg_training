import os
import re
import sys
import math
import numpy as np
import tensorflow as tf
from skimage.transform import resize
import calculon.utils.log as log
from calculon.utils.validoptions import ValidOption
from calculon.utils.image_utils import create_mask, filter_volume, get_slice, make_tfrecords_ready, make_tfrecords_ready_3d, get_n_slices, get_axis_from_view, read_nifti_file, compute_mean_variance_file, normalize_volume, two_display, standardize_volume


class Writer:
    ''' Class for writing lung CTs into Tensorflow records


    '''
    def __init__(self, output_dir, file_type, path_to_vol_data, path_to_seg_data, filter, view, input_height, input_width):
        ''' Initialize a segmentation coordinator

        Args:
            file_type (str):            File type of CT image (nifti, dicom)
            path_to_vol_data (str):     Path to data of chest CT image (volumes)
            path_to_seg_data (str):     Path to data of chest CT image (segmentations)
            filter (bool):              Indicates whether to only include images with labels
            view (str):                 The view from which to slice the 3d image
            input_height (int):         Input image height in pixel
            input_width (int):          Input image width in pixel
        '''
        self.output_dir = output_dir
        self.file_type = file_type
        self.filter = filter
        self.view = view
        self.axis = get_axis_from_view(self.view)
        self.input_height = input_height
        self.input_width = input_width
        self.vol_paths = path_to_vol_data
        self.seg_paths = path_to_seg_data
        if self.vol_paths != []:
            self.train_range = range(0, math.ceil(len(self.vol_paths) * 0.6))
            self.val_range = range(self.train_range[-1] + 1, len(self.vol_paths))
        self.path_to_tfrecords_train = os.path.join(self.output_dir, 'tfrecords/train/')
        self.path_to_tfrecords_val = os.path.join(self.output_dir, 'tfrecords/val/')

    def _bytes_feature(self, value):
        '''Returns a bytes_list from a string / byte
        
        Args:
            value (bytes):  The value to be written as a bytes feature.
        
        Returns:
            bytes_list:     The tensorflow bytes list.
        '''
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def save_data(self):
        '''Saves CT scans and segmentation masks to separate training and 
        validation tfrecords files.
        
        Raises:
            ValueError: If separation into training and validation sets has failed.
        '''
        print("Write to train.tfrecords: ", [self.vol_paths[x] for x in self.train_range])
        print("Write to val.tfrecords: ", [self.vol_paths[x] for x in self.val_range])
        for i in [self.path_to_tfrecords_train, self.path_to_tfrecords_val]:
            if not os.path.exists(i):
                os.makedirs(i)

        train_writer = tf.io.TFRecordWriter(self.path_to_tfrecords_train + 'train.tfrecords')
        val_writer = tf.io.TFRecordWriter(self.path_to_tfrecords_val + 'val.tfrecords')

        # Cycle through the CT and segmentation pairs
        for i in range(len(self.vol_paths)):
            print('Writing CT Scan: {}/{}'.format(i+1, len(self.vol_paths)))
            sys.stdout.flush()
            print(self.vol_paths[i], self.seg_paths[i])

            # Read nifti file
            vol_data = read_nifti_file(self.vol_paths[i])
            seg_data = read_nifti_file(self.seg_paths[i])

            # Clip values to -1024 HU and 600 HU
            vol_data = np.clip(vol_data, a_min=-1024, a_max=600)

            # Normalize the data
            vol_data = standardize_volume(vol_data)
            print(np.amax(vol_data), np.amin(vol_data), np.mean(vol_data), np.std(vol_data))

            # Filter out only slices that contain labels
            if self.filter:
                vol_data, seg_data = filter_volume(vol_data, seg_data, self.axis)

            # Get the number of slices
            n_slices = get_n_slices(self.view, vol_data)

            # Cycle through the image slices
            n_images = 0
            for j in range(n_slices):
                vol_slice = get_slice(self.view, vol_data, j)
                seg_slice = get_slice(self.view, seg_data, j)

                # Resize the slice if necessary
                h, w = np.shape(seg_slice)
                if h != self.input_height or w != self.input_width:
                    vol_slice = resize(vol_slice, output_shape=(self.input_height, self.input_width), order=1, mode='constant', anti_aliasing=False, preserve_range=True)
                    seg_slice = resize(seg_slice, output_shape=(self.input_height, self.input_width), order=0, mode='constant', anti_aliasing=False, preserve_range=True)

                # Expand dims and change datatype to string to be saved by TF
                vol_slice = make_tfrecords_ready(vol_slice)
                seg_slice = make_tfrecords_ready(seg_slice)

                # Create a feature
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature={
                    # Wrap the data as TensorFlow features
                    'data/slice': self._bytes_feature(vol_slice),
                    'data/seg': self._bytes_feature(seg_slice)}))

                # Serialize to string and write to the TFRecords file
                if i in self.train_range:
                    train_writer.write(example.SerializeToString())
                elif i in self.val_range:
                    val_writer.write(example.SerializeToString())
                else:
                    raise ValueError("Not in training or validation range.")
                n_images += 1
            print('Number of slices: {}'.format(n_images))
        train_writer.close()
        val_writer.close()
        sys.stdout.flush()
        log.info('YELLOW', 'Saved as TFRecords')
        return self.path_to_tfrecords_train, self.path_to_tfrecords_val


class Writer3D:
    ''' Class for writing lung CTs as 3D tensors into Tensorflow records


    '''
    def __init__(self, output_dir, file_type, path_to_vol_data, path_to_seg_data, filter, view, image_shape_resize):
        self.output_dir = output_dir
        self.file_type = file_type
        self.filter = filter
        self.view = view
        self.axis = get_axis_from_view(self.view)
        self.image_shape_resize = image_shape_resize
        self.vol_paths = path_to_vol_data
        self.seg_paths = path_to_seg_data
        self.path_to_tfrecords_train = os.path.join(self.output_dir, 'tfrecords/train/')
        self.path_to_tfrecords_val = os.path.join(self.output_dir, 'tfrecords/val/')
        if self.vol_paths != []:
            self.train_range = range(0, math.ceil(len(self.vol_paths) * 0.6))
            self.val_range = range(self.train_range[-1] + 1, len(self.vol_paths))

    def _bytes_feature(self, value):
        '''Returns a bytes_list from a string / byte
        
        Args:
            value (bytes):  The value to be written as a bytes feature.
        
        Returns:
            bytes_list:     The tensorflow bytes list.
        '''
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def save_data(self):
        '''Saves CT scans and segmentation masks to separate training and 
        validation tfrecords files.
        
        Raises:
            ValueError: If separation into training and validation sets has failed.
        '''
        print("Write to train.tfrecords: ", [self.vol_paths[x] for x in self.train_range])
        print("Write to val.tfrecords: ", [self.vol_paths[x] for x in self.val_range])
        for i in [self.path_to_tfrecords_train, self.path_to_tfrecords_val]:
            if not os.path.exists(i):
                os.makedirs(i)

        train_writer = tf.io.TFRecordWriter(self.path_to_tfrecords_train + 'train.tfrecords')
        val_writer = tf.io.TFRecordWriter(self.path_to_tfrecords_val + 'val.tfrecords')

        # Cycle through the CT and segmentation pairs
        for i in range(len(self.vol_paths)):
            print('Writing CT Scan: {}/{}'.format(i+1, len(self.vol_paths)))
            sys.stdout.flush()
            print(self.vol_paths[i], self.seg_paths[i])

            # Read nifti file
            vol_data = read_nifti_file(self.vol_paths[i])
            seg_data = read_nifti_file(self.seg_paths[i])
            print('Original shape: ', vol_data.shape, seg_data.shape)

            # Clip values to -1024 HU and 600 HU
            vol_data = np.clip(vol_data, a_min=-1024, a_max=600)
            # Normalize data between [0,1]
            #vol_data = normalize_volume(vol_data)
            print('Classes: ', np.unique(seg_data))

            # Resize to new shape
            vol_data = resize(vol_data, self.image_shape_resize, mode='constant', order=1, preserve_range=True, anti_aliasing=False) # bi-linear
            seg_data = resize(seg_data, self.image_shape_resize, mode='constant', order=0, preserve_range=True, anti_aliasing=False) # nearest neighbor
            print('Resized to shape: ', vol_data.shape, seg_data.shape)
            # Seg back to integers
            #seg_data = np.round(seg_data)
            print('Classes: ', np.unique(seg_data))
            
            #two_display([vol_data[None,64,:,:], seg_data[None,64,:,:]])
            
            # Standardize the data
            #self.mean, self.std = compute_mean_variance_file(vol_data)
            vol_data = standardize_volume(vol_data)
            print("After Normalizing: Max {}, Min {}, Mean {}, Std {}".format(np.amax(vol_data), np.amin(vol_data), np.mean(vol_data), np.std(vol_data)))

            # Expand dims and change datatype to string to be saved by TF
            vol_data = make_tfrecords_ready_3d(vol_data)
            seg_data = make_tfrecords_ready_3d(seg_data)
            # Create a feature
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature={
                # Wrap the data as TensorFlow features
                'data/img': self._bytes_feature(vol_data),
                'data/seg': self._bytes_feature(seg_data)}))

            # Serialize to string and write to the TFRecords file
            if i in self.train_range:
                train_writer.write(example.SerializeToString())
            elif i in self.val_range:
                val_writer.write(example.SerializeToString())
            else:
                raise ValueError("Not in training or validation range.")

        train_writer.close()
        val_writer.close()
        sys.stdout.flush()
        log.info('YELLOW', 'Saved as TFRecords')
        return self.path_to_tfrecords_train, self.path_to_tfrecords_val


class Writer3DSlabs:
    ''' Class for writing lung CT slabs as 3D tensors into Tensorflow records.
    A slab represent a thick axial slice of a full CT scan. Default size is 
    [32, 256, 256].

    '''
    def __init__(self, output_dir, file_type, path_to_vol_data, path_to_seg_data, filter, view, image_shape_resize):
        self.output_dir = output_dir
        self.file_type = file_type
        self.filter = filter
        self.view = view
        self.axis = get_axis_from_view(self.view)
        self.image_shape_resize = image_shape_resize
        self.vol_paths = path_to_vol_data
        self.seg_paths = path_to_seg_data
        self.path_to_tfrecords_train = os.path.join(self.output_dir, 'tfrecords/train/')
        self.path_to_tfrecords_val = os.path.join(self.output_dir, 'tfrecords/val/')
        if self.vol_paths != []:
            self.train_range = range(0, math.ceil(len(self.vol_paths) * 0.6))
            self.val_range = range(self.train_range[-1] + 1, len(self.vol_paths))

    def _bytes_feature(self, value):
        '''Returns a bytes_list from a string / byte
        
        Args:
            value (bytes):  The value to be written as a bytes feature.
        
        Returns:
            bytes_list:     The tensorflow bytes list.
        '''
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def save_data(self):
        '''Saves CT scans and segmentation masks to separate training and 
        validation tfrecords files.
        
        Raises:
            ValueError: If separation into training and validation sets has failed.
        '''
        print("Write to train.tfrecords: ", [self.vol_paths[x] for x in self.train_range])
        print("Write to val.tfrecords: ", [self.vol_paths[x] for x in self.val_range])
        for i in [self.path_to_tfrecords_train, self.path_to_tfrecords_val]:
            if not os.path.exists(i):
                os.makedirs(i)

        train_writer = tf.io.TFRecordWriter(self.path_to_tfrecords_train + 'train.tfrecords')
        val_writer = tf.io.TFRecordWriter(self.path_to_tfrecords_val + 'val.tfrecords')

        # Cycle through the CT and segmentation pairs
        for i in range(len(self.vol_paths)):
            print('Writing CT Scan: {}/{}'.format(i+1, len(self.vol_paths)))
            sys.stdout.flush()
            print(self.vol_paths[i], self.seg_paths[i])

            # Read nifti file
            vol_data = read_nifti_file(self.vol_paths[i])
            seg_data = read_nifti_file(self.seg_paths[i])
            print('Original shape: ', vol_data.shape, seg_data.shape)

            # Resize into smaller axial shape, keep original depth dimension
            vol_data = resize(vol_data, [vol_data.shape[0], self.image_shape_resize[1], self.image_shape_resize[2]])
            seg_data = resize(seg_data, [seg_data.shape[0], self.image_shape_resize[1], self.image_shape_resize[2]])
            print('Resized to shape: ', vol_data.shape, seg_data.shape)
            
            # Clip values to -1024 HU and 600 HU
            vol_data = np.clip(vol_data, a_min=-1024, a_max=600)

            # Normalize the data
            vol_data = standardize_volume(vol_data)
            print(np.amax(vol_data), np.amin(vol_data), np.mean(vol_data), np.std(vol_data))
            
            # Calculate number of axial slabs s
            s = math.floor(vol_data.shape[0] / self.image_shape_resize[0])
            print('CT scan will be split into {} slabs of size {}.'.format(s, self.image_shape_resize))
            print(range(s))
            # Split into slabs
            for j in range(s):
                start = self.image_shape_resize[0] * j
                end = self.image_shape_resize[0] * (j + 1)
                print(start, end)

                vol_slab = vol_data[start : end, :, :]
                seg_slab = seg_data[start : end, :, :]

                # Expand dims and change datatype to string to be saved by TF
                vol_slab = make_tfrecords_ready_3d(vol_slab)
                seg_slab = make_tfrecords_ready_3d(seg_slab)
                # Create a feature
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature={
                    # Wrap the data as TensorFlow features
                    'data/img': self._bytes_feature(vol_slab),
                    'data/seg': self._bytes_feature(seg_slab)}))

                # Serialize to string and write to the TFRecords file
                if i in self.train_range:
                    train_writer.write(example.SerializeToString())
                elif i in self.val_range:
                    val_writer.write(example.SerializeToString())
                else:
                    raise ValueError("Not in training or validation range.")

        train_writer.close()
        val_writer.close()
        sys.stdout.flush()
        log.info('YELLOW', 'Saved as TFRecords')
        return self.path_to_tfrecords_train, self.path_to_tfrecords_val


class Writer3DSlabsConcat:
    '''A class for using a pretrained (low-res) net and predicting (low-res) output,
    then adding new (high-res) input  and concatenating it to the predictions to 
    write new tfrecords for another training round. The new (high-res) input is therefore
    split into slabs. A slab represent a thick axial slice of a full CT scan. Default size is 
    [32, 256, 256].
    
    Raises:
        ValueError: When the slabs are not separated clearly.
    
    Returns:
        str, str, int, int: Path to tfrecords for volumes and segmentations, number of volumes
        and number of segmentations.
    '''
    def __init__(self, model, output_dir, path_to_vol_data, path_to_seg_data):
        self.output_dir = output_dir
        self.model = model
        self.image_shape_lowres = [64, 64, 64]
        self.image_shape_highres = [32, 256, 256]
        self.vol_paths = path_to_vol_data
        self.seg_paths = path_to_seg_data
        self.path_to_tfrecords_train = os.path.join(self.output_dir, 'tfrecords/train/')
        self.path_to_tfrecords_val = os.path.join(self.output_dir, 'tfrecords/val/')
        self.n_train = 0
        self.n_val = 0
        if self.vol_paths != []:
            self.train_range = range(0, math.ceil(len(self.vol_paths) * 0.6))
            self.val_range = range(self.train_range[-1] + 1, len(self.vol_paths))
    
    def _bytes_feature(self, value):
        '''Returns a bytes_list from a string / byte
        
        Args:
            value (bytes):  The value to be written as a bytes feature.
        
        Returns:
            bytes_list:     The tensorflow bytes list.
        '''
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def save_data(self):
        '''Read CT scans and images
        '''
        print("Write to train.tfrecords: ", [self.vol_paths[x] for x in self.train_range])
        print("Write to val.tfrecords: ", [self.vol_paths[x] for x in self.val_range])
        self.path_to_tfrecords_train = os.path.join(self.path_to_tfrecords_train, 'concat/')
        self.path_to_tfrecords_val = os.path.join(self.path_to_tfrecords_val, 'concat/')
        for i in [self.path_to_tfrecords_train, self.path_to_tfrecords_val]:
            if not os.path.exists(i):
                os.makedirs(i)

        train_writer = tf.io.TFRecordWriter(self.path_to_tfrecords_train + 'concat_train.tfrecords')
        val_writer = tf.io.TFRecordWriter(self.path_to_tfrecords_val + 'concat_val.tfrecords')

        # Cycle through the CT and segmentation pairs
        for i in range(len(self.vol_paths)):
            print('Writing CT Scan: {}/{}'.format(i+1, len(self.vol_paths)))
            sys.stdout.flush()
            print(self.vol_paths[i], self.seg_paths[i])

            # Read nifti file
            vol_data = read_nifti_file(self.vol_paths[i])
            seg_data = read_nifti_file(self.seg_paths[i])
            print('Original shape: ', vol_data.shape, seg_data.shape)

            # Clip values to -1024 HU and 600 HU
            vol_data = np.clip(vol_data, a_min=-1024, a_max=600)

            # Normalize the data
            vol_data = standardize_volume(vol_data)
            print(np.amax(vol_data), np.amin(vol_data), np.mean(vol_data), np.std(vol_data))

            # Resize CT to downsampled shape for prediction
            vol_data_resized = resize(vol_data, self.image_shape_lowres)
            print('Original resized to shape: ', vol_data_resized.shape)
            
            # Predict
            pred = self._predict(vol_data_resized)
            print('Predicted to shape: ', pred.shape)

            # Upsample
            pred_up = resize(pred, [vol_data.shape[0], self.image_shape_highres[1], self.image_shape_highres[2]])
            print('Prediction upsampled to shape: ', pred_up.shape)

            # Downsample the original vol_data and seg_data
            vol_data = resize(vol_data, [vol_data.shape[0], self.image_shape_highres[1], self.image_shape_highres[2]])
            seg_data = resize(seg_data, [seg_data.shape[0], self.image_shape_highres[1], self.image_shape_highres[2]])
            # Insert a new dimension (axis) at position -1
            vol_data = np.expand_dims(vol_data, axis=-1)
            seg_data = np.expand_dims(seg_data, axis=-1)
            print('Original downsampled to :', vol_data.shape, seg_data.shape)

            # Concatenate the upsampled prediction and the downsampled original
            new_vol = self._concatenate(pred_up, vol_data)
            print('Concatenated to shape: ', new_vol.shape)

            # Write new tfrecords
            train_writer, val_writer = self._write_tfrecords(i, new_vol, seg_data, train_writer, val_writer)

        print('Number of training samples: {}, Number of validation samples: {}.'.format(self.n_train, self.n_val))
        train_writer.close()
        val_writer.close()
        sys.stdout.flush()
        log.info('YELLOW', 'Saved as TFRecords')
        return self.path_to_tfrecords_train, self.path_to_tfrecords_val, self.n_train, self.n_val


    def _predict(self, vol_data):
        '''Predict on the low-res model
        '''
        # Insert a new dimension (axis) at position -1
        image_expanded = np.expand_dims(vol_data, axis=-1)
        # Insert a new dimension (axis) at position 0
        image_expanded = np.expand_dims(image_expanded, axis=0)

        # Predict
        pred_mask = self.model.predict(image_expanded)
        argmax_mask = create_mask(pred_mask)
        return argmax_mask


    def _concatenate(self, pred, vol_data):
        '''Concatenate the predictions with the original image
        '''
        return np.concatenate((pred, vol_data), axis=-1)

    def _write_tfrecords(self, i, vol_data, seg_data, train_writer, val_writer):
        '''Save the new tfrecords as axial slabs to be trained
        '''
        # Calculate number of axial slabs s
        s = math.floor(vol_data.shape[0] / self.image_shape_highres[0])
        print('CT scan will be split into {} slabs of size {}.'.format(s, self.image_shape_highres))
        print(range(s))
        # Split into slabs
        for j in range(s):
            start = self.image_shape_highres[0] * j
            end = self.image_shape_highres[0] * (j + 1)
            print(start, end)

            vol_slab = vol_data[start : end, :, :]
            seg_slab = seg_data[start : end, :, :]

            # Expand dims and change datatype to string to be saved by TF
            vol_slab = make_tfrecords_ready_3d(vol_slab)
            seg_slab = make_tfrecords_ready_3d(seg_slab)
            # Create a feature
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature={
                # Wrap the data as TensorFlow features
                'data/img': self._bytes_feature(vol_slab),
                'data/seg': self._bytes_feature(seg_slab)}))

            # Serialize to string and write to the TFRecords file
            if i in self.train_range:
                train_writer.write(example.SerializeToString())
                self.n_train += 1
            elif i in self.val_range:
                val_writer.write(example.SerializeToString())
                self.n_val += 1
            else:
                raise ValueError("Not in training or validation range.")
        return train_writer, val_writer