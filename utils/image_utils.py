import io
import os
import re
import csv
import sys
import math
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
from dipy.align.reslice import reslice


def create_mask(pred_mask):
    '''Creates prediction masks from a model.predict() output by selecting the 
    prediction with the highest probability.
    
    Args:
        pred_mask (np.array): Predictions for each class.
    
    Returns:
        tf.int64: Prediction mask tensor.
    '''
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(image, mask, model):
    '''Selects images from a dataset and predicts them on a model.
    
    Args:
        image (tf.Tensor):      Image to be segmented
        mask (tf.Tensor):       Segmentation ground truth
        model (tf.model):       TF model
    '''
    pred_mask = model.predict(image)
    figure = display([image[0], mask[0], create_mask(pred_mask)])
    return figure

def show_predictions_3d(image, mask, model):
    '''Selects images from a dataset and predicts them on a model.
    
    Args:
        image (tf.Tensor):      Image to be segmented
        mask (tf.Tensor):       Segmentation ground truth
        model (tf.model):       TF model
    '''
    # Merge one hot encoded ground truth 
    mask = create_mask(mask)
    # Get the prediction mask
    pred_mask = model.predict(image)
    argmax_mask = create_mask(pred_mask)
    # Get the middle slice (axial view)
    d_mid = int(image.shape[1]/2)
    # Build the figure
    figure = display([image[0,d_mid,:,:], mask[d_mid,:,:], argmax_mask[d_mid,:,:]])
    return figure

def show_inference_3d(image, model):
    '''Takes a ct image and a 3d model. Uses the models predict function to build
    a segmentation mask. Then, returns a figure of one slice of the original image
    and one slice of the segmentation mask.
    
    Args:
        image (tf.Tensor):  Image to be segmented
        model (tf.model):   TF model
    
    Returns:
        fig: Mathplotlib figure
    '''
    pred_mask = model.predict(image)
    argmax_mask = create_mask(pred_mask)
    print('prediction shape:', image.shape, argmax_mask.shape)
    d_mid = int(image.shape[1]/2)
    print(d_mid)
    figure = display([image[0,d_mid,:,:], argmax_mask[d_mid,:,:]])
    return figure

def display(display_list):
    '''Displays a list of images.
    
    Args:
        display_list (array): Array of image tensors to display.
    '''
    figure = plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask', 'LowRes Prediction']
    # Add the input image
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[0], data_format='channels_last'))
    # Add the true/and or predicted mask
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        # Preprocess to image
        image = tf.keras.preprocessing.image.array_to_img(display_list[i], data_format='channels_last')
        # Rotate
        #image = np.rot90(image, k=3)
        plt.title(title[i])
        # Add plot in grayscale tone
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    #plt.show()
    return figure

def two_display(display_list):
    '''Displays a Mathplotlib figure from a list of images.
    
    Args:
        display_list (array): List of image tensors to be displayed.
    '''
    plt.figure(figsize=(10, 10))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i], data_format='channels_first'), cmap='gray')
        plt.axis('off')
    plt.show()

def plot_to_image(figure):
    '''Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    
    Args:
        figure (plt.fig): Figure
        
    Returns:
        tf.image: Tensorflow image'''
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    #image = tf.expand_dims(image, 0)
    return image

def read_nifti_file(file_path):
    '''Reads a nifti file and returns a data array.
    
    Args:
        file_path (str): Path to nifti file.
    
    Returns:
        dtype: Array of image data of data type dtype.
    '''
    # Read and convert to array
    image = sitk.ReadImage(file_path)
    image = sitk.GetArrayFromImage(image)
    # Convert to int16 HU if not already the case
    if not isinstance(image, np.int16): 
        image = image.astype(np.int16)
    return image

def read_dicom_series(file_path):
    '''Reads a dicom series and returns a data array.

    Args:
        file_path (str): Path to a DICOM series directory.
    
    Returns:
        dtype: Array of image data of data type dtype.
    '''
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(file_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    origen = image.GetOrigin()
    size = image.GetSize()
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    # Get the array
    image = sitk.GetArrayFromImage(image)
    # Convert to int16 HU if not already the case
    if not isinstance(image, np.int16): 
        image = image.astype(np.int16)
    return image, origen, size, spacing, direction

def get_axis_from_view(view):
    '''Returns axis per view.
    
    Args:
        view (str): The view from which the axis shall be seen.
    
    Raises:
        ValueError: If the view is unknown.
    
    Returns:
        int: The axis as an integer.
    '''
    if view == 'axial':
        return 2
    elif view == 'coronal':
        return 1
    elif view == 'sagittal':
        return 0
    else:
        raise ValueError('Error. View is unknown. Select from: {axial, coronal, sagittal}.')

def get_slice(view, data, slice_idx):
    '''Returns slices of shape (x,y) by view.   
    
    Args:
        view (str):         The view from which to slice
        data (array):       3D data
        slice_idx (int):    Index of current slice
    
    Raises:
        ValueError: If the view is unkown.
    
    Returns:
        np.array: Image slice
    '''
    if view == 'axial':
        return np.asarray(data[:, :, slice_idx])
    elif view == 'coronal':
        return np.asarray(data[:, slice_idx, :])
    elif view == 'sagittal':
        return np.asarray(data[slice_idx, :, :])
    else:
        raise ValueError('Error. View is unknown. Select from: {axial, coronal, sagittal}.')

def get_n_slices(view, data):
    '''Returns number of slices of 3D data.   
    
    Args:
        data (array):   3D data
    
    Raises:
        ValueError: If the view is unkown.
    
    Returns:
        int: Number of slices
    '''
    if view == 'axial':
        return data.shape[2]
    elif view == 'coronal':
        return data.shape[1]
    elif view == 'sagittal':
        return data.shape[0]
    else:
        raise ValueError('Error. View is unknown. Select from: {axial, coronal, sagittal}.')

def make_tfrecords_ready(image):
    '''Expand the dimension of a (h,w) image slice and change dtype to string to be saved into a TFrecords file.
    
    Args:
        image (array): Image array of size (h,w).
    
    Returns:
        str: Image of datatype bytes string
    '''
    # Insert a new dimension (axis) at position -1
    image_expanded = np.expand_dims(image, axis=-1)

    # Change datatype to string in order to be saved by TF
    image_float = image_expanded.astype(np.float32)
    image_string = image_float.tostring()
    return image_string

def make_tfrecords_ready_3d(image):
    '''Change dtpye to string to be saved into a TFRecords file
    
    Args:
        image (array): Image array
    
    Returns:
        str: Image of dtype bytes string
    '''
    # Insert a new dimension (axis) at position -1
    image_expanded = np.expand_dims(image, axis=-1)
    # Change datatype to string in order to be saved by TF
    image_float = image_expanded.astype(np.float32)
    print("shape image float",image_float.shape)
    image_string = image_float.tostring()
    return image_string

def get_file_paths_csv(csv_path, img_base_path=""):
    '''Reads a csv file with two columns: paths to CT scans and paths to 
    segmentations. Creates two arrays: one of paths to CT scans and one of
    paths to respective segmentations. 
    
    Args:
        csv_path (str):                 Path to the csv file to be read.
        img_base_path (str, optional):  Path added in front of the paths of the csv file, e.g. path to local directory or AWS S3 bucket. Defaults to empty.
    
    Returns:
        array:  Two arrays. One with the paths to CTs and one with paths to
                segmentations
    '''
    vol_paths = []
    seg_paths = []
    seg_types = []
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            vol_paths.append(img_base_path + row[0])
            seg_paths.append(img_base_path + row[1])
    return vol_paths, seg_paths, seg_types

def get_file_paths(file_path):
    '''Goes through a directory and lists all filepaths ending with ".gz", 
    sorts them alphabetically and returns the list of paths.
    
    Args:
        file_path (str): Path to files.
    
    Returns:
        array: List of filepaths, sorted alphabetically.
    '''
    file_paths = []
    filenames = os.listdir(file_path)

    def position(text):
        return int(text) if text.isdigit() else text

    def split(text):
        return [position(c) for c in re.split('(\d+)', text)]

    filenames.sort(key=split)

    for filename in filenames:
        if filename.endswith(".gz"):
            file_paths.append(os.path.join(file_path, filename))
            continue
        else:
            continue
    return file_paths

def get_weights(path_to_vol_data, path_to_seg_data, num_classes, filter, view):
    '''Calculates 
    
    Args:
        path_to_vol_data ([type]): [description]
        path_to_seg_data ([type]): [description]
        num_classes ([type]): [description]
        filter ([type]): [description]
        view ([type]): [description]
    
    Returns:
        [type]: [description]
    ''' 
    vol_paths = get_file_paths(path_to_vol_data)
    seg_paths = get_file_paths(path_to_seg_data)
    axis = get_axis_from_view(view)
    weights_array = compute_median_frequency(vol_paths, seg_paths, num_classes, filter, axis)
    print("Weights: ", weights_array)
    return weights_array

def standardize_volume(vol_data):
    '''Standardize image data to mean = 0 and std = 1
    
    Args:
        vol_data (np.array): Image data.
    
    Returns:
        np. array: Image data standardized.
    '''
    mean = np.mean(vol_data)
    std = np.std(vol_data)
    return (vol_data - mean) / std

def normalize_volume(vol_data):
    '''Normalize image data between [0,1]
    
    Args:
        vol_data (np.array): Image array
    
    Returns:
        np.array: Normalized image array.
    '''
    img_min = np.min(vol_data)
    img_max = np.max(vol_data)
    return (vol_data - img_min)/(img_max - img_min)

def compute_mean_variance_dataset(file_paths, seg_paths, filter, axis=2):
    sum = 0
    count = 0
    data2 = 0
    n = 0
    for i in range(len(file_paths)):
        #if not n % 10:
        print('Computing mean in vols: {}/{}'.format(n, len(file_paths)))
        sys.stdout.flush()

        vol_data = read_nifti_file(file_paths[i])
        seg_data = read_nifti_file(seg_paths[i])

        # Select the slices that contain labels
        if filter:
            vol_data, seg_data = filter_volume(vol_data, seg_data, axis=axis)

        h, w, d = np.shape(vol_data)
        count += h * w * d
        sum += np.sum(vol_data)
        data2 += np.sum(vol_data * vol_data)
        n += 1

    mean = sum / count
    std_dev = math.sqrt(abs(data2 / count - (mean * mean)))
    return mean, std_dev

def compute_mean_variance_file(vol_data):
    sum = 0
    count = 0
    data2 = 0
    n = 0

    h, w, d = np.shape(vol_data)
    count += h * w * d
    sum += np.sum(vol_data)
    data2 += np.sum(vol_data * vol_data)
    n += 1

    mean = sum / count
    std_dev = math.sqrt(abs(data2 / count - (mean * mean)))
    return mean, std_dev

def compute_median_frequency(file_paths, seg_paths,  num_classes, filter, axis=2):
    histo = np.zeros(num_classes)
    for i in range(len(file_paths)):
        if not i % 2:
            print('Computing weights in vols: {}/{}'.format(i, len(file_paths)))
            sys.stdout.flush()

        ## read data
        data = nib.load(file_paths[i])
        vol_data = data.get_data()
        seg_data = nib.load(seg_paths[i])
        seg_data = seg_data.get_data()

        ## resample data to isotropic resolution
        affine = data.affine
        zooms = data.header.get_zooms()[:3]
        new_zooms = (zooms[0], zooms[0], zooms[0])
        vol_data, affine = reslice(vol_data, affine, zooms, new_zooms)
        seg_data, affine = reslice(seg_data, affine, zooms, new_zooms, order=0)

        if filter:
            data, seg_data = filter_volume(vol_data, seg_data, axis=axis)

        unique, count = np.unique(seg_data, return_counts=True)
        for n in range(num_classes):
            if n in unique:
                index = np.where(unique == n)
                histo[n] += count[index]

        # print('')

    # compute median frequency balance
    freq = histo / np.sum(histo)
    med_freq = np.median(freq)
    weights = med_freq / freq
    return np.asarray(weights)

def filter_volume(vol_data, seg_data, axis=2):
    '''Return only voxels containing labels according to axis.
    
    Args:
        vol_data (array): CT data
        seg_data (array): Seg data
        axis (int, optional): View. Defaults to 2.
    
    Returns:
        array, array: Filtered vol, filtered seg data
    '''
    h, w, d = np.where(seg_data != 0)

    if axis == 0:  # sagital
        start = min(h)
        end = max(h) + 1
        vol_filter = vol_data[start:end, :, :]
        seg_filter = seg_data[start:end, :, :]

    elif axis == 1:  # coronal
        start = min(w)
        end = max(w) + 1
        vol_filter = vol_data[:, start:end, :]
        seg_filter = seg_data[:, start:end, :]

    else:  # 2 axial
        start = min(d)
        end = max(d) + 1
        vol_filter = vol_data[:, :, start:end]
        seg_filter = seg_data[:, :, start:end]

    return vol_filter, seg_filter