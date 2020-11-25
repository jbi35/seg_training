import os
import pytest
import shutil
import tempfile
import numpy as np
import nibabel as nib
import tensorflow as tf
import calculon.utils.write_tfrecords as wtf
from calculon.utils.image_utils import normalize_volume, filter_volume, compute_median_frequency, get_file_paths, get_slice, make_tfrecords_ready, get_n_slices, get_axis_from_view, read_nifti_file

def test_init():
    a = wtf.Writer("./", "nii", "./", "./", False, "axial", 512, 512)
    assert isinstance(a, wtf.Writer)
    b = wtf.Writer3D("./", "nii", "./", "./", False, "axial", [64, 64, 64])
    assert isinstance(b, wtf.Writer3D)
    c = wtf.Writer3DSlabs("./", "nii", "./", "./", False, "axial", [64, 64, 64])
    assert isinstance(c, wtf.Writer3DSlabs)
    d = wtf.Writer3DSlabsConcat("./", "./", "./", "./")
    assert isinstance(d, wtf.Writer3DSlabsConcat)

def test_bytes_feature():
    a = wtf.Writer("./", "nii", "./", "./", False, "axial", 512, 512)

    img = np.zeros((512,512), dtype=int)
    img = make_tfrecords_ready(img)
    img = a._bytes_feature(img)

    img2 = np.zeros((512,512,1), dtype=np.float32)
    img2 = img2.tostring()
    img2 = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img2]))
    assert img == img2

def test_save_data():
    np.random.seed(seed=42)
    # Set up
    test_out_dir = tempfile.mkdtemp()
    test_vol_dir = tempfile.mkdtemp()
    test_seg_dir = tempfile.mkdtemp()

    # Create sample nifti images
    img = np.random.randint(2, size=(512,512,1))
    empty_header = nib.Nifti1Header()
    img = nib.Nifti1Image(dataobj=img, affine=None, header=empty_header)
    test_vol_path = os.path.join(test_vol_dir, 'test_vol.nii.gz')
    test_seg_path = os.path.join(test_seg_dir, 'test_seg.nii.gz')
    nib.save(img, test_vol_path)
    nib.save(img, test_seg_path)

    # Write the data with the function to test
    a = wtf.Writer(test_out_dir, "nii", [test_vol_path], [test_seg_path], False, "axial", 512, 512)
    a.save_data()
    # Read the data
    filenames = [test_out_dir + 'train.tfrecords']
    raw_dataset = tf.data.TFRecordDataset(filenames)
    # Create a dictionary describing the features
    features = {
        'data/slice': tf.io.FixedLenFeature([], tf.string),
        'data/seg': tf.io.FixedLenFeature([], tf.string),
    }
    # Read the dataset
    dataset = raw_dataset.map(lambda x: tf.io.parse_single_example(x, features))
    # Check if created a proper iterable
    tf.debugging.assert_proper_iterable(dataset)
    
    # Tear down
    shutil.rmtree(test_out_dir)
    shutil.rmtree(test_vol_dir)
    shutil.rmtree(test_seg_dir)

def test_save_data_3D():
    np.random.seed(seed=42)
    # Set up
    test_out_dir = tempfile.mkdtemp()
    test_vol_dir = tempfile.mkdtemp()
    test_seg_dir = tempfile.mkdtemp()

    # Create sample nifti images
    img = np.random.randint(2, size=(512,512,1))
    empty_header = nib.Nifti1Header()
    img = nib.Nifti1Image(dataobj=img, affine=None, header=empty_header)
    test_vol_path = os.path.join(test_vol_dir, 'test_vol.nii.gz')
    test_seg_path = os.path.join(test_seg_dir, 'test_seg.nii.gz')
    nib.save(img, test_vol_path)
    nib.save(img, test_seg_path)

    # Write the data with the function to test
    a = wtf.Writer3D(test_out_dir, "nii", [test_vol_path], [test_seg_path], False, "axial", [64, 64, 64])
    a.save_data()
    # Read the data
    filenames = [test_out_dir + 'train.tfrecords']
    raw_dataset = tf.data.TFRecordDataset(filenames)
    # Create a dictionary describing the features
    features = {
        'data/slice': tf.io.FixedLenFeature([], tf.string),
        'data/seg': tf.io.FixedLenFeature([], tf.string),
    }
    # Read the dataset
    dataset = raw_dataset.map(lambda x: tf.io.parse_single_example(x, features))
    # Check if created a proper iterable
    tf.debugging.assert_proper_iterable(dataset)
    
    # Tear down
    shutil.rmtree(test_out_dir)
    shutil.rmtree(test_vol_dir)
    shutil.rmtree(test_seg_dir)

def test_save_data_3DSlabs():
    np.random.seed(seed=42)
    # Set up
    test_out_dir = tempfile.mkdtemp()
    test_vol_dir = tempfile.mkdtemp()
    test_seg_dir = tempfile.mkdtemp()

    # Create sample nifti images
    img = np.random.randint(2, size=(512,512,1))
    empty_header = nib.Nifti1Header()
    img = nib.Nifti1Image(dataobj=img, affine=None, header=empty_header)
    test_vol_path = os.path.join(test_vol_dir, 'test_vol.nii.gz')
    test_seg_path = os.path.join(test_seg_dir, 'test_seg.nii.gz')
    nib.save(img, test_vol_path)
    nib.save(img, test_seg_path)

    # Write the data with the function to test
    a = wtf.Writer3DSlabs(test_out_dir, "nii", [test_vol_path], [test_seg_path], False, "axial", [64, 64, 64])
    a.save_data()
    # Read the data
    filenames = [test_out_dir + 'train.tfrecords']
    raw_dataset = tf.data.TFRecordDataset(filenames)
    # Create a dictionary describing the features
    features = {
        'data/slice': tf.io.FixedLenFeature([], tf.string),
        'data/seg': tf.io.FixedLenFeature([], tf.string),
    }
    # Read the dataset
    dataset = raw_dataset.map(lambda x: tf.io.parse_single_example(x, features))
    # Check if created a proper iterable
    tf.debugging.assert_proper_iterable(dataset)
    
    # Tear down
    shutil.rmtree(test_out_dir)
    shutil.rmtree(test_vol_dir)
    shutil.rmtree(test_seg_dir)

def test_save_data_3DSlabsConcat():
    np.random.seed(seed=42)
    # Set up
    inputs = tf.keras.Input(shape = (1,1,1,1))
    outputs = tf.keras.layers.Conv3D(filters=2, kernel_size=(1, 1, 1), activation='softmax')(inputs)
    test_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    test_model.compile()
    test_out_dir = tempfile.mkdtemp()
    test_vol_dir = tempfile.mkdtemp()
    test_seg_dir = tempfile.mkdtemp()

    # Create sample nifti images
    img = np.random.randint(2, size=(512,512,1))
    empty_header = nib.Nifti1Header()
    img = nib.Nifti1Image(dataobj=img, affine=None, header=empty_header)
    test_vol_path = os.path.join(test_vol_dir, 'test_vol.nii.gz')
    test_seg_path = os.path.join(test_seg_dir, 'test_seg.nii.gz')
    nib.save(img, test_vol_path)
    nib.save(img, test_seg_path)

    # Write the data with the function to test
    a = wtf.Writer3DSlabsConcat(test_model, test_out_dir, [test_vol_path], [test_seg_path])
    a.save_data()
    # Read the data
    filenames = [test_out_dir + 'concat_train.tfrecords']
    raw_dataset = tf.data.TFRecordDataset(filenames)
    # Create a dictionary describing the features
    features = {
        'data/slice': tf.io.FixedLenFeature([], tf.string),
        'data/seg': tf.io.FixedLenFeature([], tf.string),
    }
    # Read the dataset
    dataset = raw_dataset.map(lambda x: tf.io.parse_single_example(x, features))
    # Check if created a proper iterable
    tf.debugging.assert_proper_iterable(dataset)
    
    # Tear down
    shutil.rmtree(test_out_dir)
    shutil.rmtree(test_vol_dir)
    shutil.rmtree(test_seg_dir)

