import os
import math
import shutil
import datetime
import numpy as np
import tensorflow as tf
import calculon.utils.log as log
from IPython.display import clear_output
from calculon.models.unet2d import Unet2D
from calculon.models.multires_unet3d import MultiResModel3d
from calculon.training.tfrecord_provider_2d import TFRecordsReader2D
from calculon.training.tfrecord_provider_3d import TFRecordsReader3D
from calculon.utils.validoptions import ValidOption
from calculon.utils.write_tfrecords import Writer3D, Writer3DSlabsConcat
from calculon.utils.tfrecords_utils import get_tfrecord_filenames
from calculon.utils.image_utils import show_predictions, show_predictions_3d, plot_to_image
from calculon.utils.db_utils import connect_db, extract_ct_scans
from calculon.utils.aws_utils import download_from_s3
from calculon.utils.gpu_utils import check_gpus


class Launcher(object):
    ''' Class for launching training with TFRecords or Nifti Files
    '''
    ct_database_options = {
        'path_to_db':
            ValidOption(
                type=str,
                default='./',
                help='Define the path to the sqlite db file.'),
        'disease':
            ValidOption(
                type=list,
                subtype=str,
                default=['any'],
                help='Select disease.'),
        'dataset':
            ValidOption(
                type=list,
                subtype=str,
                default=['any'],
                help='Select a dataset or any.'),
        'seg_type':
            ValidOption(
                type=list,
                subtype=str,
                default=['lungs'],
                help='Select the type of segmentation.'),
        'num_scans':
            ValidOption(
                type=int,
                default=0,
                help='Select the number of scans.'),
        'scan_data_format':
            ValidOption(
                type=str,
                default='nifti',
                help='Define the CT scan file type: [nifti, dicom].'),
        'seg_data_format':
            ValidOption(
                type=str,
                default='nifti',
                help='Define the segmentation file type: [nifti, dicom].')
    }
    tfrecords_options = {
        'file_type':
            ValidOption(
                type=str,
                default='nii',
                help='Define the input file type.'),
        'path_to_csv':
            ValidOption(
                type=str,
                default='./',
                help='Specifies the path to the csv file, which contains a list of the CT and segmentation files to be read.'),
        'img_base_path':
            ValidOption(
                type=str,
                default='./',
                help='Specifies the base CT volumes directory.'),
        'filter':
            ValidOption(
                type=bool,
                default=False,
                help='Indicates whether to filter for images that include labels.'),
        'view':
            ValidOption(
                type=str,
                default='axial',
                help='From which view the file shall be sliced. Possible values: {axial, coronal, sagittal}.')      
    }
    training_options = {
        'num_gpus':
            ValidOption(
                type=int,
                default=1,
                help='Number of GPUs to train on.'),
        'path_to_tfrecords_train':
            ValidOption(
                type=str,
                default="./",
                help='Path to tfrecords for training.'),
        'path_to_tfrecords_val':
            ValidOption(
                type=str,
                default="./",
                help='Path to tfrecords for validation.'),
        'image_shape_resize':
            ValidOption(
                type=list,
                subtype=int,
                default=[128, 128, 128],
                help='Resize image to this shape when training.'),
        'num_classes':
            ValidOption(
                type=int,
                default=2,
                help='Number of classes to segment into.'),
        'channels':
            ValidOption(
                type=int,
                default=1,
                help='Number of channels.'),
        'batch_size':
            ValidOption(
                type=int,
                default=2,
                help='Number of images per batch.'),
        'epochs':
            ValidOption(
                type=int,
                default=1,
                help='Number of epochs (iterations through the complete dataset) to train.'),
        'optimizer':
            ValidOption(
                type=str,
                default="adam",
                help='Optimizer to train the network.'),
        'learning_rate':
            ValidOption(
                type=float,
                default=0.001,
                help='Learning rate of the optimizer.'),
        'decay_rate':
            ValidOption(
                type=float,
                default=1.0,
                help='Rate that is multiplied with the learning rate in order to decay it every decay_steps.'),
        'decay_steps':
            ValidOption(
                type=int,
                default=100000,
                help='Decay every n steps.'),
        'restore':
            ValidOption(
                type=bool,
                default=False,
                help='Defines if pretrained weights should be loaded from output_dir.'),
        'path_to_ckpt':
            ValidOption(
                type=str,
                default="./",
                help='Path to pretrained weights.')
    }

    @classmethod
    def from_options(cls, options):
        ''' Create trainer from options dictionary

        Args:
            options (dict): Dictionary with all necessary options

        Returns:
            Trainer:   Trainer object

        '''
        output_dir = options['output_dir']
        path_to_db = options['ct_database']['path_to_db']
        disease = options['ct_database']['disease']
        dataset = options['ct_database']['dataset']
        seg_type = options['ct_database']['seg_type']
        num_scans = options['ct_database']['num_scans']
        scan_data_format = options['ct_database']['scan_data_format']
        seg_data_format = options['ct_database']['seg_data_format']
        file_type = options['tfrecords']['file_type']
        path_to_csv = options['tfrecords']['path_to_csv']
        img_base_path = options['tfrecords']['img_base_path']
        filter = options['tfrecords']['filter']
        view = options['tfrecords']['view']
        num_gpus = options['training']['num_gpus']
        path_to_tfrecords_train = options['training']['path_to_tfrecords_train']
        path_to_tfrecords_val = options['training']['path_to_tfrecords_val']
        epochs = options['training']['epochs']
        batch_size = options['training']['batch_size']
        image_shape_resize = options['training']['image_shape_resize']
        num_classes = options['training']['num_classes']
        channels = options['training']['channels']
        restore = options['training']['restore']
        path_to_ckpt = options['training']['path_to_ckpt']
        optimizer = options['training']['optimizer']
        learning_rate = options['training']['learning_rate']
        decay_rate = options['training']['decay_rate']
        decay_steps = options['training']['decay_steps']
        return cls(output_dir, path_to_db, disease, dataset, seg_type, num_scans, 
                scan_data_format, seg_data_format, file_type, path_to_csv, img_base_path, 
                filter, view, num_gpus, path_to_tfrecords_train, path_to_tfrecords_val, 
                epochs, batch_size, image_shape_resize, 
                num_classes, channels, restore, path_to_ckpt, 
                optimizer, learning_rate, decay_rate, decay_steps)

    def __init__(self, output_dir, path_to_db, disease, dataset, seg_type, num_scans, 
                scan_data_format, seg_data_format, file_type, path_to_csv, img_base_path, 
                filter, view, num_gpus, path_to_tfrecords_train, path_to_tfrecords_val, 
                epochs, batch_size, image_shape_resize, 
                num_classes, channels, restore, path_to_ckpt, 
                optimizer, learning_rate, decay_rate, decay_steps):

        self.output_dir = output_dir
        self.path_to_db = path_to_db
        self.disease = disease
        self.dataset = dataset
        self.seg_type = seg_type
        self.num_scans = num_scans
        self.scan_data_format = scan_data_format
        self.seg_data_format = seg_data_format
        self.file_type = file_type
        self.img_base_path = img_base_path
        self.filter = filter
        self.view = view
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints/ckpt_epoch{epoch:02d}.ckpt")
        self.savedmodel_dir = os.path.join(self.output_dir, "saved_model/")
        self.num_gpus = num_gpus
        self.path_to_tfrecords_train = path_to_tfrecords_train
        self.path_to_tfrecords_val = path_to_tfrecords_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_shape_resize = image_shape_resize
        self.num_classes = num_classes
        self.channels = channels
        self.restore = restore
        self.path_to_ckpt = path_to_ckpt
        self.optimizer = optimizer
        if decay_rate < 1:
            # use a learning schedule if the decay rate is smaller than 1.
            self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
        else:
            self.learning_rate = learning_rate
        self.val_steps = int((self.num_scans*0.4) // self.batch_size)
        self.steps_per_epoch = int((self.num_scans*0.6) // self.batch_size)
        self.log_dir = self.output_dir + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.res = '_'.join(str(x) for x in self.image_shape_resize)
        
    def check_data_source(self):
        '''Checks if path to tfrecords train and val is empty. If so, first create 
        tfrecords files.
        
        Raises:
            ValueError: If tfrecords paths and raw file paths are all empty.
        '''
        if (self.path_to_tfrecords_train != "" and self.path_to_tfrecords_val != ""):
            print("Data source: TFRecords")
        elif (self.num_scans != 0):
            print("Data source: Database. Will gather paths to be converted into TFRecords.")
            self._get_paths_from_db()
            self._create_tfrecords_3d()
        else:
            raise ValueError("No training data source found. Please specify either path_to_tfrecords or set num_scans above 0 to choose from the database.")

    def _get_paths_from_db(self):
        '''Creates a python dictionary from a SQLite database file.
        '''
        # Connect to db
        c, conn = connect_db(self.path_to_db)
        # Extract scans, creates a csv
        self.paths_dict = extract_ct_scans(c, self.output_dir, self.disease, self.dataset, self.seg_type, self.num_scans, 
                            self.scan_data_format, self.seg_data_format)
        # Close the connection
        conn.close()
    
    def _create_tfrecords_3d(self):
        '''Creates tfrecords from CT scan and segmentation files.
        '''
        # Get volume and segmentation paths from csv
        if self.img_base_path == "ct_data/":
            vol_paths, seg_paths = download_from_s3(self.paths_dict, self.img_base_path, self.output_dir) #TODO: downlaod_from_s3 needs to be able to read the paths dict
        else:
            vol_paths, seg_paths = get_file_paths_csv(self.paths_dict, self.img_base_path) # TODO: get_file_paths_csv needs to be changed to read the paths dict
        # Merge segmentations if necessary
        # TODO: create a separate function which merges multiple segmentations together
        # TODO: delete the paths_csv from the input jsons
        # Write to tfrecords
        w = Writer3D(self.output_dir, self.file_type, vol_paths, seg_paths,
                self.filter, self.view, self.image_shape_resize)
        # Save tfrecords and get output directories returned
        self.path_to_tfrecords_train, self.path_to_tfrecords_val = w.save_data()
    
    def _create_tfrecords_3d_slabsconcat(self, model):
        '''Creates tfrecords from CT scan and segmentation files.
        '''
        # Get volume and segmentation paths from csv
        if self.img_base_path == "ct_data/":
            vol_paths, seg_paths = download_from_s3(self.path_to_csv, self.img_base_path, self.output_dir) # TODO: do the same here as in create_tfrecords_3d
        else:
            vol_paths, seg_paths = get_file_paths_csv(self.path_to_csv, self.img_base_path)
        # Write to tfrecords
        w = Writer3DSlabsConcat(model, self.output_dir, vol_paths, seg_paths)
        # Save tfrecords and get output directories returned
        self.path_to_tfrecords_train, self.path_to_tfrecords_val, n_train, n_val = w.save_data()

        # Change val_steps and train_steps accordingly
        self.steps_per_epoch = n_train // self.batch_size
        self.val_steps = n_val // self.batch_size

    def _read_tfrecords(self):
        '''Reads tfrecords with the TFrecordsReader for training and validation.
        
        Returns:
            tf.dataset: Returns an iterable tf dataset.
        '''
        # Read tfrecords
        train_filenames = get_tfrecord_filenames(self.path_to_tfrecords_train)
        val_filenames = get_tfrecord_filenames(self.path_to_tfrecords_val)
        
        # Load tfrecords into a TFRecordDataset
        train_tfrecorddataset = tf.data.TFRecordDataset(filenames=train_filenames)
        val_tfrecorddataset = tf.data.TFRecordDataset(filenames=val_filenames)

        # Decode the data and prepare for training
        log.info('YELLOW', 'Loading Datasets')
        print("Training Datasets: ", train_filenames)
        print("Validation Datasets: ", val_filenames)
        print('Classes:', self.num_classes, 'Channels:', self.channels, 'Input image size:', self.image_shape_resize)
        train_data_provider = TFRecordsReader3D(train_tfrecorddataset, np.ceil(self.num_scans/10), self.batch_size, self.num_classes, self.channels, self.image_shape_resize, validating=False)
        val_data_provider = TFRecordsReader3D(val_tfrecorddataset, np.ceil(self.num_scans/10), self.batch_size, self.num_classes, self.channels, self.image_shape_resize, validating=True)

        train_dataset = train_data_provider.read()
        val_dataset = val_data_provider.read()
        return train_dataset, val_dataset
    
    def launch_3d(self):
        '''Launch the training process for the 3d unet.
        '''
        print(self.res)
        # Read tfrecords
        train_dataset, val_dataset = self._read_tfrecords()

        # Open a strategy scope for multi-GPU training. Pass otherwise.
        strategy_scope = check_gpus(self.num_gpus)
        if strategy_scope:        
            with strategy_scope:
                # Check if number of training samples is compatible with a multi-GPU strategy
                # by checking if dividing the number of training samples by the number of GPUs 
                # results in an integer. 
                if math.ceil(self.num_scans * 0.6) / self.num_gpus == int(math.ceil(self.num_scans * 0.6 ) / self.num_gpus):
                    print("Training with a multi-GPU strategy.")
                else:
                    raise ValueError("Warning: The number of training sample is incompatible with this multi-GPU strategy.")
                # Everything that creates variables should be under the strategy scope.
                # In general this is only model construction & `compile()`.

                # Load the 3D Unet
                Module = MultiResModel3d(self.learning_rate, self.batch_size, self.channels, self.num_classes, self.image_shape_resize, self.optimizer)
                model = Module.build()

                # Compile
                model = Module.compile(model)
        else:
            # Load the 3D Unet
            Module = MultiResModel3d(self.learning_rate, self.batch_size, self.channels, self.num_classes, self.image_shape_resize, self.optimizer)
            model = Module.build()

            # Compile
            model = Module.compile(model)
            
        # Loads the weights
        if self.restore:
            print("Restoring model from", self.path_to_ckpt)
            model.load_weights(self.path_to_ckpt)
        
        # Create a callback that saves the model's weights
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.ckpt_dir, 
                        save_weights_only=True, verbose=1)
        
        # Clear any logs from previous runs
        shutil.rmtree(self.output_dir + "logs/")

        # Create a callback to tensorboard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1, profile_batch=0)

        # Create a callback for early stopping

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                                        monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto',
                                        baseline=None, restore_best_weights=False
                                    )


        # Creates a file writer for the image prediction log directory.
        file_writer = tf.summary.create_file_writer(self.log_dir + '/img')

        # Start the training and evaluation        
        model.fit(x=train_dataset, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch, 
                    callbacks=[DisplayCallback3D(model, val_dataset, file_writer), ckpt_callback, 
                    tensorboard_callback, early_stopping_callback],
                    validation_data=val_dataset, validation_steps=self.val_steps)
        
        # Then, save the model
        model.save(self.savedmodel_dir + '/' + self.res, save_format='tf')

    def launch_multires3d(self):
        '''Launch the training process for multires 3d unet.
        '''
        tf.test.is_gpu_available()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)

        # Read tfrecords
        train_dataset, val_dataset = self._read_tfrecords()

        # Load the 3D Unet
        Module = MultiResModel3d(self.learning_rate, self.batch_size, self.channels, self.num_classes, self.image_shape_resize, self.optimizer)
        model = Module.build()

        # Compile
        model = Module.compile(model)

        # Create a callback that saves the model's weights
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.ckpt_dir, 
                        save_weights_only=True, verbose=1)
        
        # Clear any logs from previous runs
        shutil.rmtree(self.output_dir + "logs/")

        # Create a callback to tensorboard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1, profile_batch=0)

        # Creates a file writer for the image prediction log directory.
        file_writer = tf.summary.create_file_writer(self.log_dir + '/img')

        # Start the training and evaluation        
        model.fit(x=train_dataset, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch, 
                    callbacks=[DisplayCallback3D(model, val_dataset, file_writer), ckpt_callback, tensorboard_callback],
                    validation_data=val_dataset, validation_steps=self.val_steps)
        
        # Then, save the model
        model.save(self.savedmodel_dir + '/' + self.res, save_format='tf')
    
        # Predict on the trained model and concatenate it on new images in higher resolution
        self._create_tfrecords_3d_slabsconcat(model)

        # Update the number of channels and input shapes
        self.channels = 2
        self.image_shape_resize = [256, 256, 32]

        # Read newly created concatenated tfrecords
        train_dataset_concat, val_dataset_concat = self._read_tfrecords()
        print('Datasets: ',train_dataset_concat, val_dataset_concat)

        # Train the HighRes model
        self.image_shape_resize = [256, 256, 32]
        HighResModule = MultiResModel3d(self.learning_rate, self.batch_size, self.channels, self.num_classes, self.image_shape_resize, self.optimizer)
        highres_model = HighResModule.build()
        highres_model = HighResModule.compile(highres_model)
        highres_model.fit(x=train_dataset_concat, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch, 
                    callbacks=[DisplayCallback3D(highres_model, val_dataset_concat, file_writer), ckpt_callback, tensorboard_callback],
                    validation_data=val_dataset_concat, validation_steps=self.val_steps)

        # Save the model
        highres_model.save(self.savedmodel_dir + '/' + self.res, save_format='tf')

    def launch_unet2d(self):
        '''Launch the training process.
        '''
        # Read tfrecords
        train_filenames = get_tfrecord_filenames(self.path_to_tfrecords_train)
        val_filenames = get_tfrecord_filenames(self.path_to_tfrecords_val)

        # Load tfrecords into a TFRecordDataset
        train_tfrecorddataset = tf.data.TFRecordDataset(filenames=train_filenames)
        val_tfrecorddataset = tf.data.TFRecordDataset(filenames=val_filenames)

        # Decode the data and prepare for training
        log.info('YELLOW', 'Loading Datasets')
        print("Training Datasets: ", train_filenames)
        print("Validation Datasets: ", val_filenames)

        train_data_provider = TFRecordsReader2D(train_tfrecorddataset, np.ceil(self.num_scans/10), 0, self.batch_size, 
                            512, 512, self.image_shape_resize[0], self.image_shape_resize[1], self.num_classes)
        val_data_provider = TFRecordsReader2D(val_tfrecorddataset, np.ceil(self.num_scans/10), 0, self.batch_size, 
                            512, 512, self.image_shape_resize[0], self.image_shape_resize[1], self.num_classes)

        train_dataset = train_data_provider.read()
        val_dataset = val_data_provider.read()

        for image, mask in train_dataset.take(1):
            sample_image, sample_mask = image, mask
            print(sample_image.shape)
            print(sample_mask.shape)

        # Load the Unet Model
        Module = Unet2D(learning_rate=self.learning_rate, num_classes=self.num_classes, input_size=[self.image_shape_resize[0], self.image_shape_resize[1], self.channels])
        model = Module.unet()

        # Create a callback that saves the model's weights
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.ckpt_dir, 
                        save_weights_only=True, verbose=1)
        
        # Clear any logs from previous runs
        shutil.rmtree(self.output_dir + "logs/")

        # Create a callback to tensorboard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1, profile_batch=0)

        # Creates a file writer for the image prediction log directory.
        file_writer = tf.summary.create_file_writer(self.log_dir + '/img')

        # Start the training and evaluation        
        model.fit(x=train_dataset, epochs=self.epochs, steps_per_epoch=self.steps_per_epoch, 
                callbacks=[DisplayCallback(model, val_dataset, file_writer), ckpt_callback, tensorboard_callback],
                validation_data=val_dataset, validation_steps=self.val_steps)

        # Save the trained model
        #tf.saved_model.save(model, self.savedmodel_dir)
        model.save(self.savedmodel_dir, save_format='tf')


class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, val_dataset, file_writer):
        '''Display callback for tf keras.
        
        Args:
            model (tf.keras.model):     Tensorflow model
            val_dataset (tf.data):      Validation dataset
            file_writer (tf.summary):   Writer for tf.summary files
        '''
        self.model = model
        self.val_dataset = val_dataset
        self.file_writer = file_writer

    def on_epoch_end(self, epoch, logs=None, num=8):
        '''Actions on end of training epoch.
        
        Args:
            epoch (int):            Current epoch
            logs (str, optional):   Logs. Defaults to None.
            num (int, optional):    Number of sample segmentation slices to output. Defaults to 8.
        ''' 
        clear_output(wait=True)
        figures = []
        for image, mask in self.val_dataset.take(num):
            figure = show_predictions(image, mask, model=self.model)
            figure = plot_to_image(figure)
            figures.append(figure)

        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
        images = tf.stack(figures)
        # Using the file writer, log the reshaped image.
        with self.file_writer.as_default():
            tf.summary.image("Predictions on validation set", images, step=epoch, max_outputs=8)


class DisplayCallback3D(tf.keras.callbacks.Callback):
    def __init__(self, model, val_dataset, file_writer):
        '''Display callback for tf keras.
        
        Args:
            model (tf.keras.model):     Tensorflow model
            val_dataset (tf.data):      Validation dataset
            file_writer (tf.summary):   Writer for tf.summary files
        '''
        self.model = model
        self.val_dataset = val_dataset
        self.file_writer = file_writer

    def on_epoch_end(self, epoch, logs=None, num=8):
        '''Actions on end of training epoch.
        
        Args:
            epoch (int):            Current epoch
            logs (str, optional):   Logs. Defaults to None.
            num (int, optional):    Number of sample segmentation slices to output. Defaults to 8.
        ''' 
        clear_output(wait=True)
        figures = []
        print(self.val_dataset.take(1))
        for image, mask in self.val_dataset.take(num):
            figure = show_predictions_3d(image, mask, model=self.model)
            figure = plot_to_image(figure)
            figures.append(figure)

        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
        images = tf.stack(figures)
        # Using the file writer, log the reshaped image.
        with self.file_writer.as_default():
            tf.summary.image("Predictions on validation set", images, step=epoch, max_outputs=8)