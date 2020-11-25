import tensorflow as tf

def check_gpus(num_gpus):
    '''Checks how many GPUs are listed in the input json file
    and returns the correct function to start the CUDA driver.
    
    Args:
        num_gpus (int): Number of GPUs. 0 if CPU should be used.
    
    Raises:
        ValueError: If number of GPUs is not valid.
    
    Returns:
        func: Function to start the CUDA driver or pass if num_gpus is zero.
    '''
    if isinstance(num_gpus, int):
        if num_gpus == 0:
            return run_cpu()
        if num_gpus == 1:
            return run_1_gpu()
        if num_gpus >= 2:
            return run_multiple_gpus(num_gpus)
        else:
            raise ValueError("No strategy for this number of GPUs defined.")
    else:
        raise ValueError("Number of GPUs must be an integer.")

def run_cpu():
    '''Pass to run on CPU.
    '''
    pass

def run_1_gpu():
    '''Run on one GPU. Restricts Tensorflow to use only the first GPU.
    '''
    # Check whether TensorFlow was built with CUDA (GPU) support
    tf.test.is_built_with_cuda()
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    pass 

def run_multiple_gpus(num_gpus):
    '''Run on multiple GPUs.
    
    Args:
        num_gpus (int): Number of GPUs to train on.
    
    Returns:
        func: Strategy.scope() function of Tensorflow.
    '''
    tf.test.is_built_with_cuda()
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    # TODO: restrict on how many GPUs it should run
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    return strategy.scope()