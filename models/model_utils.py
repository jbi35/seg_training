from keras import backend as K

def dice_coef(y_true, y_pred, smooth=1):
    '''Calculates the dice coefficient
    
    Args:
        y_true (tf.Tensor):     Ground truth.
        y_pred (tf.Tensor):     Predicted segmentation.
        smooth (int, optional): Smoothing factor. Defaults to 1.
    
    Returns:
        float: Dice coefficient.
    '''
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    '''Calculates the dice coefficient loss function.
    
    Args:
        y_true (tf.Tensor): Ground  truth.
        y_pred (tf.Tensor): Predicted segmentation.
    
    Returns:
        float: Dice coefficient loss.
    '''
    return 1-dice_coef(y_true, y_pred)