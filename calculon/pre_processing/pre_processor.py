from calculon.utils.validoptions import ValidOption
import calculon.utils.log as log


class Preprocessor():
    ''' Class for the preprocessing of lung CTs


    '''
    default_options = {
        'scale_images':
        ValidOption(
            type=bool,
            default=False,
            help='Scale Images during pre_processing')
    }

    @classmethod
    def from_options(cls, options):
        ''' Create pre_processor from options dictionary

        Args:
            options (dict): Dictionary with all necessary options

        Returns:
            Preprocessor:   Preprocessor object

        '''

        scaling = options['scale_images']
        return cls(scaling)

    def __init__(self, scaling):
        ''' Initialize a segmentation coordinator

        Args:
            scaling (bool): Scaling of images
        '''

        self.scaling = scaling

    def augment_data(self, path_to_dicom_data, output_dir):
        ''' Augment data

        Compute segmentation of lungs and airways based on dicom image data

        Args:
            path_to_dicom_data (str): Path to dicom data of chest CT image
            ouput_dir (str):          Path for writing created STL files

        '''       
        log.info('YELLOW', 'Augmenting data')

