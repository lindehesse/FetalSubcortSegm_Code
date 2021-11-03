
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import warnings
import torch.nn.functional as F
from utils import convert_one_hot
import torchio as tio
from pathlib import Path
warnings.filterwarnings('ignore', '.*output shape of zoom.*')


class MyDataset(Dataset):
    """
    Dataset to load the ultrasound volumes and their labels
    """

    def __init__(self, filenames, params, hor_flips=False, rotation_degrees=0,
                 trans_pixels=0,  scaling_factor=1):
        """ Initializes dataset class

        Args:
            filenames (list): filenames
            params (Param class): contains all settings
            hor_flips (bool, optional): apply horizontal flips as augmentation. Defaults to False.
            rotation_degrees (int, optional): apply rotation with n degrees. Defaults to 0.
            trans_pixels (int, optional): apply translation (num pixels). Defaults to 0.
            scaling_factor (int, optional): Apply scaling between 1/factor and factor. Defaults to 1.
        """
        self.filenames = filenames
        self.params = params

        # Augmentation values
        self.hor_flips = hor_flips
        self.rotation_degrees = rotation_degrees
        self.scaling_factor = scaling_factor
        self.trans_pixels = trans_pixels

    def __len__(self):
        """
        return number of filenames
        """

        return len(self.filenames)

    def __getitem__(self, idx):
        """
        idx: number of batch
        returns one data item
        """

        # Select data of these indices
        sample_name = self.filenames[idx]

        # Find corresponding targets
        im_array, target_array = self.find_sample(sample_name)

        # Convert to torch
        self.im_torch = torch.from_numpy(im_array).unsqueeze(0)
        self.target_torch = torch.from_numpy(target_array).unsqueeze(0)

        # Combine in torchio subject
        self.apply_augmentation()

        # Convert taregt to one hot
        one_hot = convert_one_hot(
            self.target_torch, num_classes=len(self.params.structures)+1)

        return self.im_torch, one_hot, Path(sample_name).stem

    def apply_augmentation(self):
        # Get new bounds of segmentation based on GW
        if self.scaling_factor != 1:
            lower_scale = 1/self.scaling_factor
            upper_scale = self.scaling_factor
        else:
            lower_scale = 1
            upper_scale = 1

        subject = tio.Subject(im=tio.ScalarImage(tensor=self.im_torch),
                              labels=tio.LabelMap(tensor=self.target_torch))

        # Make affine transform
        transform = tio.transforms.RandomAffine(scales=[lower_scale, upper_scale],
                                                degrees=self.rotation_degrees,
                                                translation=self.trans_pixels,
                                                isotropic=True)

        # Add horizontal flips
        if self.hor_flips:
            flip = tio.RandomFlip(axes=1)
            transform = tio.Compose([flip, transform])

        transformed_im = transform(subject)

        # Attribute transformed images to class
        self.im_torch = transformed_im.im.data
        self.target_torch = transformed_im.labels.data

    def find_sample(self, sample_name):
        """ find targets for training (gt-masks)

        Args:
            sample_name (str): path of sample

        Returns:
            array: image volume (hxwxd)
            array: array of targets (multi-label): h x w x d (exclusive labels)
        """
        # Read image
        im_array = sitk.GetArrayFromImage(
            sitk.ReadImage(sample_name)).astype(np.float32)

        # Ensure im values are between 0 and 255
        im_array = np.clip(im_array, 0, 255)

        # find targets
        targets = []
        for struc in self.params.structures:
            label_name = sample_name.replace(self.params.data_params.volume_names,
                                             struc)

            label = sitk.ReadImage(label_name)
            targets.append(sitk.GetArrayFromImage(label))

        assert(len(targets) == len(self.params.structures))

        # returns  h x w x d (exclusive labels)
        target_array = np.zeros_like(im_array, dtype=np.int64)
        for i in range(len(targets)):
            target_array[targets[i] == 1] = i + 1

        return im_array, target_array


class MyDataset_Preloader(MyDataset):
    """
    Dataset to load the ultrasound images and their labels.
    In this function all data is preloaded into memory before training, 
    possibly making training faster
    """

    def __init__(self, filenames, params, **kwargs):
        """ Initializes preloader
        Args:
            filenames (list): list of full filenames
            params (Parameter object): contains all settings
        """

        super(MyDataset_Preloader, self).__init__(filenames, params, **kwargs)

        # Initializes variables
        self.images = []
        self.targets = []

        # Do actual preloading
        self.load_data_inmemory()

    def __getitem__(self, idx):
        """
        idx: number of batch
        returns one data item
        """
        # Obtain data from preloaded batch
        im_array = self.images[idx]
        target_array = self.targets[idx]
        sample_name = self.filenames[idx]

        # Transform to pytorch tensors
        self.im_torch = torch.from_numpy(im_array).unsqueeze(0)
        self.target_torch = torch.from_numpy(target_array).unsqueeze(0)

        # Apply Augmentation
        self.apply_augmentation()

        # transform targets to one-hot
        one_hot = convert_one_hot(
            self.target_torch, len(self.params.structures)+1)

        return self.im_torch, one_hot, Path(sample_name).stem

    def load_data_inmemory(self):
        """ Loads all data into memory
        """

        # load first sample to get dimensions
        sample_name = self.filenames[0]
        im_array = sitk.GetArrayFromImage(
            sitk.ReadImage(sample_name)).astype(np.float32)

        image_size = im_array.shape

        # make zero arrays
        images = np.zeros(
            (len(self.filenames), image_size[0], image_size[1], image_size[2]),
            dtype=np.float32)
        targets = np.zeros_like(images, dtype=np.int64)

        # Load all data into memory
        for num, sample_name in enumerate(self.filenames):

            # Find sample and target
            im_array, target_array = self.find_sample(sample_name)

            # Assign to dataset
            targets[num, :, :, :] = target_array
            images[num, :, :, :] = im_array

        self.images = images
        self.targets = targets


class MyManualTestset(MyDataset):
    """
    Dataset to load in manually corrected labels

    """

    def __init__(self, filenames, params, **kwargs):
        """ Seperate class for loading in test set images
        Args:
            filenames (list[str]): list of full path names for images
            params (Dataclass): Parameter object of all settings
        """
        # Make functions of parents class available
        super(MyManualTestset, self).__init__(filenames, params, **kwargs)

        self.filenames = filenames
        self.params = params
        self.samples_list = []
        self.images = []
        self.targets = []

    def __getitem__(self, idx):
        """   Obtains 1 datasample, currently equivalent to normal dataset
        but by seperating it new things can easily be added to this class only for test set

        Args:
            idx (int): batch number

        Returns:
            tensor, tensor, str: image, mask, imname
        """

        # Select data of these indices
        sample_name = self.filenames[idx]

        # Find sample
        im_array, target_array = self.find_sample(sample_name)

        # get sample name without extension
        im_name = Path(sample_name).stem
       
        # Convert to torch
        self.im_torch = torch.from_numpy(
            im_array).unsqueeze(0)   # [1, H, W, D]
        self.target_torch = torch.from_numpy(
            target_array).unsqueeze(0)  # [1, H, W, D]

        # transform targets to one-hot # [C, H, W, D]
        one_hot = convert_one_hot(
            self.target_torch, len(self.params.structures)+1)

        return self.im_torch, one_hot, im_name


