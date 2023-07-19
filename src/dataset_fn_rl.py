import torch.utils.data as data
import torchvision.transforms as IT
import torchvision.transforms.functional as TF
import os
import random
import torch
import numpy as np
from PIL import Image
import re
DATASET_MEAN = [0.00, 0.00, 0.00]  # use this to have data between 0.0 and 1.0
DATASET_STD = [1.00, 1.00, 1.00]  # use this to have data between 0.0 and 1.0


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


class Rob_Dataset(data.Dataset):
    def __init__(self, data_dir, data_seqdirs, n_frames, n_classes, augmentation, packbits):
        ''' Initialize a dataset with the BMW environment (flying skeleton).
        
        Parameters
        ----------
        data_dir : str
            Path to the dataset directory.
        data_subdirs : list of str
            List of subdirectories used to load images and segmentation masks.
        n_frames : int
            Number of frames to use for each sample sequence.
        n_classes : int
            Number of classes in the dataset.
        augmentation : bool
            Whether to augment the dataset.
        remove_ground : bool
            Whether to remove the ground from the segmentation masks.
        packbits : bool
            Whether packbits compression is used in the segmentation mask data.

        Returns
        -------
        None.
        '''
        super(Rob_Dataset, self).__init__()
        self.data_dir = data_dir
        self.img_subdir = "img"
        self.seg_subdir = "mask"
        self.n_frames = n_frames
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.packbits = packbits

        sample_seq_dir = os.path.join(self.data_dir, self.img_subdir)
        self.num_img_seq_dirs = len(os.listdir(sample_seq_dir))

        sample_seq_dir = os.path.join(sample_seq_dir, os.listdir(sample_seq_dir)[0])    # First sequence dir
        sample_file = os.path.join(sample_seq_dir, os.listdir(sample_seq_dir)[0])       # First image in sequence dir

        self.n_frames_max = len(os.listdir(sample_seq_dir))
        with Image.open(sample_file) as img:
            self.sample_dims = np.array(img).shape
    
    @staticmethod
    def apply_jitter(image, fn_order, bri, con, sat, hue):
        ''' Apply jitter to an image.
        
        Parameters
        ----------
        image : torch.Tensor
            Image to which jitter is applied.
        fn_order : int
            Order in which jitter functions are applied.
        bri : float
            Brightness jitter.
        con : float
            Contrast jitter.
        sat : float
            Saturation jitter.
        hue : float
            Hue jitter.
        
        Returns
        -------
        image : torch.Tensor
            Transformed image.
        '''
        for fn_id in fn_order:
            if fn_id == 0 and bri is not None:
                image = TF.adjust_brightness(image, bri)
            elif fn_id == 1 and con is not None:
                image = TF.adjust_contrast(image, con)
            elif fn_id == 2 and sat is not None:
                image = TF.adjust_saturation(image, sat)
            elif fn_id == 3 and hue is not None:
                image = TF.adjust_hue(image, hue)
        return image

    def transform(self, samples, labels):
        ''' Transform the samples and labels.
        
        Parameters
        ----------
        samples : np.array
            Samples to transform.
        labels : np.array
            Labels to transform.

        Returns
        -------
        samples : torch.Tensor
            Transformed samples.
        labels : torch.Tensor
            Transformed labels.
        '''
        samples = samples.transpose((2, 0, 1, 3))
        labels = labels.transpose((2, 0, 1, 3))
        sample_list = [samples[..., t] for t in range(self.n_frames)]
        label_list = [labels[..., t] for t in range(self.n_frames)]
        crop_params = IT.RandomCrop.get_params(torch.tensor(sample_list[0]),
                                               output_size=(240, 240))
        resize = IT.Resize(size=(128, 128),
                           interpolation=IT.InterpolationMode.NEAREST)
        normalize = IT.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        jit = IT.ColorJitter(brightness=0.4,
                             contrast=0.4,
                             saturation=0.4,
                             hue=0.4)
        jit_params = IT.ColorJitter.get_params(jit.brightness,
                                               jit.contrast,
                                               jit.saturation,
                                               jit.hue)
        samples_and_labels = []
        for sample, label in zip(sample_list, label_list):
            sample = torch.tensor(sample).to(torch.float32).div(255)
            label = torch.tensor(label).to(torch.long).clip(max=self.n_classes)
            if self.augmentation:  # only for training
                sample = self.apply_jitter(TF.crop(sample, *crop_params),
                                           *jit_params)
                label = TF.crop(label, *crop_params)
            sample = normalize(resize(sample))
            label = resize(label)
            samples_and_labels.append((sample, label))

        samples = torch.stack([item[0] for item in samples_and_labels], dim=-1)
        labels = torch.stack([item[1] for item in samples_and_labels], dim=-1)
        return samples, labels
    
    def load_mask(self, file_path):
        mask = np.load(file_path)
        if self.packbits:
            mask = np.unpackbits(mask, axis=1)[:, :self.sample_dims[1]]
        return mask

    def __len__(self):
        ''' Return the number of samples in the dataset.
        
        Returns
        -------
        int
            Number of samples in the dataset.
        '''
        return self.num_img_seq_dirs #len(self.num_seq_dirs)
    
    def __getitem__(self, index):
        ''' Return a sample sequence and the corresponding segmentation masks.
        
        Parameters
        ----------
        index : int
            Index (in the dataset) of the sample sequence to return.

        Returns
        -------
        sample : torch.Tensor
            Sample (sequence of images).
        label : torch.Tensor
            Segmentation label.
        '''
        img_seq_dir = os.path.join(self.data_dir, self.img_subdir, str(index))
        mask_seq_dir = os.path.join(self.data_dir, self.seg_subdir, str(index))

        img_files = [os.path.join(img_seq_dir, path)\
            for path in os.listdir(img_seq_dir)]
        mask_files = [os.path.join(mask_seq_dir, path)\
            for path in os.listdir(mask_seq_dir) if path.endswith(".npy")]
        img_files.sort(key=natural_keys); mask_files.sort(key=natural_keys)  # ubuntu vs windows

        masks = np.stack([self.load_mask(f) for f in mask_files], axis=-1)
        imgs = np.zeros(masks.shape[:2] + (3, self.n_frames))
        for t, f in enumerate(img_files[:self.n_frames]):
            with Image.open(f) as img:
                imgs[..., t] = np.array(img)
        imgs, masks = self.transform(imgs, masks)  # torch tensors
        return {'samples': imgs, 'labels': masks}


def rob_dl(mode, data_dir, batch_size, num_frames, num_classes,
           drop_last, num_workers, packbits, tr_ratio=0.8):
    ''' Get the dataloaders for the BMW dataset.
    
    Parameters
    ----------
    mode : str
        Whether to load the training or validation set.
    data_dir : str
        Path to the dataset.
    batch_size : int
        Batch size used in the training / validation set.
    num_frames : int
        Number of frames to use for each sample sequence.
    num_classes : int
        Number of classes in the segmentation dataset.
    drop_last : bool
        Whether to drop the last batch if it is not full.
    num_workers : int
        Number of workers to use for the dataloader.
    remove_ground : bool
        Whether to remove the ground from the segmentation masks.
    tr_ratio : float
        Ratio of the training set to the validation set.

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        Dataloader for the training or validation set.
        
    '''
    
    # Dataset info
    # sample_subdir = os.listdir(os.path.join(data_dir, 'ci'))
    # random.shuffle(sample_subdir)
    # if mode == 'train':
    #     sample_subdir = sample_subdir[:int(len(sample_subdir) * tr_ratio)]
    # elif mode == 'valid':
    #     sample_subdir = sample_subdir[int(len(sample_subdir) * tr_ratio):]
    augmentation = (mode == 'train')
    shuffle = (mode == 'train')
    
    # Build dataset class
    dataset = Rob_Dataset(data_dir,
                          None, # sample_subdir,
                          num_frames,
                          num_classes,
                          augmentation,
                          packbits)
                               
    # Adapt batch size
    if mode == 'valid':
        batch_size = batch_size * 4
    batch_size = min(batch_size, len(dataset))
    
    # Build dataloader
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 drop_last=drop_last,
                                 num_workers=num_workers)
    
    # Return the dataloader
    return dataloader
