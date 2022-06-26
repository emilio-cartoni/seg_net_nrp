import torch.utils.data as data
import torchvision.transforms as IT
import torchvision.transforms.functional as TF
import os
import h5py
import random
import torch
import numpy as np
DATASET_MEAN = [0.00, 0.00, 0.00]  # use this to have data between 0.0 and 1.0
DATASET_STD = [1.00, 1.00, 1.00]  # use this to have data between 0.0 and 1.0


class Handover_Dataset(data.Dataset):
    def __init__(self, h5_path, load_on_ram, n_frames,
                 n_classes, augmentation, remove_ground):
        ''' Initialize a dataset with the BMW environment (flying skeleton).
        
        Parameters
        ----------
        h5_path : str
            Path to the h5 file containing the dataset.
        n_frames : int
            Number of frames to use for each sample sequence.
        n_classes : int
            Number of classes in the dataset.
        augmentation : bool
            Whether to augment the dataset.
        remove_ground : bool
            Whether to remove the ground from the segmentation masks.

        Returns
        -------
        None.
        '''
        super(Handover_Dataset, self).__init__()
        f = h5py.File(name=h5_path, mode='r')
        if load_on_ram:
            self.samples = np.array(f['samples'])
            self.labels = np.array(f['labels'])
        else:
            self.samples = f['samples']
            self.labels = f['labels']
        self.n_frames_max = self.samples.shape[-1]
        self.n_frames = n_frames
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.remove_ground = remove_ground
    
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
        sample_list = [samples[..., t] for t in range(self.n_frames)]
        label_list = [labels[..., t] for t in range(self.n_frames)]
        crop_params = IT.RandomCrop.get_params(torch.tensor(sample_list[0]), output_size=(240, 240))
        resize = IT.Resize(size=(128, 128), interpolation=IT.InterpolationMode.NEAREST)
        normalize = IT.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        jit = IT.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
        jit_params = IT.ColorJitter.get_params(jit.brightness, jit.contrast,
                                               jit.saturation, jit.hue)
        samples_and_labels = []
        for sample, label in zip(sample_list, label_list):
            sample = torch.tensor(sample).to(torch.float32).div(255)
            label = torch.tensor(label).to(torch.long).clip(max=self.n_classes)
            if self.augmentation:  # only for training
                sample = self.apply_jitter(TF.crop(sample, *crop_params), *jit_params)
                label = TF.crop(label, *crop_params)
            sample = normalize(resize(sample))
            label = resize(label)
            samples_and_labels.append((sample, label))

        samples = torch.stack([item[0] for item in samples_and_labels], dim=-1)
        labels = torch.stack([item[1] for item in samples_and_labels], dim=-1)
        return samples, labels

    def __len__(self):
        ''' Return the number of samples in the dataset.
        
        Returns
        -------
        int
            Number of samples in the dataset.
        '''
        return self.samples.shape[0]
    
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
        samples = self.samples[index]
        labels = self.labels[index]
        if self.n_frames < self.n_frames_max:
            n_frames = min(self.n_frames, self.n_frames_max - 1)
            first_frame_choices = range(self.n_frames_max - n_frames)
            first_frame = random.choice(first_frame_choices)
            samples = samples[..., first_frame:first_frame + n_frames]
            labels = labels[..., first_frame:first_frame + n_frames]
        
        samples, labels = self.transform(samples, labels)  # torch tensors
        # if not self.remove_ground:
        #     labels = labels.sum(dim=-1) then do something like this
        return {'samples': samples, 'labels': labels}


def handover_dl(mode, data_dir, batch_size, num_frames, num_classes,
                drop_last, num_workers, remove_ground=True):
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

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        Dataloader for the training or validation set.
        
    '''
    
    # Dataset info
    data_path = os.path.join(data_dir, f'{mode}.hdf5')
    augmentation = (mode == 'train')
    shuffle = (mode == 'train')
    load_on_ram = False  # (mode == 'train')
    
    # Build dataset class
    dataset = Handover_Dataset(data_path,
                               load_on_ram,
                               num_frames,
                               num_classes,
                               augmentation,
                               remove_ground)
                               
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
