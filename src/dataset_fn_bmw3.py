import torch.utils.data as data
import numpy as np
import torchvision.transforms as IT
import torchvision.transforms.functional as TF
import os
import h5py
import random
import torch
from PIL import Image
DATASET_MEAN = [0.00, 0.00, 0.00]  # use this to have data between 0.0 and 1.0
DATASET_STD = [1.00, 1.00, 1.00]  # use this to have data between 0.0 and 1.0


class BMW_Dataset(data.Dataset):
    def __init__(self, h5_path, n_frames, augmentation, remove_ground):
        super(BMW_Dataset, self).__init__()
        with h5py.File(name=h5_path, mode='r') as f:
            self.samples = np.array(f['samples'])  # np.array to load directly
            self.labels = np.array(f['labels'])  # np.array to load directly
            self.n_frames_max = self.samples.shape[-1]
        self.n_frames = n_frames  # idea: variable n_frames for each batch?
        self.augmentation = augmentation
        self.remove_ground = remove_ground
    
    @staticmethod
    def apply_jitter(image, fn_order, bri, con, sat, hue):
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
        sample_list = [samples[..., t] for t in range(self.n_frames)]
        label_list = [labels[..., t] for t in range(self.n_frames)]
        crop_params = IT.RandomCrop.get_params(torch.tensor(sample_list[0]), output_size=(240, 240))
        resize = IT.Resize(size=(320, 320), interpolation=IT.InterpolationMode.NEAREST)
        normalize = IT.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        jit = IT.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
        jit_params = IT.ColorJitter.get_params(jit.brightness, jit.contrast,
                                               jit.saturation, jit.hue)
        samples_and_labels = []
        for sample, label in zip(sample_list, label_list):
            sample = torch.tensor(sample).to(torch.float32).div(255)
            label = torch.tensor(label).to(torch.float32)
            # sample = TF.to_tensor(sample)
            # label = TF.to_tensor(label)
            if self.augmentation:  # only for training
                sample = self.apply_jitter(sample, *jit_params)
                # sample = self.apply_jitter(TF.crop(sample, *crop_params), *jit_params)
                # label = TF.crop(label, *crop_params)
            sample = normalize(resize(sample))
            label = resize(label)
            samples_and_labels.append((sample, label))

        samples = torch.stack([item[0] for item in samples_and_labels], dim=-1)
        labels = torch.stack([item[1] for item in samples_and_labels], dim=-1)
        return samples, labels

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        samples = self.samples[index]
        labels = self.labels[index]
        if self.n_frames < self.n_frames_max:  # todo: check this works
            n_frames = min(self.n_frames, self.n_frames_max - 1)
            first_frame = random.choice(range(self.n_frames_max - n_frames))
            samples = samples[..., first_frame:first_frame + n_frames]
            labels = labels[..., first_frame:first_frame + n_frames]
        
        samples, labels = self.transform(samples, labels)  # torch tensors
        # if not self.remove_ground:
        #     do_something
        return samples, labels


def get_bmw_dataloaders(root_dir, train_valid_ratio, batch_size_train, batch_size_valid,
                        n_frames, augmentation, remove_ground):
    # Dataset info
    train_path = os.path.join(root_dir, 'train.hdf5')
    valid_path = os.path.join(root_dir, 'valid.hdf5')
    n_classes = 5

    # Training dataloader
    print('Loading training dataset (takes time)')
    train_dataset = BMW_Dataset(train_path, n_frames, augmentation, remove_ground)
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True)

    # Validation dataloader
    print('Loading validation dataset (takes a bit less time)')
    valid_dataset = BMW_Dataset(valid_path, n_frames, False, remove_ground)
    valid_dataloader = data.DataLoader(
        valid_dataset, batch_size=batch_size_valid, shuffle=True, pin_memory=True)

    # Return the dataloaders and number of classes to the computer
    return train_dataloader, valid_dataloader, n_classes
