import torch.utils.data as data
import numpy as np
import torchvision.transforms as IT
import torchvision.transforms.functional as TF
import os
import random
import torch
from PIL import Image
DATASET_MEAN = [0.00, 0.00, 0.00]  # use this to have data between 0.0 and 1.0
DATASET_STD = [1.00, 1.00, 1.00]  # use this to have data between 0.0 and 1.0


class BMW_Dataset(data.Dataset):
    def __init__(self, sample_dir, label_dir, sequence_indices, n_frames,
                 augmentation, remove_ground):
        super(BMW_Dataset, self).__init__()
        self.sample_dir = sample_dir
        self.label_dir = label_dir
        self.sequence_indices = sequence_indices
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

    def transform(self, image_list, label_list):
        crop_params = IT.RandomCrop.get_params(image_list[0], output_size=(240, 240))
        resize = IT.Resize(size=(256, 256), interpolation=IT.InterpolationMode.NEAREST)
        normalize = IT.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        jit = IT.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
        jit_params = IT.ColorJitter.get_params(jit.brightness, jit.contrast,
                                               jit.saturation, jit.hue)
        transformed_images_and_labels = []
        for image, label in zip(image_list, label_list):
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)
            if self.augmentation:  # only for training
                image = self.apply_jitter(TF.crop(image, *crop_params), *jit_params)
                label = TF.crop(label, *crop_params)
            image = normalize(resize(image))
            label = resize(label)
            transformed_images_and_labels.append((image, label))
        return transformed_images_and_labels

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, index):
        sample_subpath = f'cam_img_{self.sequence_indices[index]:05}'
        label_subpath = f'seg_img_{self.sequence_indices[index]:05}'
        image_path_list = [path for path in os.listdir(self.sample_dir) if sample_subpath in path]
        label_path_list = [path for path in os.listdir(self.label_dir) if label_subpath in path]
        n_frames = min(self.n_frames, len(image_path_list) - 1)  # - 1 or not (?)
        first_frame = random.choice(np.arange(len(image_path_list) - n_frames))
        last_frame = first_frame + n_frames
        image_path_list = image_path_list[first_frame:last_frame]
        label_path_list = label_path_list[first_frame:last_frame]

        image_list, label_list = [], []
        for (image_path, label_path) in zip(image_path_list, label_path_list):
            with Image.open(os.path.join(self.sample_dir, image_path)) as image:
                label = np.load(os.path.join(self.label_dir, label_path))
                image_list.append(image)
                label_list.append(label)

        samples_and_labels = self.transform(image_list, label_list)
        samples = torch.stack([item[0] for item in samples_and_labels], dim=-1)
        labels = torch.stack([item[1] for item in samples_and_labels], dim=-1)
        # if not self.remove_ground:
        #     do_something
        return samples, labels


def get_bmw_dataloaders(root_dir, train_valid_ratio, batch_size_train, batch_size_valid,
                        n_frames, augmentation, remove_ground):
    # Dataset info
    sample_dir = os.path.join(root_dir, 'camera_images')
    label_dir = os.path.join(root_dir, 'segment_masks')
    n_indices = len(set([path[8:13] for path in os.listdir(sample_dir)]))  # set does it unique
    train_indices = list(range(0, int(n_indices * train_valid_ratio)))
    valid_indices = list(range(int(n_indices * train_valid_ratio), n_indices))
    n_classes = 5

    # Training dataloader
    train_dataset = BMW_Dataset(sample_dir, label_dir, train_indices, n_frames, augmentation, remove_ground)
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True)

    # Validation dataloader
    valid_dataset = BMW_Dataset(sample_dir, label_dir, valid_indices, n_frames, False, remove_ground)
    valid_dataloader = data.DataLoader(
        valid_dataset, batch_size=batch_size_valid, shuffle=True, pin_memory=True)

    # Return the dataloaders and number of classes to the computer
    return train_dataloader, valid_dataloader, n_classes
