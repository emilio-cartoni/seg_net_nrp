from cProfile import label
from locale import locale_alias
import torch.utils.data as data
import numpy as np
import torchvision.transforms as IT
import torchvision.transforms.functional as TF
import os
import random
import torch
from PIL import Image, ImageOps
DATASET_MEAN = [0.00, 0.00, 0.00]  # use this for data between 0.0 and 1.0
DATASET_STD = [1.00, 1.00, 1.00]  # use this for data between 0.0 and 1.0


class Multi_Dataset(data.Dataset):
    def __init__(self, sample_dir, label_dir, n_frames, n_classes, augmentation,
                 remove_ground, sequence_ids, label_categories):
        super(Multi_Dataset, self).__init__()
        self.sample_dir = sample_dir
        self.label_dir = label_dir
        self.n_frames = n_frames
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.remove_ground = remove_ground
        self.sequence_ids = sequence_ids
        self.label_categories = label_categories

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
        return len(self.sequence_ids)

    def __getitem__(self, index):
        '''Worst function ever created
        Arguments: index - index of the fuck
        '''
        sample_subdir = os.path.join(self.sample_dir, f'{self.sequence_ids[index]:05}')
        label_subdir = os.path.join(self.label_dir, f'{self.sequence_ids[index]:05}')
        image_path_list = [os.path.join(sample_subdir, p) for p in os.listdir(sample_subdir)]
        label_path_dict = {l: [os.path.join(label_subdir, self.label_categories[l], p)
                            for p in os.listdir(os.path.join(label_subdir, self.label_categories[l]))]
                            for l in self.label_categories.keys()}
        n_frames = min(self.n_frames, len(image_path_list) - 1)  # - 1 or not (?)
        first_frame = random.choice(np.arange(len(image_path_list) - n_frames))
        last_frame = first_frame + n_frames
        image_path_list = image_path_list[first_frame:last_frame]
        label_path_dict = {k: v[first_frame:last_frame] for (k, v) in label_path_dict.items()}

        image_list = []
        for image_path in image_path_list:
            # with Image.open(os.path.join(self.sample_dir, image_path)) as image:
            #     image_list.append(image)
            image_list.append(Image.open(os.path.join(self.sample_dir, image_path)))
        label_list = [np.zeros((self.n_classes,) + image_list[0].size) for n in range(n_frames)]
        for label_id, label_path_list in label_path_dict.items():
            for frame_id, label_path in enumerate(label_path_list):
                # with Image.open(os.path.join(self.label_dir, label_path)) as mask:
                #     label_list[frame_id][label_id] = ImageOps.grayscale(mask)
                mask = Image.open(os.path.join(self.label_dir, label_path))
                label_list[frame_id][label_id] = ImageOps.grayscale(mask)
        samples_and_labels = self.transform(image_list, label_list)
        samples = torch.stack([item[0] for item in samples_and_labels], dim=-1)
        labels = torch.stack([item[1] for item in samples_and_labels], dim=-1)
        # if not self.remove_ground:
        #     do_something like add one label map full of zeros at channel zero
        return samples, labels


def get_multi_dataloaders(root_dir, train_valid_ratio, batch_size_train,
                          batch_size_valid, n_frames, augmentation, remove_ground):
    # Dataset info
    sample_dir = os.path.join(root_dir, 'camera_images')
    label_dir = os.path.join(root_dir, 'segment_images')
    n_sequences = len(os.listdir(sample_dir))
    train_sequence_ids = list(range(0, int(n_sequences * train_valid_ratio)))
    valid_sequence_ids = list(range(int(n_sequences * train_valid_ratio), n_sequences))
    label_categories = {0: 'adjustable_wrench',
                        1: 'chips_can',
                        2: 'flat_screwdriver',
                        3: 'phillips_screwdriver',
                        4: 'hammer',
                        5: 'power_drill',
                        6: 'scissors',
                        7: 'timer'}
    n_classes = len(label_categories.keys()) + int(not remove_ground)

    # Training dataloader
    train_dataset = Multi_Dataset(sample_dir, label_dir, n_frames, n_classes, augmentation,
                                  remove_ground, train_sequence_ids, label_categories)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size_train,
                                       shuffle=True, pin_memory=True)

    # Validation dataloader
    valid_dataset = Multi_Dataset(sample_dir, label_dir, n_frames, n_classes, False,
                                remove_ground, valid_sequence_ids, label_categories)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=batch_size_valid,
                                       shuffle=True, pin_memory=True)

    # Return the dataloaders and number of classes to the computer
    return train_dataloader, valid_dataloader, n_classes
