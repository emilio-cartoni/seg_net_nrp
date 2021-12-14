import torch.utils.data as data
import numpy as np
import torchvision.transforms as IT
import torchvision.transforms.functional as TF
import os
import random
import torch
from PIL import Image
# DATASET_MEAN = [0.38, 0.38, 0.38]
# DATASET_STD = [0.28, 0.28, 0.28]
DATASET_MEAN = [0.00, 0.00, 0.00]  # use this to have data between 0.0 and 1.0
DATASET_STD = [1.00, 1.00, 1.00]  # use this to have data between 0.0 and 1.0


class Mots_Dataset(data.Dataset):
    def __init__(self, root_dir, subfolders, n_classes, n_frames, remove_ground):
        self.sample_root = os.path.join(root_dir, 'training', 'image_02')
        self.label_root = os.path.join(root_dir, 'instances')
        self.subfolders = subfolders
        self.n_classes = n_classes
        self.n_frames = n_frames  # idea: variable n_frames for each batch?
        self.remove_ground = remove_ground

    def transform(self, list_of_images):
        """
        Parameters
        ----------
        list_of_images : list
        Must be a list of tuples, where each tuple inside the list contains both a training image
        and its corresponding label.

        Returns
        -------
        transformed_images : list
        A list of tuples, where each tuple is a set of (image, label) for a specific timepoint.
        """
        transformed_images = []
        resize = IT.Resize(size=(80, 256), interpolation=IT.InterpolationMode.NEAREST)
        # resize = IT.Resize(size=(100, 320), interpolation=IT.InterpolationMode.NEAREST)
        # resize = IT.Resize(size=(160, 512), interpolation=IT.InterpolationMode.NEAREST)
        normalize = IT.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        i, j, h, w = IT.RandomCrop.get_params(list_of_images[0][0], output_size=(320, 1024))
        for image, label in list_of_images:
            image, label = TF.crop(image, i, j, h, w), TF.crop(label, i, j, h, w)
            image, label = resize(image), resize(label)
            image, label = TF.to_tensor(np.array(image)), TF.to_tensor(np.array(label))
            image = normalize(image)
            transformed_images.append((image, label))
        return transformed_images

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, index):
        sample_dir = os.path.join(self.sample_root, self.subfolders[index])  # f'{index:04}')
        label_dir = os.path.join(self.label_root, self.subfolders[index])  # f'{index:04}')
        sample_paths = [os.path.join(sample_dir, p) for p in os.listdir(sample_dir)]
        label_paths = [os.path.join(label_dir, p) for p in os.listdir(label_dir)]
        n_frames = min(self.n_frames, len(sample_paths) - 1)
        first_frame = random.choice(np.arange(len(sample_paths) - n_frames))
        sample_frames = sample_paths[first_frame:first_frame + n_frames]
        label_frames = label_paths[first_frame:first_frame + n_frames]

        samples_and_labels = []
        for i in range(n_frames):
            sample = Image.open(os.path.join(sample_dir, sample_frames[i]))
            label = Image.open(os.path.join(label_dir, label_frames[i]))
            samples_and_labels.append((sample, label))
        samples_and_labels = self.transform(samples_and_labels)

        samples = torch.stack([item[0] for item in samples_and_labels], dim=-1)
        labels = torch.stack([item[1] for item in samples_and_labels], dim=-1)
        labels = torch.div(labels, 1000, rounding_mode='floor')
        labels[labels == 10] = self.n_classes - int(not self.remove_ground)
        labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=self.n_classes + int(self.remove_ground))
        labels = torch.movedim(labels, -1, 1)
        labels = torch.squeeze(labels, dim=0)
        labels = labels.type(torch.FloatTensor)
        if self.remove_ground:
            labels = labels[1:]
        return samples, labels


class SQMDataset(data.Dataset):
    def __init__(self, root_dir, testing_folders, n_classes, remove_ground):
        self.sample_root = os.path.join(root_dir, 'testing')
        self.testing_folders = testing_folders
        self.n_classes = n_classes
        self.remove_ground = remove_ground

    def transform(self, list_of_images):
        """
        Parameters
        ----------
        list_of_images : list
        Must be a list of tuples, where each tuple inside the list contains both a training image
        and its corresponding label.

        Returns
        -------
        transformed_images : list
        A list of tuples, where each tuple is a set of (image, label) for a specific timepoint.
        """
        transformed_images = []
        normalize = IT.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        for image, label in list_of_images:
            image, label = TF.to_tensor(np.array(image)), TF.to_tensor(np.array(label))
            image = normalize(image)
            transformed_images.append((image, label))
        return transformed_images

    def __len__(self):
        return len(self.testing_folders)

    def __getitem__(self, index):
        sample_dir = os.path.join(self.sample_root, f'{index:04}')
        sample_paths = [os.path.join(sample_dir, p) for p in os.listdir(sample_dir)]
        n_frames = len(sample_paths) - 1
        first_frame = random.choice(np.arange(len(sample_paths) - n_frames))
        sample_frames = sample_paths[first_frame:first_frame + n_frames]

        samples_and_labels = []
        for i in range(n_frames):
            sample = Image.open(os.path.join(sample_dir, sample_frames[i]))
            samples_and_labels.append((sample, sample))
        samples_and_labels = self.transform(samples_and_labels)

        samples = torch.stack([item[0] for item in samples_and_labels], dim=-1)
        labels = torch.stack([item[1] for item in samples_and_labels], dim=-1)
        labels = torch.div(labels, 1000, rounding_mode='floor')
        labels[labels == 10] = self.n_classes - int(not self.remove_ground)
        labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=self.n_classes + int(self.remove_ground))
        labels = torch.movedim(labels, -1, 1)
        #labels = torch.squeeze(labels, dim=0)
        labels = labels[0]
        labels = labels.type(torch.FloatTensor)
        if self.remove_ground:
            labels = labels[1:]

        return samples, labels


def get_segmentation_dataloaders_mots(
    root_dir, train_valid_ratio, batch_size_train, batch_size_valid, n_frames, augmentation, n_classes, remove_ground):
  
    # Data train valid
    training_folders = os.listdir(os.path.join(root_dir, 'training', 'image_02'))
    train_subfolders = training_folders[int((1.0 - train_valid_ratio) * len(training_folders)):]
    valid_subfolders = training_folders[:int((1.0 - train_valid_ratio) * len(training_folders))]

    # Training dataloader
    train_dataset = Mots_Dataset(root_dir, train_subfolders, n_classes, n_frames, remove_ground)
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True)

    # Validation dataloader
    valid_dataset = Mots_Dataset(root_dir, valid_subfolders, n_classes, n_frames, remove_ground)
    valid_dataloader = data.DataLoader(
        valid_dataset, batch_size=batch_size_valid, shuffle=True, pin_memory=True)
    
    # label_mean = torch.zeros((n_classes,))
    # for sample, label in train_dataloader:
    #     label_mean = label_mean + torch.mean(label, dim=(0, 2, 3, 4))
    # for sample, label in valid_dataloader:
    #     label_mean = label_mean + torch.mean(label, dim=(0, 2, 3, 4))
    # label_mean = label_mean / (len(train_dataloader) + len(valid_dataloader))
    # print(label_mean)
    # exit()

    # Return the dataloaders to the computer
    return train_dataloader, valid_dataloader


def get_SQM_dataloaders(root_dir, n_classes, remove_ground):
    testing_folders = os.listdir(os.path.join(root_dir, 'testing'))
    testing_subfolders = [folder for folder in testing_folders]

    testing_dataset = SQMDataset(root_dir, testing_subfolders, n_classes, remove_ground)
    testing_dataloader = data.DataLoader(testing_dataset, batch_size=1, shuffle=False)
    return testing_dataloader
